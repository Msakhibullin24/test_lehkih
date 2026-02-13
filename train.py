"""
Обучение 3D U-Net для детекции узелков в лёгких (LUNA16).
Полная поддержка CUDA — автоматически использует GPU если доступен.
Автоматически находит все subset* папки из data/.

Структура проекта:
    LUNA16_Project/
    ├── data/              ← subset0..subset6 внутри
    ├── masks/             ← маски (генерирует prepare_masks.py)
    ├── seg-lungs/         ← маски лёгких (опционально)
    ├── annotations.csv
    ├── candidates.csv
    ├── prepare_masks.py
    └── train.py           ← ЭТОТ ФАЙЛ
"""

import os
import glob
import time
import torch
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    SpatialPadd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
)
import itk  # noqa: F401 — нужен для MONAI ITKReader
from monai.data import Dataset, DataLoader, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference


def main():
    # =========================================================================
    # НАСТРОЙКИ ПУТЕЙ
    # =========================================================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")           # subset'ы внутри data/
    MASK_DIR = os.path.join(BASE_DIR, "masks")           # маски узелков
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "luna16_unet.pth")

    LEARNING_RATE = 1e-4

    # =========================================================================
    # УСТРОЙСТВО — CUDA первым приоритетом
    # =========================================================================
    if not torch.cuda.is_available():
        print("⚠  CUDA не найдена, работаем на CPU (облегчённый режим)")
        device = torch.device("cpu")
        MAX_EPOCHS = 3
        BATCH_SIZE = 1
        PATCH_SIZE = (64, 64, 64)
        NUM_SAMPLES = 2
        NUM_WORKERS = 0
        CHANNELS = (16, 32, 64, 128)
        STRIDES = (2, 2, 2)
    else:
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"✓  CUDA найдена: {gpu_name} ({gpu_mem:.1f} GB)")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Параметры масштабируются под GPU-память
        if gpu_mem >= 20:       # A100 / RTX 4090 / A6000
            MAX_EPOCHS = 50
            BATCH_SIZE = 4
            PATCH_SIZE = (128, 128, 128)
            NUM_SAMPLES = 4
            NUM_WORKERS = 8
            CHANNELS = (32, 64, 128, 256, 512)
            STRIDES = (2, 2, 2, 2)
        elif gpu_mem >= 8:      # RTX 3060-3080 / RTX 4060-4070
            MAX_EPOCHS = 30
            BATCH_SIZE = 2
            PATCH_SIZE = (96, 96, 96)
            NUM_SAMPLES = 4
            NUM_WORKERS = 4
            CHANNELS = (16, 32, 64, 128, 256)
            STRIDES = (2, 2, 2, 2)
        else:                   # GTX 1060-1080 / RTX 2060
            MAX_EPOCHS = 20
            BATCH_SIZE = 1
            PATCH_SIZE = (64, 64, 64)
            NUM_SAMPLES = 2
            NUM_WORKERS = 4
            CHANNELS = (16, 32, 64, 128)
            STRIDES = (2, 2, 2)

    print(f"Устройство: {device}")
    print(f"Патчи: {PATCH_SIZE}, Batch: {BATCH_SIZE}, Эпохи: {MAX_EPOCHS}")

    # =========================================================================
    # СБОР ДАННЫХ — автоматический поиск всех subset* папок в data/
    # =========================================================================
    if not os.path.isdir(DATA_DIR):
        print(f"ОШИБКА: Не найдена папка {DATA_DIR}")
        return

    subset_dirs = sorted([
        os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
        if d.startswith("subset") and os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    print(f"Найдено subset-папок: {len(subset_dirs)} → "
          f"{[os.path.basename(d) for d in subset_dirs]}")

    data_dicts = []
    for sdir in subset_dirs:
        images = sorted(glob.glob(os.path.join(sdir, "*.mhd")))
        for img_path in images:
            uid = os.path.basename(img_path).replace(".mhd", "")
            mask_path = os.path.join(MASK_DIR, f"{uid}_mask.mhd")
            if os.path.exists(mask_path):
                data_dicts.append({"image": img_path, "label": mask_path})

    if not data_dicts:
        print("ОШИБКА: Не найдено пар (снимок, маска)!")
        print("Сначала запусти: python prepare_masks.py")
        return

    # Разделение: 80% train / 20% validation
    split = max(1, int(len(data_dicts) * 0.8))
    train_dicts = data_dicts[:split]
    val_dicts = data_dicts[split:]
    print(f"Найдено {len(data_dicts)} снимков с узелками")
    print(f"  Train: {len(train_dicts)}, Val: {len(val_dicts)}")

    # =========================================================================
    # ТРАНСФОРМАЦИИ
    # =========================================================================
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image", "label"]),

        # Нормализация КТ Hounsfield Units → [0, 1]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),

        # Паддинг до минимального размера патча (если скан меньше)
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),

        # Вырезаем 3D-патчи (весь скан не влезет в GPU)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=1, neg=1,
            num_samples=NUM_SAMPLES,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),

        # Аугментации
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
    ])

    # =========================================================================
    # ДАТАСЕТЫ И ЛОАДЕРЫ
    # =========================================================================
    train_ds = Dataset(data=train_dicts, transform=train_transforms)
    val_ds = Dataset(data=val_dicts, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
    )

    # =========================================================================
    # МОДЕЛЬ — 3D U-Net (на CUDA)
    # =========================================================================
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,           # Фон + Узелок
        channels=CHANNELS,
        strides=STRIDES,
        num_res_units=2,
        dropout=0.2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Модель: 3D U-Net, параметров: {total_params:,}")

    # Loss и Optimizer
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # AMP (Mixed Precision) — ускорение на GPU
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # =========================================================================
    # ЦИКЛ ОБУЧЕНИЯ
    # =========================================================================
    best_metric = -1.0

    print(f"\n{'='*60}")
    print(f"Начинаем обучение: {MAX_EPOCHS} эпох")
    print(f"Mixed Precision (AMP): {'Да' if use_amp else 'Нет'}")
    print(f"{'='*60}\n")

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        print(f"--- Эпоха {epoch + 1}/{MAX_EPOCHS} ---")

        # === TRAIN ===
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % 10 == 0:
                print(f"  Шаг {step}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / max(step, 1)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        # === VALIDATION ===
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs = val_data["image"].to(device, non_blocking=True)
                val_labels = val_data["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    val_outputs = sliding_window_inference(
                        val_inputs, PATCH_SIZE, sw_batch_size=4, predictor=model,
                    )

                val_outputs_onehot = torch.argmax(val_outputs, dim=1, keepdim=True)
                val_labels_onehot = (val_labels > 0).long()
                dice_metric(y_pred=val_outputs_onehot, y=val_labels_onehot)

            metric = dice_metric.aggregate().item()
            dice_metric.reset()

        elapsed = time.time() - epoch_start

        print(f"  Avg Loss: {avg_loss:.4f} | Val Dice: {metric:.4f} | "
              f"LR: {lr_now:.2e} | Время: {elapsed:.1f}с")

        # Сохраняем лучшую модель
        if metric > best_metric:
            best_metric = metric
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dice": best_metric,
            }, MODEL_SAVE_PATH)
            print(f"  ★ Лучшая модель сохранена (Dice: {best_metric:.4f})")

        # Очистка GPU-кэша
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # =========================================================================
    # ИТОГИ
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Обучение завершено!")
    print(f"Лучший Dice на валидации: {best_metric:.4f}")
    print(f"Модель сохранена: {MODEL_SAVE_PATH}")

    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Пик GPU памяти: {peak_mem:.2f} GB")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
