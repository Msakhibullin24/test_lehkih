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
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged,
    Spacingd, SpatialPadd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandGaussianNoised, RandScaleIntensityd, RandShiftIntensityd,
)
import SimpleITK as sitk
import itk  # noqa: F401 — нужен для MONAI ITKReader
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss
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
    VAL_INTERVAL = 2    # Валидация каждые N эпох
    SAVE_INTERVAL = 10  # Сохранение модели каждые N эпох

    # =========================================================================
    # УСТРОЙСТВО — CUDA первым приоритетом
    # =========================================================================
    if not torch.cuda.is_available():
        print("⚠  CUDA не найдена, работаем на CPU (облегчённый режим)")
        device = torch.device("cpu")
        num_gpus = 0
        MAX_EPOCHS = 3
        BATCH_SIZE = 1
        ACCUM_STEPS = 1
        PATCH_SIZE = (64, 64, 64)
        NUM_SAMPLES = 2
        NUM_WORKERS = 0
        CHANNELS = (16, 32, 64, 128)
        STRIDES = (2, 2, 2)
    else:
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Информация обо всех GPU
        total_gpu_mem = 0.0
        for i in range(num_gpus):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            total_gpu_mem += mem
            print(f"✓  GPU {i}: {name} ({mem:.1f} GB)")

        # Параметры масштабируются под СУММАРНУЮ GPU-память
        if total_gpu_mem >= 20:     # 2x RTX 4070 = 24 GB / A100 / etc.
            MAX_EPOCHS = 80
            BATCH_SIZE = 3 * num_gpus   # 3 на GPU — ~8-9ГБ из 11.6ГБ, с запасом
            ACCUM_STEPS = 1             # Без accumulation
            PATCH_SIZE = (128, 128, 128)
            NUM_SAMPLES = 6
            NUM_WORKERS = 4
            CHANNELS = (32, 64, 128, 256, 512)
            STRIDES = (2, 2, 2, 2)
        elif total_gpu_mem >= 8:
            MAX_EPOCHS = 60
            BATCH_SIZE = 2 * num_gpus
            ACCUM_STEPS = 1
            PATCH_SIZE = (96, 96, 96)
            NUM_SAMPLES = 6
            NUM_WORKERS = 4
            CHANNELS = (16, 32, 64, 128, 256)
            STRIDES = (2, 2, 2, 2)
        else:
            MAX_EPOCHS = 40
            BATCH_SIZE = 1 * num_gpus
            ACCUM_STEPS = 1
            PATCH_SIZE = (64, 64, 64)
            NUM_SAMPLES = 2
            NUM_WORKERS = 4
            CHANNELS = (16, 32, 64, 128)
            STRIDES = (2, 2, 2)

    print(f"Устройство: {device} (GPU: {num_gpus})")
    print(f"Патчи: {PATCH_SIZE}, Batch: {BATCH_SIZE}x{ACCUM_STEPS}accum={BATCH_SIZE*ACCUM_STEPS}eff, Эпохи: {MAX_EPOCHS}")

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
                # Берём только сканы с узелками (foreground > 0)
                mask_img = sitk.ReadImage(mask_path)
                mask_arr = sitk.GetArrayFromImage(mask_img)
                if mask_arr.max() > 0:
                    data_dicts.append({"image": img_path, "label": mask_path})

    print(f"Сканов с узелками: {len(data_dicts)}")

    if not data_dicts:
        print("ОШИБКА: Не найдено сканов с узелками!")
        print("Сначала запусти: python prepare_masks.py")
        return

    np.random.seed(42)
    np.random.shuffle(data_dicts)

    # Разделение: 80% train / 20% validation
    split = max(1, int(len(data_dicts) * 0.8))
    train_dicts = data_dicts[:split]
    val_dicts = data_dicts[split:]
    print(f"  Train: {len(train_dicts)}, Val: {len(val_dicts)}")

    # =========================================================================
    # ТРАНСФОРМАЦИИ
    # =========================================================================
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image", "label"]),

        # Изотропный ресамплинг 1мм — критично при разном slice thickness
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),

        # Нормализация КТ Hounsfield Units → [0, 1]
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),

        # Паддинг до минимального размера патча (если скан меньше)
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),

        # Вырезаем 3D-патчи: pos=3/neg=1 — акцент на узелки
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=3, neg=1,
            num_samples=NUM_SAMPLES,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),

        # Геометрические аугментации
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

        # Интенсивностные аугментации (только image)
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.3),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000, a_max=400,
            b_min=0.0, b_max=1.0,
            clip=True,
        ),
    ])

    # =========================================================================
    # ДАТАСЕТЫ И ЛОАДЕРЫ — CacheDataset кэширует снимки в RAM
    # =========================================================================
    print("Кэширование данных в RAM (первый запуск может занять несколько минут)...")
    train_ds = CacheDataset(
        data=train_dicts, transform=train_transforms,
        cache_rate=0.3, num_workers=NUM_WORKERS,
    )
    val_ds = CacheDataset(
        data=val_dicts, transform=val_transforms,
        cache_rate=0.5, num_workers=NUM_WORKERS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=3 if NUM_WORKERS > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=3 if NUM_WORKERS > 0 else None,
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
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Модель: 3D U-Net, параметров: {total_params:,}")

    # Multi-GPU: DataParallel
    if torch.cuda.is_available() and num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"✓  DataParallel: модель на {num_gpus} GPU")

    # Loss — DiceFocalLoss: Dice для перекрытия + Focal для мелких узелков
    loss_function = DiceFocalLoss(
        to_onehot_y=True, softmax=True,
        include_background=False,
        gamma=2.0,
        lambda_dice=1.0,
        lambda_focal=2.0,   # Усиленный Focal для редкого foreground
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * max(num_gpus, 1), weight_decay=1e-5)
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
    print(f"Валидация каждые {VAL_INTERVAL} эпох")
    print(f"Сохранение каждые {SAVE_INTERVAL} эпох")
    print(f"Mixed Precision (AMP): {'Да' if use_amp else 'Нет'}")
    print(f"{'='*60}\n")

    total_start = time.time()

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        elapsed_total = time.time() - total_start
        if epoch > 0:
            avg_epoch_time = elapsed_total / epoch
            eta = avg_epoch_time * (MAX_EPOCHS - epoch)
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        else:
            eta_str = '??:??:??'

        print(f"\n{'='*60}")
        print(f"  Эпоха {epoch + 1}/{MAX_EPOCHS}  |  "
              f"Прошло: {time.strftime('%H:%M:%S', time.gmtime(elapsed_total))}  |  "
              f"ETA: {eta_str}")
        print(f"{'='*60}")

        # === TRAIN ===
        model.train()
        epoch_loss = 0.0
        step = 0

        optimizer.zero_grad(set_to_none=True)
        train_pbar = tqdm(train_loader, desc=f"[Train {epoch+1}/{MAX_EPOCHS}]",
                          unit="batch", leave=True, dynamic_ncols=True)
        for batch_data in train_pbar:
            step += 1
            inputs = batch_data["image"].to(device, non_blocking=True)
            labels = batch_data["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss = loss / ACCUM_STEPS  # Нормализация для gradient accumulation

            scaler.scale(loss).backward()

            if step % ACCUM_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * ACCUM_STEPS  # Реальный loss (без нормализации)
            avg_so_far = epoch_loss / step
            train_pbar.set_postfix(loss=f"{loss.item()*ACCUM_STEPS:.4f}", avg=f"{avg_so_far:.4f}")

        # Flush остатка gradient accumulation
        if step % ACCUM_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss = epoch_loss / max(step, 1)
        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']

        # === VALIDATION (каждые VAL_INTERVAL эпох) ===
        metric_val = -1.0
        if (epoch + 1) % VAL_INTERVAL == 0 or (epoch + 1) == MAX_EPOCHS:
            model.eval()
            raw_model = model.module if hasattr(model, 'module') else model
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"[Val   {epoch+1}/{MAX_EPOCHS}]",
                                unit="scan", leave=True, dynamic_ncols=True)
                for val_data in val_pbar:
                    val_inputs = val_data["image"].to(device, non_blocking=True)
                    val_labels = val_data["label"].to(device, non_blocking=True)

                    with torch.amp.autocast("cuda", enabled=use_amp):
                        val_outputs = sliding_window_inference(
                            val_inputs, PATCH_SIZE, sw_batch_size=4, predictor=raw_model,
                        )

                    # One-hot предсказания для DiceMetric
                    val_pred = torch.argmax(val_outputs, dim=1, keepdim=True)
                    num_classes = val_outputs.shape[1]
                    val_pred_onehot = torch.nn.functional.one_hot(
                        val_pred.squeeze(1).long(), num_classes
                    ).permute(0, 4, 1, 2, 3).float()
                    val_labels_onehot = torch.nn.functional.one_hot(
                        val_labels.squeeze(1).long(), num_classes
                    ).permute(0, 4, 1, 2, 3).float()
                    dice_metric(y_pred=val_pred_onehot, y=val_labels_onehot)

                metric_val = dice_metric.aggregate().item()
                dice_metric.reset()

            elapsed = time.time() - epoch_start
            print(f"\n  >> Эпоха {epoch+1}/{MAX_EPOCHS}: "
                  f"Loss={avg_loss:.4f} | Dice={metric_val:.4f} | "
                  f"LR={lr_now:.2e} | Время: {elapsed:.1f}с")

            # Сохраняем лучшую модель по Dice
            if metric_val > best_metric:
                best_metric = metric_val
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_metric,
                    "loss": avg_loss,
                }, MODEL_SAVE_PATH)
                print(f"  ★ Лучшая модель сохранена (Dice: {best_metric:.4f})")
        else:
            elapsed = time.time() - epoch_start
            print(f"\n  >> Эпоха {epoch+1}/{MAX_EPOCHS}: "
                  f"Loss={avg_loss:.4f} | Val: следующая на эп. {((epoch+1)//VAL_INTERVAL+1)*VAL_INTERVAL} | "
                  f"LR={lr_now:.2e} | Время: {elapsed:.1f}с")

        # Периодическое сохранение чекпоинтов
        if (epoch + 1) % SAVE_INTERVAL == 0:
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            save_path = MODEL_SAVE_PATH.replace(".pth", f"_ep{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "dice": metric_val,
            }, save_path)
            print(f"  📁 Чекпоинт: {os.path.basename(save_path)}")

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
        for i in range(torch.cuda.device_count()):
            peak_mem = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
            print(f"Пик GPU {i} памяти: {peak_mem:.2f} GB")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
