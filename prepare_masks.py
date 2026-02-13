"""
Генерация 3D-масок узелков из annotations.csv для LUNA16.
Превращает координаты центров сфер в бинарные маски.
Автоматически обрабатывает ВСЕ subset* папки из data/.

Структура проекта:
    LUNA16_Project/
    ├── data/              ← subset0..subset6 внутри
    ├── masks/             ← сюда сохраняются маски
    ├── seg-lungs/         ← маски лёгких (опционально)
    ├── annotations.csv
    ├── candidates.csv
    ├── prepare_masks.py   ← ЭТОТ ФАЙЛ
    └── train.py
"""

import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


def main():
    # --- НАСТРОЙКИ ПУТЕЙ ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")              # Папка с subset'ами
    CSV_PATH = os.path.join(BASE_DIR, "annotations.csv")   # Аннотации
    OUTPUT_MASK_DIR = os.path.join(BASE_DIR, "masks")       # Куда сохранять маски

    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    # Проверяем наличие файлов
    if not os.path.isfile(CSV_PATH):
        print(f"ОШИБКА: Не найден {CSV_PATH}")
        print("Положи annotations.csv рядом с этим скриптом.")
        return

    if not os.path.isdir(DATA_DIR):
        print(f"ОШИБКА: Не найдена папка {DATA_DIR}")
        print("Создай папку data/ и положи в неё subset0..subsetN.")
        return

    # Автоматический поиск всех subset* папок внутри data/
    subset_dirs = sorted([
        os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
        if d.startswith("subset") and os.path.isdir(os.path.join(DATA_DIR, d))
    ])

    if not subset_dirs:
        print(f"ОШИБКА: Внутри {DATA_DIR} не найдено ни одной папки subset*!")
        return

    print(f"Найдено {len(subset_dirs)} subset-папок: "
          f"{[os.path.basename(d) for d in subset_dirs]}")

    # Читаем аннотации
    df = pd.read_csv(CSV_PATH)
    print(f"Всего аннотаций: {len(df)}")

    # Собираем все .mhd файлы из всех subset'ов
    all_files = []
    for sdir in subset_dirs:
        mhd_files = sorted([f for f in os.listdir(sdir) if f.endswith('.mhd')])
        for f in mhd_files:
            all_files.append((sdir, f))

    print(f"Найдено {len(all_files)} снимков. Начинаем генерацию масок...\n")

    created = 0
    skipped = 0

    for img_dir, file in tqdm(all_files, desc="Генерация масок"):
        series_uid = file[:-4]  # Убираем .mhd

        # Ищем узелки для этого снимка
        nodules = df[df['seriesuid'] == series_uid]

        # Пропускаем если маска уже существует
        out_path = os.path.join(OUTPUT_MASK_DIR, f"{series_uid}_mask.mhd")
        if os.path.exists(out_path):
            created += 1
            continue

        if len(nodules) == 0:
            # Пустая маска для негативных сканов (важно для обучения!)
            img_path = os.path.join(img_dir, file)
            itk_img = sitk.ReadImage(img_path)
            img_array = sitk.GetArrayFromImage(itk_img)
            mask_array = np.zeros_like(img_array, dtype=np.uint8)
            mask_itk = sitk.GetImageFromArray(mask_array)
            mask_itk.CopyInformation(itk_img)
            sitk.WriteImage(mask_itk, out_path)
            created += 1
            continue

        # 1. Загружаем снимок
        img_path = os.path.join(img_dir, file)
        itk_img = sitk.ReadImage(img_path)

        origin = np.array(itk_img.GetOrigin())     # (x, y, z) в мм
        spacing = np.array(itk_img.GetSpacing())    # Размер вокселя в мм
        img_array = sitk.GetArrayFromImage(itk_img) # (z, y, x) — numpy порядок

        # 2. Создаём пустую маску
        mask_array = np.zeros_like(img_array, dtype=np.uint8)

        # 3. Рисуем сферу для каждого узелка
        for _, row in nodules.iterrows():
            center_world = np.array([row['coordX'], row['coordY'], row['coordZ']])
            diameter = row['diameter_mm']
            radius = diameter / 2.0

            # Мировые координаты (мм) → индексы вокселей
            center_idx = itk_img.TransformPhysicalPointToIndex(center_world.tolist())
            cx, cy, cz = center_idx  # (x, y, z) порядок

            # Радиус в вокселях по каждой оси
            r_voxel = np.ceil(radius / spacing).astype(int)

            # Границы куба вокруг узелка (numpy-порядок: z, y, x)
            z_min = max(0, cz - r_voxel[2])
            z_max = min(img_array.shape[0], cz + r_voxel[2] + 1)
            y_min = max(0, cy - r_voxel[1])
            y_max = min(img_array.shape[1], cy + r_voxel[1] + 1)
            x_min = max(0, cx - r_voxel[0])
            x_max = min(img_array.shape[2], cx + r_voxel[0] + 1)

            # Сетка координат
            z_grid, y_grid, x_grid = np.ogrid[z_min:z_max, y_min:y_max, x_min:x_max]

            # Расстояние в мм (эллипсоид, учитывает анизотропный spacing)
            dist_sq = (
                ((x_grid - cx) * spacing[0]) ** 2 +
                ((y_grid - cy) * spacing[1]) ** 2 +
                ((z_grid - cz) * spacing[2]) ** 2
            )

            mask_array[z_min:z_max, y_min:y_max, x_min:x_max][dist_sq <= radius ** 2] = 1

        # 4. Сохраняем маску в .mhd (с теми же метаданными)
        mask_itk = sitk.GetImageFromArray(mask_array)
        mask_itk.CopyInformation(itk_img)
        sitk.WriteImage(mask_itk, out_path)
        created += 1

    print(f"\nГотово! Создано масок: {created} (включая пустые для негативных сканов), пропущено: {skipped}")


if __name__ == "__main__":
    main()
