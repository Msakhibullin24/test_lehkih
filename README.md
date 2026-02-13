# LUNA16 — Детекция узелков в лёгких (3D U-Net)

## Структура проекта

```
LUNA16_Project/
├── data/                  ← subset0..subset6 (КТ-снимки .mhd/.raw)
├── masks/                 ← генерируемые маски узелков
├── seg-lungs/             ← маски лёгких (опционально, для улучшения)
├── annotations.csv        ← координаты узелков
├── candidates.csv         ← кандидаты
├── prepare_masks.py       ← генерация 3D-масок из CSV
├── train.py               ← обучение 3D U-Net (CUDA)
├── setup.sh               ← установка окружения
└── .gitignore
```

## Быстрый старт

### 1. Клонируй репо и положи данные
```bash
git clone git@github.com:Msakhibullin24/test_lehkih.git
cd test_lehkih
```

Положи рядом:
- `annotations.csv`, `candidates.csv` — CSV-файлы из LUNA16
- `data/subset0/` ... `data/subset6/` — распакованные снимки

### 2. Установи окружение
```bash
bash setup.sh
```

Или вручную:
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install monai itk simpleitk pandas numpy tqdm
```

### 3. Сгенерируй маски
```bash
source .venv/bin/activate
python prepare_masks.py
```

### 4. Запусти обучение
```bash
python train.py
```

## Автоматическое масштабирование под GPU

| GPU память  | Эпохи | Batch | Патч    | Архитектура U-Net     |
|-------------|-------|-------|---------|-----------------------|
| ≥20 GB      | 50    | 4     | 128³    | 5 уровней (32→512)    |
| 8–20 GB     | 30    | 2     | 96³     | 5 уровней (16→256)    |
| <8 GB       | 20    | 1     | 64³     | 4 уровня (16→128)     |
| CPU         | 3     | 1     | 64³     | 4 уровня (16→128)     |

## Технологии

- **PyTorch 2.6 + CUDA 12.4**
- **MONAI** — фреймворк для медицинских нейросетей
- **3D U-Net** — сегментация с Dice Loss
- **Mixed Precision (AMP)** — ускорение на GPU
- **SimpleITK** — чтение КТ-снимков (.mhd/.raw)
