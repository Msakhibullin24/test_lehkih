#!/bin/bash
# ============================================================
# Установка окружения для LUNA16 проекта
# Запуск: bash setup.sh
# ============================================================

set -e

echo "============================================"
echo "  LUNA16 — Установка окружения"
echo "============================================"

# 1. Проверяем uv
if ! command -v uv &> /dev/null; then
    echo "uv не найден. Устанавливаю..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✓ uv: $(uv --version)"

# 2. Создаём venv с Python 3.12
echo ""
echo "→ Создаю виртуальное окружение (Python 3.12)..."
uv venv --python 3.12
source .venv/bin/activate

echo "✓ Python: $(python --version)"

# 3. Устанавливаем PyTorch с CUDA 12.4
echo ""
echo "→ Устанавливаю PyTorch + CUDA 12.4..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 4. Устанавливаем MONAI и зависимости
echo ""
echo "→ Устанавливаю MONAI, ITK, SimpleITK и прочее..."
uv pip install monai itk simpleitk pandas numpy tqdm

# 5. Проверяем CUDA
echo ""
echo "============================================"
echo "  Проверка окружения"
echo "============================================"
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
    mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Память:   {mem:.1f} GB')
else:
    print('GPU:      Не найден (будет работать на CPU)')

import monai
print(f'MONAI:    {monai.__version__}')

import SimpleITK as sitk
print(f'SimpleITK: {sitk.Version.VersionString()}')
print()
print('✓ Всё установлено!')
"

echo ""
echo "============================================"
echo "  Готово! Следующие шаги:"
echo "============================================"
echo ""
echo "  1. Активируй окружение:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Сгенерируй маски:"
echo "     python prepare_masks.py"
echo ""
echo "  3. Запусти обучение:"
echo "     python train.py"
echo ""
echo "============================================"
