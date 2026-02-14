#!/bin/bash
# Запуск обучения на 2 GPU через DDP
# Использование: bash run_train.sh

echo "=== Запуск обучения на 2 GPU (DDP) ==="
torchrun --nproc_per_node=2 train.py
