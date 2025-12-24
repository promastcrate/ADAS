"""
ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ МОДЕЛИ
"""
from ultralytics import YOLO
import torch
from pathlib import Path
from datetime import datetime

print("🚀 ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ")

# Загружаем текущую модель
model_path = "runs/detect/yolov8s_safe_training/weights/best.pt"
if Path(model_path).exists():
    print(f"📦 Загружаю: {model_path}")
    model = YOLO(model_path)
else:
    print("⚠️  Начинаю с yolov8s.pt")
    model = YOLO("yolov8s.pt")

# Параметры обучения
config = {
    'data': 'data/racetrack/data.yaml',
    'epochs': 50,  # Добавим 50 эпох
    'imgsz': 640,
    'batch': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'resume': True,  # Продолжить обучение
    'name': 'yolov8_continued',
    'project': 'ADAS',
    'save_period': 10,
    'exist_ok': True,
}

print(f"⚙️  Эпохи: {config['epochs']}")
print(f"⚙️  Устройство: {config['device']}")

# Запуск
results = model.train(**config)

print(f"✅ Обучение завершено!")
print(f"📁 Модель: runs/detect/yolov8_continued/weights/best.pt")
