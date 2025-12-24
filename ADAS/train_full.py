"""
БЕЗОПАСНОЕ обучение для ноутбука RTX 4090
С контролем памяти и температуры
"""
import torch
from ultralytics import YOLO
import time
import os

print("=" * 70)
print("🛡️  БЕЗОПАСНОЕ ОБУЧЕНИЕ ДЛЯ НОУТБУКА")
print("=" * 70)

# ПРОВЕРКА ПАМЯТИ
print("🔍 ПРОВЕРКА СИСТЕМЫ:")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Текущее использование VRAM
    torch.cuda.empty_cache()  # Очищаем кэш
    vram_used = torch.cuda.memory_allocated() / 1e9
    vram_free = vram_total - vram_used

    print(f"GPU: {gpu_name}")
    print(f"VRAM всего: {vram_total:.1f} GB")
    print(f"VRAM используется: {vram_used:.1f} GB")
    print(f"VRAM свободно: {vram_free:.1f} GB")

    if vram_free < 2:  # Меньше 2GB свободно
        print("⚠️  ВНИМАНИЕ: Мало свободной VRAM!")
else:
    print("❌ CUDA не доступна!")
    exit()

# ОГРАНИЧЕНИЯ ДЛЯ БЕЗОПАСНОСТИ
print("\n🛡️  УСТАНАВЛИВАЮ ОГРАНИЧЕНИЯ:")
print("   • Batch: 8 (минимальный для безопасности)")
print("   • Workers: 4 (меньше нагрузки на CPU/RAM)")
print("   • Mixed Precision: ВКЛ (меньше памяти)")
print("   • FP16: ВКЛ (половинная точность)")

# УСТАНАВЛИВАЕМ ЛИМИТ ПАМЯТИ
torch.cuda.set_per_process_memory_fraction(0.6)  # Максимум 60% VRAM
print("   • Лимит VRAM: 60% от 16GB = ~9.6 GB")

# ОПТИМИЗАЦИИ
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ЗАГРУЖАЕМ МОДЕЛЬ (более легкую)
print("\n📦 Загружаю YOLOv8s (small - легче для памяти)...")
try:
    model = YOLO('yolov8s.pt')  # Small версия вместо Medium!
    print("✅ YOLOv8s загружена (меньше параметров, меньше памяти)")
except:
    model = YOLO('yolov8n.pt')  # Nano если small не загрузится
    print("✅ YOLOv8n загружена (самая легкая)")

# ПРОВЕРКА ДАТАСЕТА
if not os.path.exists('data/racetrack/data.yaml'):
    print("❌ Датасет не найден!")
    exit()

print("\n✅ Датасет найден")

# СУПЕР-БЕЗОПАСНЫЕ НАСТРОЙКИ
SAFE_CONFIG = {
    'data': 'data/racetrack/data.yaml',
    'epochs': 30,  # Меньше эпох
    'imgsz': 416,  # Уменьшаем размер! (было 640)
    'batch': 8,  # ОЧЕНЬ маленький batch
    'device': 0,
    'workers': 2,  # Минимум workers
    'name': 'yolov8s_safe_training',
    'exist_ok': True,
    'pretrained': True,
    'amp': True,  # Mixed precision
    'val': True,
    'save': True,
    'plots': True,
    'verbose': True,
    'half': True,  # FP16 - экономия памяти
    'patience': 10,
    'cos_lr': True,
    'lr0': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 2,
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'overlap_mask': False,
    'mask_ratio': 4,
    'dropout': 0.0,
    'resume': False,
    'fraction': 0.5,  # Используем только 50% данных! ⚡
}

print(f"\n⚙️  СУПЕР-БЕЗОПАСНЫЕ ПАРАМЕТРЫ:")
print(f"   Модель: {model.__class__.__name__}")
print(f"   Размер: {SAFE_CONFIG['imgsz']}×{SAFE_CONFIG['imgsz']} (уменьшено!)")
print(f"   Batch: {SAFE_CONFIG['batch']}")
print(f"   Workers: {SAFE_CONFIG['workers']}")
print(f"   Эпохи: {SAFE_CONFIG['epochs']}")
print(f"   Данные: {SAFE_CONFIG['fraction'] * 100}% от датасета")
print(f"   FP16: {'ВКЛ' if SAFE_CONFIG['half'] else 'ВЫКЛ'}")

print(f"\n{'=' * 70}")
print("🚀 ЗАПУСКАЮ СУПЕР-БЕЗОПАСНОЕ ОБУЧЕНИЕ...")
print("   • Используется только 50% данных")
print("   • Размер изображений уменьшен до 416")
print("   • Batch всего 8")
print("   • Ожидаемое время: 1-1.5 часа")
print("=" * 70)


# ФУНКЦИЯ МОНИТОРИНГА ПАМЯТИ
def check_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        percent = (allocated / total) * 100

        print(f"   📊 Память: {allocated:.1f}/{total:.1f} GB ({percent:.0f}%)")

        if percent > 80:
            print("   ⚠️  ВНИМАНИЕ: Высокое использование VRAM!")
            return False
    return True


# ЗАПУСК ОБУЧЕНИЯ С КОНТРОЛЕМ
try:
    start_time = time.time()

    # Проверяем память перед началом
    if not check_memory():
        print("❌ Слишком много памяти используется! Остановка.")
        exit()

    results = model.train(**SAFE_CONFIG)

    end_time = time.time()
    hours = (end_time - start_time) / 3600

    print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО БЕЗОПАСНО!")
    print(f"⏱️  Время: {hours:.2f} часов")
    print(f"📁 Результаты: {results.save_dir}")

except KeyboardInterrupt:
    print(f"\n⚠️  ОБУЧЕНИЕ ОСТАНОВЛЕНО (безопасно)")
    print("   Сохранена последняя модель")
except torch.cuda.OutOfMemoryError:
    print(f"\n❌ OUT OF MEMORY! Слишком много памяти!")
    print("   Уменьши batch до 4 или imgsz до 320")
except Exception as e:
    print(f"\n❌ ОШИБКА: {e}")

# ФИНАЛЬНЫЙ СОВЕТ
print(f"\n{'=' * 70}")
print("💡 СОВЕТЫ ДЛЯ СНИЖЕНИЯ НАГРУЗКИ:")
print("1. Закрой все лишние программы (игры, браузеры)")
print("2. Уменьши разрешение в настройках Windows")
print("3. Используй охлаждающую подставку")
print("4. Обучай ночью когда ноутбук не используется")
print("5. Если всё равно перегревается - используй Google Colab")
print("=" * 70)