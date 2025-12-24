"""
ОПТИМАЛЬНОЕ ОБУЧЕНИЕ - НОВЫЙ ЦИКЛ
"""
from ultralytics import YOLO
import torch
import time
from datetime import datetime
from pathlib import Path

if __name__ == '__main__':
    print("⚡ ОПТИМАЛЬНОЕ ОБУЧЕНИЕ - НОВЫЙ ЦИКЛ")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent

    # 1. Загружаем БАЗОВУЮ модель для НОВОГО обучения
    # Указываем путь к ПРЕДЫДУЩЕЙ модели как базовую
    base_model_relative_path = Path("ADAS") / "yolov8_new_80epochs_1215_1311" / "weights" / "best.pt"
    base_model_path_full = project_root / base_model_relative_path

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Устройство для обучения: {'cuda:0' if device == 0 else 'cpu'}")

    if not base_model_path_full.exists():
        print(f"❌ Файл базовой модели не найден: {base_model_path_full}")
        print("   Загружаю стандартную модель yolov8s.pt")
        model = YOLO('yolov8s.pt')
    else:
        print(f"📦 Загружаю модель как базовую: {base_model_path_full}")
        model = YOLO(str(base_model_path_full))
        print("🎯 Начинаю НОВОЕ обучение на основе существующей модели")

    # 2. НОВЫЕ ПАРАМЕТРЫ - НАЧИНАЕМ С НУЛЯ
    config = {
        'data': str(project_root / 'ADAS' / 'data' / 'racetrack' / 'data.yaml'),
        'epochs': 60,  # МЕНЬШЕ эпох, так как модель уже обучена
        'imgsz': 512,
        'batch': 24,
        'device': device,
        'workers': 4,

        # Важно: НЕ продолжаем обучение, а начинаем новое
        'resume': False,  # ⚠️ ИЗМЕНИТЬ НА False!
        'pretrained': False,  # Модель уже загружена

        # Уникальное имя для нового цикла обучения
        'name': f'yolov8_refined_{datetime.now().strftime("%m%d_%H%M")}',
        'project': 'ADAS',
        'exist_ok': True,

        # Остальные параметры остаются
        'val': True,
        'plots': True,
        'save': True,
        'save_period': 10,
        'verbose': True,
        'amp': True,
        'half': True,
        'cos_lr': True,
        'optimizer': 'AdamW',

        # Уменьшаем аугментацию для тонкой настройки
        'mosaic': 0.3,
        'mixup': 0.0,
        'degrees': 2.0,
        'shear': 0.2,
        'perspective': 0.0001,
        'fliplr': 0.3,
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,

        # Уменьшаем learning rate
        'lr0': 0.001,  # Меньше, чем обычно
        'lrf': 0.001,
        'momentum': 0.9,
        'weight_decay': 0.0001,
        'warmup_epochs': 1,
        'warmup_momentum': 0.8,
        'patience': 50,
    }

    print(f"\n⚙️  ПАРАМЕТРЫ ОБУЧЕНИЯ (НОВЫЙ ЦИКЛ):")
    for k, v in config.items():
        print(f"   {k}: {v}")

    print(f"\n📊 СТРАТЕГИЯ:")
    print(f"   • НЕ продолжаем обучение (resume=False)")
    print(f"   • Используем обученную модель как начальные веса")
    print(f"   • Меньше эпох (60 вместо 80)")
    print(f"   • Меньший learning rate для тонкой настройки")
    print(f"   • Уменьшенная аугментация")

    print("\n" + "=" * 60)

    # 3. ЗАПУСК НОВОГО ОБУЧЕНИЯ
    start_time = time.time()
    print("🚀 ЗАПУСКАЮ НОВЫЙ ЦИКЛ ОБУЧЕНИЯ...")

    try:
        results = model.train(**config)

        # 4. РЕЗУЛЬТАТЫ
        end_time = time.time()
        total_hours = (end_time - start_time) / 3600

        print(f"\n{'=' * 60}")
        print(f"✅ НОВОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"⏱️  Время: {total_hours:.2f} часов")
        print(f"📁 Папка с результатами: {results.save_dir}")

        if hasattr(results, 'metrics'):
            print(f"\n📊 МЕТРИКИ ПОСЛЕ ОБУЧЕНИЯ:")
            print(f"   mAP50 (box): {results.metrics.box.map50:.3f}")
            print(f"   mAP50-95 (box): {results.metrics.box.map:.3f}")

        # 5. БЫСТРЫЙ ТЕСТ
        print(f"\n{'=' * 60}")
        print(f"🧪 ТЕСТИРУЮ УЛУЧШЕННУЮ МОДЕЛЬ...")

        test_image_relative_path = Path(
            "ADAS") / "data" / "racetrack" / "valid" / "images" / "green_10_Color_png.rf.0f353b7850a5deade30ca2a6b2b692a6.jpg"
        test_image_path = project_root / test_image_relative_path

        if not test_image_path.exists():
            print(f"❌ Тестовое изображение не найдено")
        else:
            new_best_model_path = Path(results.save_dir) / "weights" / "best.pt"
            if new_best_model_path.exists():
                new_model = YOLO(str(new_best_model_path))
                print(f"✅ Загружена улучшенная модель: {new_best_model_path}")

                test_results = new_model.predict(
                    source=str(test_image_path),
                    conf=0.3,
                    imgsz=config['imgsz'],
                    device=device,
                    save=True,
                    project="ADAS",
                    name=f"{config['name']}_test",
                    exist_ok=True
                )

                print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТА:")
                for r in test_results:
                    if r.boxes is not None:
                        print(f"   Обнаружено объектов: {len(r.boxes)}")
                        for box in r.boxes:
                            class_id = int(box.cls)
                            class_name = new_model.names.get(class_id, f"Class_{class_id}")
                            conf = float(box.conf)
                            print(f"      • {class_name}: {conf:.2f}")
                    else:
                        print("   Объекты не обнаружены")

    except KeyboardInterrupt:
        print(f"\n⚠️  Обучение остановлено вручную.")
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback

        traceback.print_exc()

    # 6. ИНСТРУКЦИЯ
    print(f"\n{'=' * 60}")
    print(f"🎯 КАК ИСПОЛЬЗОВАТЬ УЛУЧШЕННУЮ МОДЕЛЬ:")
    print(f'1. **В src/detector.py замени путь на:**')
    print(
        f'   model_path = r"C:\\Users\\sande\\Downloads\\ADAS\\ADAS\\runs\\detect\\{config["name"]}\\weights\\best.pt"')
    print(f'2. **Или используйте относительный путь:**')
    print(f'   Path("ADAS") / "runs" / "detect" / "{config["name"]}" / "weights" / "best.pt"')
    print(f'3. **Перезапусти API сервер** (python api/main.py)')
    print(f"\n{'=' * 60}")
    print("✅ ГОТОВО К ИСПОЛЬЗОВАНИЮ")
    print("=" * 60)