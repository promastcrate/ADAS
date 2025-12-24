"""
Быстрая проверка датасета
"""
from pathlib import Path
import yaml

print("🔍 ПРОВЕРКА ДАТАСЕТА")
print("=" * 50)

dataset_path = Path("data/racetrack")
yaml_file = dataset_path / "data.yaml"

if yaml_file.exists():
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✅ Датасет найден: {dataset_path}")
    print(f"\n📊 ИНФОРМАЦИЯ:")
    print(f"   Классы: {config.get('names', 'не указаны')}")
    print(f"   Количество классов: {config.get('nc', 'не указано')}")
    print(f"   Путь: {config.get('path', 'не указан')}")

    # Считаем изображения
    train_images = list((dataset_path / "train" / "images").glob("*"))
    val_images = list((dataset_path / "valid" / "images").glob("*"))

    print(f"\n📈 СТАТИСТИКА:")
    print(f"   Тренировочных изображений: {len(train_images)}")
    print(f"   Валидационных изображений: {len(val_images)}")

    if train_images:
        # Показываем пример
        from PIL import Image
        import matplotlib.pyplot as plt

        print(f"\n👀 Пример изображения: {train_images[0].name}")
        img = Image.open(train_images[0])
        print(f"   Размер: {img.size}")

        # Показываем первое изображение
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title(f"Пример: {train_images[0].name}")
        plt.axis('off')
        plt.show()

else:
    print(f"❌ Файл data.yaml не найден в {dataset_path}")
    print("Убедись, что распаковал архив в правильную папку!")