"""
ГИБРИДНЫЙ ДЕТЕКТОР ADAS
"""
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import uuid
import json
from collections import Counter


class HybridADASDetector:
    def __init__(self, static_dir: str = None, static_url_prefix: str = "/static"):
        """Инициализация гибридного детектора"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Устройство: {self.device}")

        # Сохраняем префикс для формирования URL'ов
        self.static_url_prefix = (static_url_prefix or "").rstrip('/')

        # static_dir - абсолютный путь к папке static
        if static_dir is None:
            self.static_dir = (Path.cwd() / "static").resolve()
        else:
            self.static_dir = Path(static_dir).resolve()

        self.static_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Используемая static папка: {self.static_dir}")

        print("\n" + "=" * 60)
        print("🤝 ЗАГРУЗКА ГИБРИДНОГО ДЕТЕКТОРА")
        print("   Ваша модель для машин + COCO для остального")
        print("=" * 60)

        # 1. 🚗 Ваша дообученная модель для машин (8 классов)
        car_model_path = r"C:\Users\sande\Downloads\ADAS\ADAS\yolov8_finetuned_city\weights\best.pt"
        if Path(car_model_path).exists():
            self.car_model = YOLO(car_model_path)
            self.car_model.to(self.device)
            print(f"✅ Загружена ВАША модель для машин")
            print(f"   Классы: {self.car_model.names}")
        else:
            print(f"❌ Ваша модель не найдена: {car_model_path}")
            print(f"   Использую только COCO модель")
            self.car_model = None

        # 2. 👤 Предобученная COCO модель для всего остального
        self.coco_model = YOLO('yolov8s.pt')  # YOLOv8s с COCO датасетом (80 классов)
        self.coco_model.to(self.device)
        print(f"✅ Загружена COCO модель (80 классов)")
        print(f"   Включает: person, traffic light, stop sign, и др.")

        # Определяем какие классы ищет каждая модель
        if self.car_model:
            self.car_classes = ['car', 'truck', 'bus', 'bike', 'ego_vehicle', 'racetrack', 'obstacle']
        else:
            self.car_classes = []

        # COCO classes (80 classes)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

        # Папки внутри static
        self.results_dir = self.static_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.uploads_dir = self.static_dir / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        # Папка для логов
        self.logs_dir = Path("detection_logs")
        self.logs_dir.mkdir(exist_ok=True)

        # История запросов
        self.history = []

        print(f"\n✅ Гибридный детектор инициализирован!")
        print(f"   🚗 Ваша модель: {len(self.car_classes) if self.car_model else 0} классов")
        print(f"   👤 COCO модель: {len(self.coco_classes)} классов")
        print(f"   📁 Результаты: {self.results_dir}")

    def predict(self, image_path: str, confidence: float = 0.25):
        """Объединенная детекция от двух моделей"""
        print(f"\n{'=' * 60}")
        print(f"🔍 ГИБРИДНАЯ ДЕТЕКЦИЯ: {Path(image_path).name}")
        print(f"🎯 Ваша модель + COCO модель")
        print(f"📊 Порог уверенности: {confidence}")
        print(f"{'=' * 60}")

        try:
            start_time = time.time()

            # Загрузка изображения
            print(f"📥 Загружаю изображение: {image_path}")
            img = cv2.imread(str(image_path))
            if img is None:
                return {
                    "success": False,
                    "error": f"Не удалось загрузить изображение: {image_path}",
                    "count": 0,
                    "detections": [],
                    "processing_time_ms": 0
                }

            original_height, original_width = img.shape[:2]
            print(f"📏 Размер изображения: {original_width}x{original_height}")

            # Все детекции
            all_detections = []

            # 1. 🚗 Детекция от ВАШЕЙ модели (если доступна)
            if self.car_model:
                print(f"\n🚗 ЗАПУСК ВАШЕЙ МОДЕЛИ (дообученная)...")
                car_results = self.car_model.predict(
                    source=img,
                    conf=confidence,
                    imgsz=512,
                    verbose=False,
                    device=self.device
                )[0]

                if car_results.boxes is not None:
                    car_count = 0
                    for box in car_results.boxes:
                        cls_id = int(box.cls)
                        cls_name = car_results.names.get(cls_id, str(cls_id))
                        conf = float(box.conf)
                        bbox = box.xyxy[0].tolist()

                        # Берем только нужные классы из вашей модели
                        if cls_name in self.car_classes:
                            all_detections.append({
                                'class': cls_name,
                                'confidence': conf,
                                'bbox': [round(val, 2) for val in bbox],
                                'model': 'finetuned'
                            })
                            car_count += 1

                    print(f"   ✅ Найдено объектов от вашей модели: {car_count}")
                else:
                    print(f"   ⚠️  Ваша модель не нашла объектов")

            # 2. 👤 Детекция от COCO модели
            print(f"\n👤 ЗАПУСК COCO МОДЕЛИ (80 классов)...")
            coco_results = self.coco_model.predict(
                source=img,
                conf=confidence,
                imgsz=640,
                verbose=False,
                device=self.device
            )[0]

            if coco_results.boxes is not None:
                coco_count = 0
                for box in coco_results.boxes:
                    cls_id = int(box.cls)
                    cls_name = coco_results.names.get(cls_id, str(cls_id))
                    conf = float(box.conf)
                    bbox = box.xyxy[0].tolist()

                    # Берем классы, которые не ищет ваша модель
                    # Исключаем 'car', 'truck', 'bus' так как их лучше ищет ваша модель
                    exclude_classes = ['car', 'truck', 'bus', 'motorcycle']

                    if cls_name not in exclude_classes:
                        all_detections.append({
                            'class': cls_name,
                            'confidence': conf,
                            'bbox': [round(val, 2) for val in bbox],
                            'model': 'coco'
                        })
                        coco_count += 1

                print(f"   ✅ Найдено объектов от COCO модели: {coco_count}")
            else:
                print(f"   ⚠️  COCO модель не нашла объектов")

            # Удаляем дубликаты
            all_detections = self._remove_duplicates(all_detections)

            # Время обработки
            total_time_ms = (time.time() - start_time) * 1000

            # Визуализация результата
            result_image_url = None
            if all_detections:
                # Создаем уникальное имя для результата
                result_id = str(uuid.uuid4())[:8]
                result_filename = f"hybrid_result_{result_id}.jpg"
                result_path = self.results_dir / result_filename

                # Аннотируем изображение
                annotated_img = self._annotate_image(img.copy(), all_detections)

                # Сохраняем результат
                cv2.imwrite(str(result_path), annotated_img)
                result_image_url = f"{self.static_url_prefix}/results/{result_filename}"
                print(f"💾 Результат сохранен: {result_path}")

            # Статистика
            print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
            print(f"   Всего объектов: {len(all_detections)}")
            print(f"   Время обработки: {total_time_ms:.1f} мс")

            if all_detections:
                # Группировка по классам
                class_stats = Counter([d['class'] for d in all_detections])
                model_stats = Counter([d.get('model', 'unknown') for d in all_detections])

                print(f"   Распределение по классам:")
                for cls, count in class_stats.most_common(10):  # Топ-10 классов
                    print(f"     - {cls}: {count}")

                print(f"   Распределение по моделям:")
                for model, count in model_stats.items():
                    print(f"     - {model}: {count}")

            # Сохраняем в историю
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "image": str(image_path),
                "detections": len(all_detections),
                "time_ms": total_time_ms,
                "result_image": result_image_url
            }

            self.history.append(log_entry)
            if len(self.history) > 100:
                self.history = self.history[-100:]

            # Возвращаем результат
            result = {
                "success": True,
                "detections": all_detections,
                "count": len(all_detections),
                "processing_time_ms": round(total_time_ms, 2),
                "timestamp": datetime.now().isoformat(),
                "result_image": result_image_url
            }

            print(f"\n✅ ГИБРИДНАЯ ДЕТЕКЦИЯ ЗАВЕРШЕНА!")
            print(f"{'=' * 60}")

            return result

        except Exception as e:
            import traceback
            error_msg = f"Ошибка гибридной детекции: {str(e)}"
            print(f"\n❌ {error_msg}")
            traceback.print_exc()

            return {
                "success": False,
                "error": error_msg,
                "count": 0,
                "detections": [],
                "processing_time_ms": 0
            }

    def _remove_duplicates(self, detections, iou_threshold=0.5):
        """Удаление дублирующихся детекций"""
        if not detections:
            return detections

        # Сортируем по уверенности
        detections.sort(key=lambda x: x['confidence'], reverse=True)

        filtered = []
        used_boxes = []

        for det in detections:
            box = det['bbox']
            is_duplicate = False

            # Проверяем пересечение с уже выбранными боксами
            for used in used_boxes:
                iou = self._calculate_iou(box, used)
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(det)
                used_boxes.append(box)

        return filtered

    def _calculate_iou(self, box1, box2):
        """Вычисление IoU (Intersection over Union)"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _annotate_image(self, image, detections):
        """Аннотация изображения с цветами по типу модели"""
        # Цвета для разных типов объектов
        colors = {
            'car': (0, 255, 0),  # Зеленый - ваши машины
            'truck': (0, 165, 255),  # Оранжевый
            'bus': (255, 0, 0),  # Синий
            'bike': (255, 255, 0),  # Голубой
            'ego_vehicle': (255, 0, 255),  # Розовый
            'racetrack': (0, 255, 255),  # Желтый
            'obstacle': (128, 0, 128),  # Фиолетовый
            'person': (255, 165, 0),  # Оранжевый яркий
            'traffic light': (0, 100, 255),  # Темно-синий
            'stop sign': (0, 0, 255),  # Красный
        }

        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls = det['class']
            conf = det['confidence']
            model_source = det.get('model', 'unknown')

            # Цвет по классу или серый по умолчанию
            color = colors.get(cls, (128, 128, 128))

            # Толщина рамки по модели
            thickness = 3 if model_source == 'finetuned' else 2

            # Рисуем рамку
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

            # Подпись
            source_tag = "🚗" if model_source == 'finetuned' else "👤"
            label = f"{source_tag} {cls} {conf:.2f}"

            # Фон для текста
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # Прямоугольник под текст
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Текст
            cv2.putText(
                image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # Белый текст
                2
            )

        return image

    def get_stats(self):
        """Получить статистику"""
        if not self.history:
            return {
                "total_predictions": 0,
                "avg_time_ms": 0,
                "last_prediction": None,
                "models": "Гибридный (finetuned + COCO)",
                "total_classes": 80 + (len(self.car_classes) if self.car_model else 0)
            }

        avg_time = sum(h["time_ms"] for h in self.history) / len(self.history)

        return {
            "total_predictions": len(self.history),
            "avg_time_ms": round(avg_time, 2),
            "last_prediction": self.history[-1]["timestamp"],
            "models": "Гибридный (finetuned + COCO)",
            "total_classes": 80 + (len(self.car_classes) if self.car_model else 0)
        }

    def get_model_info(self):
        """Получить информацию о модели"""
        info = {
            "num_classes": 80 + (len(self.car_classes) if self.car_model else 0),
            "models": [
                {
                    "name": "finetuned_city",
                    "path": r"C:\Users\sande\Downloads\ADAS\ADAS\yolov8_finetuned_city\weights\best.pt",
                    "classes": self.car_classes if self.car_model else [],
                    "status": "loaded" if self.car_model else "not_found"
                },
                {
                    "name": "yolov8s_coco",
                    "path": "yolov8s.pt",
                    "classes_count": 80,
                    "description": "COCO датасет (person, traffic light, sign и др.)"
                }
            ],
            "device": self.device,
            "description": "Гибридный детектор ADAS: ваша модель для транспорта + COCO для остального",
            "type": "hybrid"
        }

        return info


# Тестирование
if __name__ == "__main__":
    print("🧪 ТЕСТ ГИБРИДНОГО ДЕТЕКТОРА")

    # Создаем тестовые папки
    test_static = Path("test_hybrid_static")
    test_static.mkdir(exist_ok=True)

    detector = HybridADASDetector(static_dir=str(test_static))

    # Тест на городском изображении
    test_images = [
        r"C:\Users\sande\Downloads\ADAS\new_cars_dataset\valid\images\DJI_20231027225840_0010_D_1_mp4-100_jpg.rf.ef829dd91a9f1c47908442bf190d7222.jpg",
        r"C:\Users\sande\Downloads\ADAS\new_cars_dataset\valid\images\DJI_20231027225840_0010_D_1_mp4-102_jpg.rf.b424d55f22daec5ad100dc4335b5dc2e.jpg"
    ]

    for test_img in test_images:
        if Path(test_img).exists():
            print(f"\n🔍 Тест на: {Path(test_img).name}")
            result = detector.predict(test_img)
            print(f"   Найдено объектов: {result['count']}")
        else:
            print(f"\n⚠️  Тестовое изображение не найдено: {test_img}")