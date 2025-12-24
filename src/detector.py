"""
ИСПРАВЛЕННЫЙ ДЕТЕКТОР ДЛЯ ADAS СИСТЕМЫ
- Использует ДООБУЧЕННУЮ модель для городских изображений
- Гарантирует, что результаты сохраняются в том же static-папке, которую обслуживает сервер
- Корректно обрабатывает типы изображений (PIL/NumPy) и конвертирует цвета для cv2.imwrite
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
import os

try:
    from PIL import Image
except Exception:
    Image = None


class SimpleADASDetector:
    def __init__(self, model_path: str = None, static_dir: str = None, static_url_prefix: str = "/static"):
        """Инициализация детектора с ДООБУЧЕННОЙ МОДЕЛЬЮ для городских изображений

        Args:
            model_path: путь к .pt модели (если None — использует дообученную модель для городских изображений)
            static_dir: абсолютный путь к каталогу static
            static_url_prefix: префикс URL для статических файлов (например, "/static")
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Устройство: {self.device}")

        # Сохраняем префикс для формирования URL'ов
        self.static_url_prefix = (static_url_prefix or "").rstrip('/')

        # static_dir - абсолютный путь к папке static
        if static_dir is None:
            print("⚠️  static_dir не был передан в детектор. Использование Path.cwd() / 'static' как запасной вариант.")
            self.static_dir = (Path.cwd() / "static").resolve()
        else:
            self.static_dir = Path(static_dir).resolve()

        self.static_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Используемая static папка: {self.static_dir}")

        # ⭐⭐ ДООБУЧЕННАЯ МОДЕЛЬ ДЛЯ ГОРОДСКИХ ИЗОБРАЖЕНИЙ ⭐⭐
        if model_path is None:
            # ПУТЬ К ДООБУЧЕННОЙ МОДЕЛИ
            model_path = r"C:\Users\sande\Downloads\ADAS\ADAS\yolov8_finetuned_city\weights\best.pt"

            # Проверяем существует ли файл
            if not Path(model_path).exists():
                print(f"❌ Дообученная модель не найдена по пути: {model_path}")
                # Пробуем найти альтернативные пути
                model_path = self._find_model_relative()
                if model_path is None:
                    print("❌ Альтернативные модели не найдены. Использую yolov8s.pt")
                    model_path = "yolov8s.pt"
            else:
                print("✅ Дообученная модель для городских изображений найдена!")

        self.model_path = str(model_path)
        print(f"📦 Загружаю ДООБУЧЕННУЮ модель: {self.model_path}")

        try:
            # Проверка существования файла
            if not Path(self.model_path).exists() and self.model_path != "yolov8s.pt":
                raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

            # Загрузка модели
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

            print(f"🎯 Классы модели: {self.model.names}")
            print(f"🔢 Всего классов: {len(self.model.names)}")

            # Выводим информацию о модели
            if "yolov8_finetuned_city" in self.model_path:
                print("✅ Загружена ДООБУЧЕННАЯ модель для городских изображений")
                print("🎯 Специализируется на: car, truck, bus, bike")
            elif "yolov8_refined" in self.model_path:
                print("⚠️  Загружена предыдущая модель (mAP50: 0.82)")
            else:
                print(f"ℹ️  Загружена модель: {Path(self.model_path).name}")

        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            if "No such file or directory" in str(e) or "File not found" in str(e):
                print(f"   Убедитесь, что файл модели существует по пути: {self.model_path}")
            raise

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

        print(f"✅ Детектор инициализирован с ДООБУЧЕННОЙ моделью!")
        print(f"📁 Результаты будут сохраняться в: {self.results_dir}")

    def _find_model_relative(self):
        """Пытается найти модель по известным относительным путям"""
        detector_file_path = Path(__file__).resolve()
        project_root = detector_file_path.parents[1]

        # Проверяем разные версии моделей
        paths_to_check = [
            # Дообученная модель для городских изображений
            project_root / "ADAS" / "yolov8_finetuned_city" / "weights" / "best.pt",
            # Новая улучшенная модель
            project_root / "ADAS" / "yolov8_refined_1222_2207" / "weights" / "best.pt",
            # Предыдущая модель
            project_root / "ADAS" / "yolov8_new_80epochs_1215_1311" / "weights" / "best.pt",
            project_root / "ADAS" / "runs" / "detect" / "yolov8s_safe_training" / "weights" / "best.pt",
        ]

        for path in paths_to_check:
            if path.exists():
                print(f"   ✅ Найдена модель: {path}")
                return str(path)

        return None

    def predict(self, image_path: str, confidence: float = 0.25):
        """Выполнить детекцию на изображении с ДООБУЧЕННОЙ моделью"""
        print(f"\n{'=' * 60}")
        print(f"🔍 НАЧАЛО ОБРАБОТКИ: {Path(image_path).name}")
        print(f"🎯 Использую ДООБУЧЕННУЮ модель для городских изображений")
        print(f"📊 Порог уверенности: {confidence}")
        print(f"{'=' * 60}")

        try:
            start_time = time.time()

            # 1) Загрузка изображения
            print(f"📥 Загружаю изображение: {image_path}")
            img = cv2.imread(str(image_path))
            if img is None:
                err = f"Не удалось загрузить изображение: {image_path}"
                print(f"❌ {err}")
                return {
                    "success": False,
                    "error": err,
                    "count": 0,
                    "detections": [],
                    "processing_time_ms": 0
                }

            original_height, original_width = img.shape[:2]
            print(f"📏 Размер изображения: {original_width}x{original_height}")

            # 2) Запуск детекции с imgsz=512 (как в обучении)
            print(f"🎯 Запускаю детекцию (conf={confidence}, imgsz=512)...")
            results = self.model(
                source=img,
                conf=confidence,
                imgsz=512,
                verbose=False,
                save=False,
                device=self.device
            )

            detections = []
            result_image_url = None

            # 3) Обработка результатов
            for i, r in enumerate(results):
                print(f"\n  📄 Результат {i+1}:")

                if r.boxes is not None and len(r.boxes) > 0:
                    print(f"    ✅ Найдено объектов: {len(r.boxes)}")

                    # Создаем уникальное имя для результата
                    result_id = str(uuid.uuid4())[:8]
                    result_filename = f"result_{result_id}.jpg"
                    result_path = self.results_dir / result_filename

                    print(f"    🖼️  Сохраняю результат в: {result_path}")

                    # Получаем аннотированное изображение
                    annotated_img_raw = r.plot(
                        line_width=2,
                        font_size=1.0,
                        labels=True,
                        conf=True
                    )

                    # Конвертируем в формат для сохранения
                    save_img = self._prepare_image_for_save(annotated_img_raw, img)

                    # Сохраняем изображение
                    saved = False
                    if save_img is not None:
                        try:
                            saved = cv2.imwrite(str(result_path), save_img)
                            if saved:
                                result_image_url = f"{self.static_url_prefix}/results/{result_filename}"
                                print(f"    💾 Изображение сохранено успешно!")

                                # Проверяем размер файла
                                if result_path.exists():
                                    file_size = result_path.stat().st_size
                                    print(f"    📁 Размер файла: {file_size} байт")
                            else:
                                print(f"    ⚠️  Не удалось сохранить файл!")
                        except Exception as e_save:
                            print(f"    ⚠️  Ошибка сохранения: {e_save}")
                            saved = False

                    if not saved:
                        # Fallback: сохраняем оригинал
                        fallback_path = self.results_dir / f"fallback_{result_id}.jpg"
                        cv2.imwrite(str(fallback_path), img)
                        result_image_url = f"{self.static_url_prefix}/results/{fallback_path.name}"
                        print(f"    ⚠️  Сохранён оригинал как fallback: {fallback_path}")

                    # Собираем информацию о детекциях
                    for j, box in enumerate(r.boxes):
                        class_id = int(box.cls)
                        class_name = self.model.names.get(class_id, str(class_id))
                        conf = float(box.conf)
                        bbox = box.xyxy[0].tolist()

                        detections.append({
                            "class": class_name,
                            "class_name": class_name,
                            "confidence": round(conf, 4),
                            "bbox": [round(val, 2) for val in bbox]
                        })

                        # Выводим первые 5 детекций для отладки
                        if j < 5:
                            print(f"      {j+1}. {class_name}: {conf:.3f}")

                    if len(r.boxes) > 5:
                        print(f"      ... и ещё {len(r.boxes)-5} объектов")

                else:
                    print(f"    ⚠️  Объекты не обнаружены")
                    # Сохраняем оригинал
                    result_id = str(uuid.uuid4())[:8]
                    result_filename = f"no_detections_{result_id}.jpg"
                    result_path = self.results_dir / result_filename
                    cv2.imwrite(str(result_path), img)
                    result_image_url = f"{self.static_url_prefix}/results/{result_filename}"
                    print(f"    💾 Сохранено оригинальное изображение")

            # 4) Логирование результата
            total_time_ms = (time.time() - start_time) * 1000

            print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
            print(f"   Найдено объектов: {len(detections)}")
            print(f"   Время обработки: {total_time_ms:.1f} мс")

            if detections:
                # Группируем по классам
                class_stats = {}
                for det in detections:
                    cls = det['class']
                    class_stats[cls] = class_stats.get(cls, 0) + 1

                print(f"   Распределение по классам:")
                for cls, count in class_stats.items():
                    print(f"     - {cls}: {count}")

            # 5) Сохраняем в историю
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "image": str(image_path),
                "detections": len(detections),
                "time_ms": total_time_ms,
                "result_image": result_image_url
            }

            self.history.append(log_entry)
            if len(self.history) > 100:
                self.history = self.history[-100:]

            # 6) Сохраняем лог в файл
            self._save_log(log_entry)

            # 7) Возвращаем результат
            result = {
                "success": True,
                "detections": detections,
                "count": len(detections),
                "processing_time_ms": round(total_time_ms, 2),
                "timestamp": datetime.now().isoformat(),
            }

            if result_image_url:
                result["result_image"] = result_image_url

            print(f"\n✅ ОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
            print(f"{'=' * 60}")

            return result

        except Exception as e:
            import traceback
            error_msg = f"Критическая ошибка в predict(): {str(e)}"
            print(f"\n❌ {error_msg}")
            traceback.print_exc()

            return {
                "success": False,
                "error": error_msg,
                "count": 0,
                "detections": [],
                "processing_time_ms": 0
            }

    def _prepare_image_for_save(self, annotated_img_raw, original_img):
        """Подготовка изображения для сохранения"""
        try:
            # Если это PIL Image, конвертируем в NumPy
            if Image is not None and isinstance(annotated_img_raw, Image.Image):
                arr = np.array(annotated_img_raw)
            else:
                arr = np.asarray(annotated_img_raw)

            # YOLOv8 .plot() возвращает RGB, cv2.imwrite ожидает BGR
            if arr.ndim == 3 and arr.shape[2] == 3:
                try:
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                except:
                    return arr  # Если не удалось конвертировать
            else:
                return arr
        except:
            # Fallback: возвращаем оригинал
            return original_img.copy()

    def _save_log(self, log_entry: dict):
        """Сохраняет лог детекции"""
        try:
            log_filename = f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = self.logs_dir / log_filename

            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)

            print(f"📝 Лог сохранён: {log_path}")
        except Exception as e:
            print(f"⚠️  Не удалось сохранить лог: {e}")

    def get_stats(self):
        """Получить простую статистику"""
        if not self.history:
            return {
                "total_predictions": 0,
                "avg_time_ms": 0,
                "last_prediction": None,
                "model_classes": len(self.model.names),
                "model_version": "Дообученная для городских изображений" if "yolov8_finetuned_city" in self.model_path else "Стандартная"
            }

        avg_time = sum(h["time_ms"] for h in self.history) / len(self.history)

        return {
            "total_predictions": len(self.history),
            "avg_time_ms": round(avg_time, 2),
            "last_prediction": self.history[-1]["timestamp"],
            "model_classes": len(self.model.names),
            "model_version": "Дообученная для городских изображений" if "yolov8_finetuned_city" in self.model_path else "Стандартная"
        }

    def get_detailed_stats(self):
        """Получить подробную статистику"""
        if not self.history:
            return {
                "total": 0,
                "model_info": {
                    "num_classes": len(self.model.names),
                    "model_version": "Дообученная для городских изображений" if "yolov8_finetuned_city" in self.model_path else "Стандартная"
                }
            }

        stats = {
            "total_predictions": len(self.history),
            "detection_summary": {
                "with_detections": sum(1 for h in self.history if h.get("detections", 0) > 0),
                "without_detections": sum(1 for h in self.history if h.get("detections", 0) == 0),
            },
            "avg_processing_time_ms": round(sum(h["time_ms"] for h in self.history) / len(self.history), 2),
            "recent_predictions": self.history[-5:] if len(self.history) >= 5 else self.history,
            "model_info": {
                "num_classes": len(self.model.names),
                "class_names": list(self.model.names.values()),
                "model_version": "Дообученная для городских изображений" if "yolov8_finetuned_city" in self.model_path else "Стандартная",
                "model_path": self.model_path
            }
        }

        return stats

    def get_model_info(self):
        """Получить информацию о модели"""
        if "yolov8_finetuned_city" in self.model_path:
            model_version = "Дообученная для городских изображений"
            accuracy = "Специализируется на обнаружении транспорта в городской среде"
        elif "yolov8_refined" in self.model_path:
            model_version = "Улучшенная (mAP50: 0.82)"
            accuracy = "mAP50: 0.82"
        else:
            model_version = "Стандартная"
            accuracy = "mAP50: 0.716"

        return {
            "num_classes": len(self.model.names),
            "classes": {k: v for k, v in self.model.names.items()},
            "model_path": self.model_path,
            "device": self.device,
            "model_version": model_version,
            "description": "ADAS Object Detection Model",
            "accuracy": accuracy
        }

# Тестирование детектора
if __name__ == "__main__":
    print("🧪 Тестирование детектора с ДООБУЧЕННОЙ моделью...")

    # Создаем тестовую папку static если её нет
    test_static_dir = Path(__file__).parent.parent / "api" / "static"
    test_static_dir.mkdir(parents=True, exist_ok=True)

    # Инициализируем детектор
    detector = SimpleADASDetector(
        static_dir=str(test_static_dir),
        static_url_prefix="/static"
    )

    print("\n📊 Информация о модели:")
    model_info = detector.get_model_info()
    print(f"  • Количество классов: {model_info['num_classes']}")
    print(f"  • Версия модели: {model_info['model_version']}")
    print(f"  • Точность: {model_info['accuracy']}")
    print(f"  • Устройство: {model_info['device']}")
    print(f"  • Путь к модели: {model_info['model_path']}")

    # Тестовое изображение
    test_image = Path(__file__).parent.parent / "test_image.jpg"
    if test_image.exists():
        print(f"\n🔍 Тестирую на изображении: {test_image.name}")
        result = detector.predict(str(test_image))
        print(f"\n📊 Результат теста:")
        print(f"  • Успех: {result['success']}")
        print(f"  • Найдено объектов: {result['count']}")
        print(f"  • Время обработки: {result['processing_time_ms']} мс")

        if result.get('result_image'):
            print(f"  • Результат сохранён: {result['result_image']}")
    else:
        print(f"\n⚠️  Тестовое изображение не найдено: {test_image}")
        print("   Создайте файл test_image.jpg в корневой папке для теста")

    print("\n✅ Тестирование завершено!")