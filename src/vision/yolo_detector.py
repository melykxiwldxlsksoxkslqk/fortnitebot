"""
YOLO детектор объектов.

Улучшенный модуль с поддержкой:
- Кэширования детекций
- Адаптивного порога confidence
- Временного сглаживания (tracking)
- Мета-детекции UI элементов Fortnite
"""

import os
import time
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

# YOLO модель (опционально)
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    _YOLO = None
    YOLO_AVAILABLE = False

from .constants import ROOT_DIR

# Глобальная модель
_yolo_model = None
_model_path = None

# Кэш детекций
_detection_cache: Dict[str, Tuple[float, List[Dict]]] = {}
_CACHE_TTL = 0.1  # 100ms

# Tracking буфер для сглаживания
_tracking_buffer: Dict[str, deque] = {}
_TRACKING_WINDOW = 5


# ============================================================================
# FORTNITE UI CLASSES
# ============================================================================

# Классы для обучения модели на UI элементах Fortnite
FORTNITE_UI_CLASSES = {
    0: 'play_button',
    1: 'search_icon',
    2: 'search_input',
    3: 'select_button',
    4: 'island_card',
    5: 'lobby_tab',
    6: 'creative_tab',
    7: 'battle_royale_tab',
    8: 'settings_icon',
    9: 'loading_spinner',
    10: 'error_dialog',
    11: 'confirm_button',
    12: 'cancel_button',
    13: 'player_marker',
    14: 'health_bar',
    15: 'shield_bar',
    16: 'minimap',
    17: 'inventory_slot',
    18: 'weapon_icon',
    19: 'ammo_count',
}


@dataclass
class Detection:
    """Структурированная детекция."""
    cls: int
    name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    timestamp: float = field(default_factory=time.time)
    
    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)
    
    @property
    def width(self) -> int:
        return self.x2 - self.x1
    
    @property
    def height(self) -> int:
        return self.y2 - self.y1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def to_dict(self) -> Dict:
        return {
            "cls": self.cls,
            "name": self.name,
            "conf": self.confidence,
            "xyxy": (self.x1, self.y1, self.x2, self.y2),
            "center": self.center,
            "width": self.width,
            "height": self.height,
        }


# ============================================================================
# MODEL LOADING
# ============================================================================

def yolo_load_model(weights_path: str = None, force: bool = False) -> object:
    """
    Загружает YOLO модель.
    
    Args:
        weights_path: Путь к файлу весов (по умолчанию: config/yolo/model.pt)
        force: Принудительная перезагрузка
    
    Returns:
        YOLO модель
    """
    global _yolo_model, _model_path
    
    if _yolo_model is not None and not force:
        return _yolo_model
    
    if not YOLO_AVAILABLE:
        raise RuntimeError("ultralytics не установлен. Установите: pip install ultralytics")
    
    if weights_path is None:
        # Пробуем несколько путей
        possible_paths = [
            os.path.join(ROOT_DIR, 'config', 'yolo', 'model.pt'),
            os.path.join(ROOT_DIR, 'models', 'yolo', 'fortnite_ui.pt'),
            os.path.join(ROOT_DIR, 'weights', 'yolov8n.pt'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                weights_path = path
                break
    
    if weights_path is None or not os.path.exists(weights_path):
        # Используем предобученную модель YOLOv8n как fallback
        weights_path = 'yolov8n.pt'
    
    try:
        _yolo_model = _YOLO(weights_path)
        _model_path = weights_path
        return _yolo_model
    except Exception as e:
        raise RuntimeError(f"Не удалось загрузить YOLO модель: {e}")


def yolo_is_loaded() -> bool:
    """Проверяет, загружена ли модель YOLO."""
    return _yolo_model is not None


def yolo_unload() -> None:
    """Выгружает модель YOLO для освобождения памяти."""
    global _yolo_model, _model_path
    _yolo_model = None
    _model_path = None
    _detection_cache.clear()
    _tracking_buffer.clear()


def yolo_get_model_info() -> Dict:
    """Получает информацию о загруженной модели."""
    if _yolo_model is None:
        return {"loaded": False}
    
    return {
        "loaded": True,
        "path": _model_path,
        "names": getattr(_yolo_model, 'names', {}),
        "task": getattr(_yolo_model, 'task', 'detect'),
    }


# ============================================================================
# DETECTION
# ============================================================================

def yolo_detect(
    frame_bgr: np.ndarray,
    conf: float = 0.35,
    classes: List[int] = None,
    use_cache: bool = True,
    nms_iou: float = 0.45
) -> List[Dict]:
    """
    Выполняет детекцию YOLO по кадру BGR.
    
    Args:
        frame_bgr: BGR изображение
        conf: Минимальный порог confidence
        classes: Список классов для фильтрации (None = все)
        use_cache: Использовать кэш детекций
        nms_iou: IoU порог для NMS
    
    Returns:
        Список детекций: [{"cls": int, "name": str, "conf": float, "xyxy": (x1,y1,x2,y2), "center": (cx, cy)}]
    """
    global _yolo_model
    
    if frame_bgr is None:
        return []
    
    # Проверяем кэш
    if use_cache:
        cache_key = f"{id(frame_bgr)}_{conf}_{classes}"
        if cache_key in _detection_cache:
            cached_time, cached_result = _detection_cache[cache_key]
            if time.time() - cached_time < _CACHE_TTL:
                return cached_result
    
    # Загружаем модель если нужно
    if _yolo_model is None:
        try:
            yolo_load_model()
        except Exception:
            return []
    
    if _yolo_model is None:
        return []
    
    try:
        results = _yolo_model.predict(
            frame_bgr,
            conf=conf,
            iou=nms_iou,
            verbose=False,
            classes=classes
        )
        
        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                confidence = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                
                # Центр объекта
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # Имя класса
                name = r.names.get(cls_id, str(cls_id))
                
                detections.append({
                    "cls": cls_id,
                    "name": name,
                    "conf": confidence,
                    "xyxy": (int(x1), int(y1), int(x2), int(y2)),
                    "center": (cx, cy),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                })
        
        # Сохраняем в кэш
        if use_cache:
            _detection_cache[cache_key] = (time.time(), detections)
            # Очищаем старый кэш
            _cleanup_cache()
        
        return detections
    except Exception:
        return []


def _cleanup_cache():
    """Очищает устаревший кэш."""
    now = time.time()
    expired = [k for k, (t, _) in _detection_cache.items() if now - t > _CACHE_TTL * 10]
    for k in expired:
        del _detection_cache[k]


# ============================================================================
# ADVANCED DETECTION METHODS
# ============================================================================

def yolo_detect_best(
    frame_bgr: np.ndarray,
    class_name: str,
    conf: float = 0.35
) -> Optional[Dict]:
    """
    Находит лучшую (с наибольшей confidence) детекцию указанного класса.
    
    Args:
        frame_bgr: BGR изображение
        class_name: Имя класса для поиска
        conf: Минимальный порог confidence
    
    Returns:
        Детекция или None
    """
    detections = yolo_detect(frame_bgr, conf=conf)
    
    class_name_lower = class_name.lower()
    matching = [d for d in detections if d['name'].lower() == class_name_lower]
    
    if not matching:
        return None
    
    return max(matching, key=lambda d: d['conf'])


def yolo_detect_all_of_class(
    frame_bgr: np.ndarray,
    class_name: str,
    conf: float = 0.35
) -> List[Dict]:
    """
    Находит все детекции указанного класса.
    
    Args:
        frame_bgr: BGR изображение
        class_name: Имя класса для поиска
        conf: Минимальный порог confidence
    
    Returns:
        Список детекций
    """
    detections = yolo_detect(frame_bgr, conf=conf)
    class_name_lower = class_name.lower()
    return [d for d in detections if d['name'].lower() == class_name_lower]


def yolo_count_class(
    frame_bgr: np.ndarray,
    class_name: str,
    conf: float = 0.35
) -> int:
    """Подсчитывает количество объектов указанного класса."""
    return len(yolo_detect_all_of_class(frame_bgr, class_name, conf))


def yolo_detect_in_region(
    frame_bgr: np.ndarray,
    region: Tuple[int, int, int, int],
    conf: float = 0.35,
    classes: List[int] = None
) -> List[Dict]:
    """
    Детекция в определённом регионе изображения.
    
    Args:
        frame_bgr: BGR изображение
        region: (x, y, width, height) региона
        conf: Минимальный порог
        classes: Классы для фильтрации
        
    Returns:
        Детекции в регионе
    """
    x, y, w, h = region
    roi = frame_bgr[y:y+h, x:x+w]
    
    detections = yolo_detect(roi, conf=conf, classes=classes, use_cache=False)
    
    # Корректируем координаты на offset
    for d in detections:
        x1, y1, x2, y2 = d['xyxy']
        d['xyxy'] = (x1 + x, y1 + y, x2 + x, y2 + y)
        cx, cy = d['center']
        d['center'] = (cx + x, cy + y)
    
    return detections


def yolo_detect_with_tracking(
    frame_bgr: np.ndarray,
    class_name: str,
    conf: float = 0.35
) -> Optional[Dict]:
    """
    Детекция с временным сглаживанием (tracking).
    
    Использует буфер последних N детекций для стабилизации.
    
    Args:
        frame_bgr: BGR изображение
        class_name: Имя класса
        conf: Минимальный порог
        
    Returns:
        Сглаженная детекция или None
    """
    detection = yolo_detect_best(frame_bgr, class_name, conf)
    
    if class_name not in _tracking_buffer:
        _tracking_buffer[class_name] = deque(maxlen=_TRACKING_WINDOW)
    
    buffer = _tracking_buffer[class_name]
    
    if detection:
        buffer.append(detection)
    
    if not buffer:
        return None
    
    # Усредняем координаты
    avg_x1 = int(np.mean([d['xyxy'][0] for d in buffer]))
    avg_y1 = int(np.mean([d['xyxy'][1] for d in buffer]))
    avg_x2 = int(np.mean([d['xyxy'][2] for d in buffer]))
    avg_y2 = int(np.mean([d['xyxy'][3] for d in buffer]))
    avg_conf = np.mean([d['conf'] for d in buffer])
    
    return {
        "cls": buffer[-1]['cls'],
        "name": class_name,
        "conf": avg_conf,
        "xyxy": (avg_x1, avg_y1, avg_x2, avg_y2),
        "center": ((avg_x1 + avg_x2) // 2, (avg_y1 + avg_y2) // 2),
        "tracked": True,
    }


# ============================================================================
# FORTNITE-SPECIFIC DETECTION
# ============================================================================

def yolo_detect_ui_elements(frame_bgr: np.ndarray, conf: float = 0.4) -> Dict[str, Optional[Dict]]:
    """
    Детектирует все UI элементы Fortnite на кадре.
    
    Returns:
        Словарь {element_name: detection or None}
    """
    detections = yolo_detect(frame_bgr, conf=conf)
    
    ui_elements = {
        'play_button': None,
        'search_icon': None,
        'search_input': None,
        'select_button': None,
        'loading_spinner': None,
    }
    
    for d in detections:
        name = d['name'].lower()
        if name in ui_elements and (ui_elements[name] is None or d['conf'] > ui_elements[name]['conf']):
            ui_elements[name] = d
    
    return ui_elements


def yolo_detect_game_state(frame_bgr: np.ndarray, conf: float = 0.35) -> str:
    """
    Определяет текущее состояние игры по детекциям.
    
    Returns:
        'lobby', 'loading', 'ingame', 'menu', 'unknown'
    """
    detections = yolo_detect(frame_bgr, conf=conf)
    names = set(d['name'].lower() for d in detections)
    
    # Логика определения состояния
    if 'loading_spinner' in names:
        return 'loading'
    
    if 'play_button' in names or 'search_icon' in names:
        return 'lobby'
    
    if 'health_bar' in names or 'minimap' in names or 'weapon_icon' in names:
        return 'ingame'
    
    if 'settings_icon' in names or 'confirm_button' in names:
        return 'menu'
    
    return 'unknown'


# ============================================================================
# ADAPTIVE CONFIDENCE
# ============================================================================

class AdaptiveConfidence:
    """
    Адаптивный порог confidence на основе истории детекций.
    """
    
    def __init__(self, initial_conf: float = 0.5, min_conf: float = 0.25, max_conf: float = 0.85):
        self.current_conf = initial_conf
        self.min_conf = min_conf
        self.max_conf = max_conf
        self.success_history: deque = deque(maxlen=20)
    
    def update(self, found: bool) -> None:
        """Обновляет порог на основе успеха."""
        self.success_history.append(found)
        
        if len(self.success_history) < 5:
            return
        
        success_rate = sum(self.success_history) / len(self.success_history)
        
        # Если много успешных детекций - повышаем порог
        if success_rate > 0.8:
            self.current_conf = min(self.max_conf, self.current_conf + 0.02)
        # Если мало - понижаем
        elif success_rate < 0.3:
            self.current_conf = max(self.min_conf, self.current_conf - 0.03)
    
    def get(self) -> float:
        """Возвращает текущий порог."""
        return self.current_conf


# Глобальный адаптивный confidence
_adaptive_conf = AdaptiveConfidence()


def yolo_detect_adaptive(
    frame_bgr: np.ndarray,
    class_name: str = None,
    classes: List[int] = None
) -> List[Dict]:
    """
    Детекция с адаптивным порогом confidence.
    
    Args:
        frame_bgr: BGR изображение
        class_name: Опциональный фильтр по имени класса
        classes: Опциональный фильтр по ID классов
        
    Returns:
        Список детекций
    """
    conf = _adaptive_conf.get()
    detections = yolo_detect(frame_bgr, conf=conf, classes=classes)
    
    if class_name:
        class_name_lower = class_name.lower()
        detections = [d for d in detections if d['name'].lower() == class_name_lower]
    
    _adaptive_conf.update(len(detections) > 0)
    
    return detections


# ============================================================================
# TRAINING HELPERS
# ============================================================================

def prepare_training_dataset(screenshots_dir: str, output_dir: str) -> bool:
    """
    Подготовка датасета для обучения YOLO на UI элементах.
    
    Args:
        screenshots_dir: Папка со скриншотами
        output_dir: Папка для выходного датасета
        
    Returns:
        True если успешно
    """
    import shutil
    
    try:
        os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
        
        # Создаём data.yaml
        data_yaml = f"""
path: {output_dir}
train: images/train
val: images/val

names:
"""
        for cls_id, name in FORTNITE_UI_CLASSES.items():
            data_yaml += f"  {cls_id}: {name}\n"
        
        with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
            f.write(data_yaml)
        
        return True
    except Exception:
        return False


def train_yolo_model(
    data_yaml: str,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    name: str = 'fortnite_ui'
) -> Optional[str]:
    """
    Обучение YOLO модели на кастомном датасете.
    
    Args:
        data_yaml: Путь к data.yaml
        epochs: Количество эпох
        imgsz: Размер изображений
        batch: Размер батча
        name: Имя эксперимента
        
    Returns:
        Путь к лучшим весам или None
    """
    if not YOLO_AVAILABLE:
        return None
    
    try:
        model = _YOLO('yolov8n.pt')
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name=name,
            patience=20,
            save=True,
            plots=True,
        )
        
        # Путь к лучшим весам
        best_weights = os.path.join(results.save_dir, 'weights', 'best.pt')
        if os.path.exists(best_weights):
            return best_weights
        
        return None
    except Exception:
        return None
        
        return detections
    except Exception:
        return []


def yolo_detect_best(frame_bgr: np.ndarray, class_name: str, conf: float = 0.35) -> Optional[Dict]:
    """
    Находит лучшую (с наибольшей confidence) детекцию указанного класса.
    
    Args:
        frame_bgr: BGR изображение
        class_name: Имя класса для поиска
        conf: Минимальный порог confidence
    
    Returns:
        Детекция или None
    """
    detections = yolo_detect(frame_bgr, conf=conf)
    
    class_name_lower = class_name.lower()
    matching = [d for d in detections if d['name'].lower() == class_name_lower]
    
    if not matching:
        return None
    
    return max(matching, key=lambda d: d['conf'])


def yolo_detect_all_of_class(frame_bgr: np.ndarray, class_name: str, conf: float = 0.35) -> List[Dict]:
    """
    Находит все детекции указанного класса.
    
    Args:
        frame_bgr: BGR изображение
        class_name: Имя класса для поиска
        conf: Минимальный порог confidence
    
    Returns:
        Список детекций
    """
    detections = yolo_detect(frame_bgr, conf=conf)
    class_name_lower = class_name.lower()
    return [d for d in detections if d['name'].lower() == class_name_lower]


def yolo_count_class(frame_bgr: np.ndarray, class_name: str, conf: float = 0.35) -> int:
    """Подсчитывает количество объектов указанного класса."""
    return len(yolo_detect_all_of_class(frame_bgr, class_name, conf))


def yolo_is_loaded() -> bool:
    """Проверяет, загружена ли модель YOLO."""
    return _yolo_model is not None


def yolo_unload() -> None:
    """Выгружает модель YOLO для освобождения памяти."""
    global _yolo_model
    _yolo_model = None
