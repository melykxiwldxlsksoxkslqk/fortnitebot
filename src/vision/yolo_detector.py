"""
YOLO детектор объектов.
"""

import os
import numpy as np
from typing import List, Dict, Optional

# YOLO модель (опционально)
try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    _YOLO = None
    YOLO_AVAILABLE = False

from .constants import ROOT_DIR

_yolo_model = None


def yolo_load_model(weights_path: str = None) -> object:
    """
    Загружает YOLO модель.
    
    Args:
        weights_path: Путь к файлу весов (по умолчанию: config/yolo/model.pt)
    
    Returns:
        YOLO модель
    """
    global _yolo_model
    
    if _yolo_model is not None:
        return _yolo_model
    
    if not YOLO_AVAILABLE:
        raise RuntimeError("ultralytics не установлен")
    
    if weights_path is None:
        weights_path = os.path.join(ROOT_DIR, 'config', 'yolo', 'model.pt')
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO веса не найдены: {weights_path}")
    
    _yolo_model = _YOLO(weights_path)
    return _yolo_model


def yolo_detect(frame_bgr: np.ndarray, conf: float = 0.35, classes: List[int] = None) -> List[Dict]:
    """
    Выполняет детекцию YOLO по кадру BGR.
    
    Args:
        frame_bgr: BGR изображение
        conf: Минимальный порог confidence
        classes: Список классов для фильтрации (None = все)
    
    Returns:
        Список детекций: [{"cls": int, "name": str, "conf": float, "xyxy": (x1,y1,x2,y2), "center": (cx, cy)}]
    """
    global _yolo_model
    
    if _yolo_model is None:
        try:
            yolo_load_model()
        except Exception:
            return []
    
    if _yolo_model is None:
        return []
    
    try:
        results = _yolo_model.predict(frame_bgr, conf=conf, verbose=False, classes=classes)
        
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
                })
        
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
