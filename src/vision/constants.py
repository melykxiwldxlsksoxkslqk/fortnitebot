"""
Базовые константы и кэши для модуля Vision.
"""

import os
from typing import Dict, List, Tuple
import numpy as np

# Пути
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
DEBUG_DIR = os.path.join(ROOT_DIR, 'debug')

# Глобальные флаги
_GLOBAL_DEBUG_ALWAYS = False

# ============================================================================
# КЭШИ
# ============================================================================

# Кэш загруженных шаблонов (RGB, optional alpha)
TEMPLATE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

# Кэш последнего найденного прямоугольника по шаблону
LAST_RECT_CACHE: Dict[str, Tuple[int, int, int, int]] = {}

# Кэш хешей шаблонов для быстрого сравнения
TEMPLATE_HASH_CACHE: Dict[str, np.ndarray] = {}

# Адаптивный кэш порогов confidence для каждого шаблона
ADAPTIVE_CONFIDENCE: Dict[str, float] = {}

# Кэш истории совпадений для адаптивного подбора
MATCH_HISTORY: Dict[str, List[Tuple[float, float, float]]] = {}

# ============================================================================
# НАСТРОЙКИ ДЕТЕКЦИИ
# ============================================================================

# Параметры Template Matching
DEFAULT_CONFIDENCE = 0.80
HIGH_CONFIDENCE = 0.92
LOW_CONFIDENCE = 0.55

# Параметры стабильности
STABLE_FRAMES_NEEDED = 1
STABLE_RADIUS_PX = 20

# Пирамида изображений
PYRAMID_LEVELS = 2

# Multi-scale параметры
MULTI_SCALE_DEFAULT = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
ROTATION_ANGLES = [-15, -10, -5, -2, 0, 2, 5, 10, 15]

# ORB детектор
ORB_FEATURES = 800
ORB_MATCH_RATIO = 0.75

# Peak ratio для фильтрации ложных срабатываний
PEAK_RATIO_MIN = 1.02


# ============================================================================
# ФУНКЦИИ УПРАВЛЕНИЯ
# ============================================================================

def set_global_debug(enabled: bool) -> None:
    """Включает/выключает глобальный режим отладки."""
    global _GLOBAL_DEBUG_ALWAYS
    _GLOBAL_DEBUG_ALWAYS = bool(enabled)


def is_debug_enabled() -> bool:
    """Возвращает текущее состояние режима отладки."""
    return _GLOBAL_DEBUG_ALWAYS


def clear_caches() -> None:
    """Очищает все кэши модуля."""
    TEMPLATE_CACHE.clear()
    LAST_RECT_CACHE.clear()
    TEMPLATE_HASH_CACHE.clear()
    ADAPTIVE_CONFIDENCE.clear()
    MATCH_HISTORY.clear()
