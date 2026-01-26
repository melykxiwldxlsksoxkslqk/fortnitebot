"""
Template-based Navigation - навигация по шаблонам изображений.

Этот модуль ищет UI элементы по скриншотам и возвращает их координаты
для точного клика/наведения.
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from ..core import get_logger, ROOT_DIR

logger = get_logger(__name__)

# Папка с шаблонами
TEMPLATES_DIR = os.path.join(ROOT_DIR, 'assets')

# Кэш загруженных шаблонов
_template_cache: Dict[str, np.ndarray] = {}


@dataclass
class TemplateMatch:
    """Результат поиска шаблона."""
    name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float
    
    @property
    def center(self) -> Tuple[int, int]:
        """Центр найденного элемента."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """(x, y, width, height)"""
        return (self.x, self.y, self.width, self.height)


def load_template(name: str) -> Optional[np.ndarray]:
    """
    Загружает шаблон по имени.
    
    Args:
        name: Имя файла шаблона (например 'search_discover_bar.png')
        
    Returns:
        Grayscale изображение шаблона или None
    """
    if name in _template_cache:
        return _template_cache[name]
    
    # Пробуем разные расширения
    for ext in ['', '.png', '.jpg', '.jpeg']:
        path = os.path.join(TEMPLATES_DIR, name + ext if not name.endswith(('.png', '.jpg')) else name)
        if os.path.exists(path):
            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                _template_cache[name] = template
                logger.debug(f"Шаблон загружен: {name} ({template.shape[1]}x{template.shape[0]})")
                return template
    
    logger.warning(f"Шаблон не найден: {name}")
    return None


def find_template(
    img: np.ndarray,
    template_name: str,
    threshold: float = 0.7,
    roi: Optional[Tuple[float, float, float, float]] = None,
    scales: List[float] = None
) -> Optional[TemplateMatch]:
    """
    Ищет шаблон на изображении.
    
    Args:
        img: BGR изображение экрана
        template_name: Имя файла шаблона
        threshold: Минимальная уверенность (0.0-1.0)
        roi: Область поиска (rel_x0, rel_y0, rel_x1, rel_y1) в долях от размера
        scales: Масштабы для multi-scale matching (по умолчанию [0.5, 0.75, 1.0, 1.25, 1.5])
        
    Returns:
        TemplateMatch или None
    """
    template = load_template(template_name)
    if template is None:
        return None
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    # Применяем ROI если задан
    roi_x0, roi_y0 = 0, 0
    if roi:
        roi_x0 = int(w * roi[0])
        roi_y0 = int(h * roi[1])
        roi_x1 = int(w * roi[2])
        roi_y1 = int(h * roi[3])
        gray = gray[roi_y0:roi_y1, roi_x0:roi_x1]
    
    if scales is None:
        scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    
    best_match = None
    best_val = 0
    
    for scale in scales:
        # Масштабируем шаблон
        scaled_template = cv2.resize(template, None, fx=scale, fy=scale)
        th, tw = scaled_template.shape[:2]
        
        # Пропускаем если шаблон больше изображения
        if th > gray.shape[0] or tw > gray.shape[1]:
            continue
        
        # Template matching
        result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val and max_val >= threshold:
            best_val = max_val
            best_match = TemplateMatch(
                name=template_name,
                x=max_loc[0] + roi_x0,
                y=max_loc[1] + roi_y0,
                width=tw,
                height=th,
                confidence=max_val
            )
    
    if best_match:
        logger.info(f"Найден шаблон '{template_name}': {best_match.center} (conf={best_match.confidence:.3f})")
    
    return best_match


def find_all_templates(
    img: np.ndarray,
    template_name: str,
    threshold: float = 0.7,
    roi: Optional[Tuple[float, float, float, float]] = None,
    max_results: int = 10
) -> List[TemplateMatch]:
    """
    Ищет ВСЕ вхождения шаблона на изображении.
    
    Полезно для поиска нескольких карточек островов.
    """
    template = load_template(template_name)
    if template is None:
        return []
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    roi_x0, roi_y0 = 0, 0
    if roi:
        roi_x0 = int(w * roi[0])
        roi_y0 = int(h * roi[1])
        roi_x1 = int(w * roi[2])
        roi_y1 = int(h * roi[3])
        gray = gray[roi_y0:roi_y1, roi_x0:roi_x1]
    
    th, tw = template.shape[:2]
    
    if th > gray.shape[0] or tw > gray.shape[1]:
        return []
    
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    
    matches = []
    result_copy = result.copy()
    
    for _ in range(max_results):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_copy)
        
        if max_val < threshold:
            break
        
        match = TemplateMatch(
            name=template_name,
            x=max_loc[0] + roi_x0,
            y=max_loc[1] + roi_y0,
            width=tw,
            height=th,
            confidence=max_val
        )
        matches.append(match)
        
        # Закрашиваем найденную область чтобы не находить повторно
        cv2.rectangle(
            result_copy,
            (max_loc[0] - tw//2, max_loc[1] - th//2),
            (max_loc[0] + tw//2, max_loc[1] + th//2),
            0, -1
        )
    
    return matches


# ============================================================================
# СПЕЦИФИЧНЫЕ ФУНКЦИИ ПОИСКА UI ЭЛЕМЕНТОВ
# ============================================================================

def find_search_discover_bar(img: np.ndarray) -> Optional[TemplateMatch]:
    """
    Ищет поле "Search Discover" на экране.
    
    Находится в верхней левой части экрана Discover.
    """
    # Сначала пробуем шаблон
    match = find_template(
        img, 
        'search_discover_bar.png',
        threshold=0.6,
        roi=(0.0, 0.0, 0.50, 0.20)  # Верхняя левая четверть
    )
    
    if match:
        return match
    
    # Fallback: ищем search_icon.png
    match = find_template(
        img,
        'search_icon.png',
        threshold=0.5,
        roi=(0.0, 0.0, 0.40, 0.20)
    )
    
    return match


def find_island_card(img: np.ndarray) -> Optional[TemplateMatch]:
    """
    Ищет карточку острова в результатах поиска.
    
    Карточки обычно в левой/центральной части экрана.
    """
    match = find_template(
        img,
        'island_card.png',
        threshold=0.6,
        roi=(0.0, 0.15, 0.70, 0.85)  # Основная область экрана
    )
    
    return match


def find_select_button(img: np.ndarray) -> Optional[TemplateMatch]:
    """
    Ищет кнопку SELECT на экране превью острова.
    """
    # Пробуем разные варианты имён
    for name in ['select_button.png', 'submit_button.png']:
        match = find_template(
            img,
            name,
            threshold=0.6,
            roi=(0.20, 0.70, 0.80, 1.0)  # Нижняя часть экрана
        )
        if match:
            return match
    
    return None


def find_play_button(img: np.ndarray) -> Optional[TemplateMatch]:
    """
    Ищет кнопку PLAY для запуска острова.
    """
    # Пробуем разные варианты
    for name in ['play_button_island.png', 'play_button_yellow.png', 'play_button.png']:
        match = find_template(
            img,
            name,
            threshold=0.6,
            roi=(0.20, 0.70, 0.80, 1.0)  # Нижняя часть экрана
        )
        if match:
            return match
    
    return None


def find_input_field(img: np.ndarray) -> Optional[TemplateMatch]:
    """
    Ищет поле ввода кода острова.
    """
    match = find_template(
        img,
        'island_code_input_field.png',
        threshold=0.5,
        roi=(0.10, 0.30, 0.90, 0.70)  # Центральная часть (диалог)
    )
    
    return match


# ============================================================================
# ВИЗУАЛИЗАЦИЯ ДЛЯ DEBUG
# ============================================================================

def draw_matches(img: np.ndarray, matches: List[TemplateMatch]) -> np.ndarray:
    """Рисует найденные элементы на изображении."""
    result = img.copy()
    
    for match in matches:
        # Рисуем прямоугольник
        cv2.rectangle(
            result,
            (match.x, match.y),
            (match.x + match.width, match.y + match.height),
            (0, 255, 0), 2
        )
        
        # Рисуем центр
        cx, cy = match.center
        cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
        
        # Подпись
        label = f"{match.name}: {match.confidence:.2f}"
        cv2.putText(
            result, label,
            (match.x, match.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )
    
    return result
