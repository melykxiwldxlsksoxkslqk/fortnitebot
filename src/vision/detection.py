"""
Детекция элементов на экране.
"""

import time
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Union

from .constants import (
    DEFAULT_CONFIDENCE,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    MULTI_SCALE_DEFAULT,
    LAST_RECT_CACHE,
    MATCH_HISTORY,
)
from .templates import load_template, resolve_asset_path
from .capture import capture_page_bgr, capture_screen


def _resolve_roi_abs(w: int, h: int, roi: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """Преобразует относительный ROI в абсолютные координаты."""
    x0 = max(0, min(w - 1, int(w * roi[0])))
    y0 = max(0, min(h - 1, int(h * roi[1])))
    x1 = max(0, min(w, int(w * roi[2])))
    y1 = max(0, min(h, int(h * roi[3])))
    return x0, y0, x1, y1


def _match_template_multiscale(
    img_gray: np.ndarray,
    template_gray: np.ndarray,
    scales: List[float] = None,
    method: int = cv2.TM_CCOEFF_NORMED
) -> Tuple[float, Tuple[int, int], float]:
    """
    Выполняет template matching на нескольких масштабах.
    
    Returns:
        (best_confidence, (x, y), best_scale)
    """
    if scales is None:
        scales = MULTI_SCALE_DEFAULT
    
    best_val = -1
    best_loc = (0, 0)
    best_scale = 1.0
    
    th, tw = template_gray.shape[:2]
    ih, iw = img_gray.shape[:2]
    
    for scale in scales:
        new_w = int(tw * scale)
        new_h = int(th * scale)
        
        if new_w >= iw or new_h >= ih or new_w < 10 or new_h < 10:
            continue
        
        scaled_template = cv2.resize(template_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        try:
            result = cv2.matchTemplate(img_gray, scaled_template, method)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_scale = scale
        except Exception:
            continue
    
    return (best_val, best_loc, best_scale)


def find_template(
    img_or_page,
    template_path: str,
    confidence: float = None,
    roi: Tuple[float, float, float, float] = None,
    return_all: bool = False
) -> Optional[Tuple[int, int, int, int]]:
    """
    Ищет шаблон на изображении или странице.
    
    Args:
        img_or_page: BGR изображение или Playwright page
        template_path: Путь к шаблону
        confidence: Минимальный порог (по умолчанию DEFAULT_CONFIDENCE)
        roi: Область поиска (x0_frac, y0_frac, x1_frac, y1_frac)
        return_all: Если True, возвращает список всех совпадений
    
    Returns:
        (x, y, w, h) или None если не найдено
    """
    if confidence is None:
        confidence = DEFAULT_CONFIDENCE
    
    # Получаем изображение
    if hasattr(img_or_page, 'screenshot'):
        img = capture_page_bgr(img_or_page)
    else:
        img = img_or_page
    
    h, w = img.shape[:2]
    
    # Применяем ROI
    if roi:
        x0, y0, x1, y1 = _resolve_roi_abs(w, h, roi)
        crop = img[y0:y1, x0:x1]
        offset_x, offset_y = x0, y0
    else:
        crop = img
        offset_x, offset_y = 0, 0
    
    # Загружаем шаблон
    template_bgr, alpha = load_template(template_path)
    
    # Конвертируем в grayscale
    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    # Multi-scale matching
    best_val, best_loc, best_scale = _match_template_multiscale(img_gray, template_gray)
    
    if best_val < confidence:
        return None
    
    # Вычисляем размер найденного элемента
    th, tw = template_gray.shape[:2]
    found_w = int(tw * best_scale)
    found_h = int(th * best_scale)
    
    # Координаты с учётом offset
    x = offset_x + best_loc[0]
    y = offset_y + best_loc[1]
    
    # Кэшируем результат
    resolved = resolve_asset_path(template_path)
    LAST_RECT_CACHE[resolved] = (x, y, found_w, found_h)
    
    # Записываем в историю
    _record_match_success(template_path, best_val, best_scale)
    
    return (x, y, found_w, found_h)


def _record_match_success(template_path: str, confidence: float, scale: float, angle: float = 0.0):
    """Записывает успешное совпадение."""
    resolved = resolve_asset_path(template_path)
    if resolved not in MATCH_HISTORY:
        MATCH_HISTORY[resolved] = []
    MATCH_HISTORY[resolved].append((confidence, scale, angle))
    if len(MATCH_HISTORY[resolved]) > 50:
        MATCH_HISTORY[resolved] = MATCH_HISTORY[resolved][-50:]


def find_template_multi(
    img_or_page,
    template_path: str,
    confidence: float = None,
    max_results: int = 10
) -> List[Tuple[int, int, int, int, float]]:
    """
    Находит все вхождения шаблона.
    
    Returns:
        Список [(x, y, w, h, confidence), ...]
    """
    if confidence is None:
        confidence = DEFAULT_CONFIDENCE
    
    if hasattr(img_or_page, 'screenshot'):
        img = capture_page_bgr(img_or_page)
    else:
        img = img_or_page
    
    template_bgr, _ = load_template(template_path)
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    th, tw = template_gray.shape[:2]
    
    result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    locations = np.where(result >= confidence)
    matches = []
    
    for pt in zip(*locations[::-1]):
        conf = float(result[pt[1], pt[0]])
        matches.append((pt[0], pt[1], tw, th, conf))
    
    # Сортируем по confidence и NMS
    matches.sort(key=lambda x: x[4], reverse=True)
    
    # Non-maximum suppression
    final_matches = []
    for m in matches:
        is_duplicate = False
        for f in final_matches:
            if abs(m[0] - f[0]) < tw // 2 and abs(m[1] - f[1]) < th // 2:
                is_duplicate = True
                break
        if not is_duplicate:
            final_matches.append(m)
            if len(final_matches) >= max_results:
                break
    
    return final_matches


def wait_for_template(
    page,
    template_path: str,
    timeout: float = 10.0,
    confidence: float = None,
    poll_interval: float = 0.3
) -> Optional[Tuple[int, int, int, int]]:
    """
    Ожидает появления шаблона на странице.
    
    Args:
        page: Playwright page
        template_path: Путь к шаблону
        timeout: Таймаут в секундах
        confidence: Минимальный порог
        poll_interval: Интервал проверки
    
    Returns:
        (x, y, w, h) или None если таймаут
    """
    start = time.time()
    while time.time() - start < timeout:
        result = find_template(page, template_path, confidence)
        if result:
            return result
        time.sleep(poll_interval)
    return None


def detect_color_region(
    img_bgr: np.ndarray,
    color_name: str,
    roi: Tuple[float, float, float, float] = None,
    min_area: int = 100
) -> Optional[Tuple[int, int, int, int]]:
    """
    Находит область с указанным цветом.
    
    Поддерживаемые цвета: 'green', 'red', 'blue', 'yellow', 'cyan', 'white', 'orange', 'purple'
    
    Returns:
        (x, y, w, h) или None
    """
    h, w = img_bgr.shape[:2]
    
    if roi:
        x0, y0, x1, y1 = _resolve_roi_abs(w, h, roi)
        crop = img_bgr[y0:y1, x0:x1]
    else:
        x0, y0 = 0, 0
        crop = img_bgr
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        'green': [(35, 50, 50), (85, 255, 255)],
        'red': [(0, 100, 100), (10, 255, 255)],
        'blue': [(100, 100, 100), (130, 255, 255)],
        'yellow': [(20, 100, 100), (35, 255, 255)],
        'cyan': [(85, 100, 100), (100, 255, 255)],
        'white': [(0, 0, 200), (180, 30, 255)],
        'orange': [(10, 100, 100), (20, 255, 255)],
        'purple': [(130, 50, 50), (160, 255, 255)],
    }
    
    color_name = color_name.lower()
    if color_name not in color_ranges:
        return None
    
    lower, upper = color_ranges[color_name]
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    
    if color_name == 'red':
        mask2 = cv2.inRange(hsv, np.array([170, 100, 100], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        mask = cv2.bitwise_or(mask, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None
    
    rx, ry, rw, rh = cv2.boundingRect(largest)
    return (x0 + rx, y0 + ry, rw, rh)


def detect_text_region(
    img_bgr: np.ndarray,
    roi: Tuple[float, float, float, float] = None,
    min_area: int = 500
) -> List[Dict]:
    """
    Находит текстовые области на изображении.
    
    Returns:
        Список [{"bbox": (x,y,w,h), "area": float}]
    """
    h, w = img_bgr.shape[:2]
    
    if roi:
        x0, y0, x1, y1 = _resolve_roi_abs(w, h, roi)
        crop = img_bgr[y0:y1, x0:x1]
    else:
        x0, y0 = 0, 0
        crop = img_bgr
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # MSER для текста
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    results = []
    for region in regions:
        x, y, rw, rh = cv2.boundingRect(region)
        area = rw * rh
        if area < min_area:
            continue
        
        aspect = rw / rh if rh > 0 else 0
        if aspect < 0.1 or aspect > 15:
            continue
        
        results.append({
            "bbox": (x0 + x, y0 + y, rw, rh),
            "area": area
        })
    
    return results


def detect_button(
    img_bgr: np.ndarray,
    color: str = None,
    min_size: Tuple[int, int] = (50, 20),
    max_size: Tuple[int, int] = (500, 200)
) -> List[Dict]:
    """
    Находит кнопки на изображении по форме и цвету.
    
    Returns:
        Список [{"bbox": (x,y,w,h), "color": str, "center": (cx,cy)}]
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    # Края для поиска прямоугольников
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Фильтр по размеру
        if w < min_size[0] or h < min_size[1]:
            continue
        if w > max_size[0] or h > max_size[1]:
            continue
        
        # Прямоугольность
        area = cv2.contourArea(cnt)
        rect_area = w * h
        if rect_area == 0 or area / rect_area < 0.5:
            continue
        
        # Определяем цвет
        roi_hsv = hsv[y:y+h, x:x+w]
        mean_h = np.mean(roi_hsv[:, :, 0])
        mean_s = np.mean(roi_hsv[:, :, 1])
        mean_v = np.mean(roi_hsv[:, :, 2])
        
        detected_color = 'unknown'
        if mean_s < 30:
            detected_color = 'white' if mean_v > 200 else 'gray'
        elif mean_h < 15 or mean_h > 165:
            detected_color = 'red'
        elif 15 <= mean_h < 35:
            detected_color = 'yellow'
        elif 35 <= mean_h < 85:
            detected_color = 'green'
        elif 85 <= mean_h < 130:
            detected_color = 'blue'
        elif 130 <= mean_h < 165:
            detected_color = 'purple'
        
        if color and detected_color != color:
            continue
        
        results.append({
            "bbox": (x, y, w, h),
            "color": detected_color,
            "center": (x + w // 2, y + h // 2)
        })
    
    return results


def smart_find_element(
    page_or_img,
    methods: List[str] = None,
    template_path: str = None,
    color: str = None,
    element_type: str = None,
    confidence: float = None
) -> Optional[Dict]:
    """
    Умный поиск элемента несколькими методами.
    
    Args:
        page_or_img: Playwright page или BGR изображение
        methods: Список методов ['template', 'yolo', 'color', 'button']
        template_path: Путь к шаблону (для method='template')
        color: Цвет элемента (для method='color')
        element_type: Тип элемента (для method='yolo')
        confidence: Минимальный порог
    
    Returns:
        {"method": str, "bbox": (x,y,w,h), "center": (cx,cy), "confidence": float} или None
    """
    if methods is None:
        methods = ['template', 'color', 'button']
    
    if hasattr(page_or_img, 'screenshot'):
        img = capture_page_bgr(page_or_img)
    else:
        img = page_or_img
    
    if confidence is None:
        confidence = DEFAULT_CONFIDENCE
    
    for method in methods:
        try:
            if method == 'template' and template_path:
                result = find_template(img, template_path, confidence)
                if result:
                    x, y, w, h = result
                    return {
                        "method": "template",
                        "bbox": result,
                        "center": (x + w // 2, y + h // 2),
                        "confidence": confidence
                    }
            
            elif method == 'color' and color:
                result = detect_color_region(img, color)
                if result:
                    x, y, w, h = result
                    return {
                        "method": "color",
                        "bbox": result,
                        "center": (x + w // 2, y + h // 2),
                        "confidence": 1.0
                    }
            
            elif method == 'button':
                buttons = detect_button(img, color=color)
                if buttons:
                    btn = buttons[0]
                    return {
                        "method": "button",
                        "bbox": btn["bbox"],
                        "center": btn["center"],
                        "confidence": 1.0,
                        "color": btn["color"]
                    }
            
            elif method == 'yolo' and element_type:
                from .yolo_detector import yolo_detect_best
                result = yolo_detect_best(img, element_type, confidence)
                if result:
                    x1, y1, x2, y2 = result["xyxy"]
                    return {
                        "method": "yolo",
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "center": result["center"],
                        "confidence": result["conf"]
                    }
        except Exception:
            continue
    
    return None
