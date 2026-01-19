"""
Определение состояния экрана.
"""

import time
import cv2
import numpy as np
from typing import Optional, Union, List, Dict, Callable
from enum import Enum, auto

from .capture import capture_page_bgr, capture_screen


class ScreenState(Enum):
    """Состояния экрана для умной детекции."""
    UNKNOWN = auto()
    LOADING = auto()           # Экран загрузки (тёмный, спиннер)
    CONNECTING = auto()        # CONNECTING/LOGGING IN оверлей
    PLANE_SCREEN = auto()      # Зелёный самолёт Xbox
    XBOX_LOADING = auto()      # Xbox Cloud Gaming загрузка
    LOGIN_PAGE = auto()        # Страница входа
    LOBBY = auto()             # Лобби Fortnite
    IN_GAME = auto()           # В игре
    MENU = auto()              # Меню/настройки
    ERROR = auto()             # Ошибка/диалог


# Кэш состояния
_LAST_SCREEN_STATE: ScreenState = ScreenState.UNKNOWN
_LAST_SCREEN_STATE_TIME: float = 0.0


def _get_image(page_or_img) -> np.ndarray:
    """Получает изображение из page или возвращает как есть."""
    if hasattr(page_or_img, 'screenshot'):
        return capture_page_bgr(page_or_img)
    return page_or_img


def _detect_connecting_overlay(img: np.ndarray) -> bool:
    """Детектирует оверлей CONNECTING на изображении."""
    try:
        h, w = img.shape[:2]
        
        def roi_abs(fr):
            x0 = max(0, min(w - 1, int(w * fr[0])))
            y0 = max(0, min(h - 1, int(h * fr[1])))
            x1 = max(0, min(w, int(w * fr[2])))
            y1 = max(0, min(h, int(h * fr[3])))
            return x0, y0, x1, y1
        
        # Расширенные зоны поиска
        rois = [
            (0.00, 0.80, 0.50, 0.99),  # левый-низ
            (0.25, 0.70, 0.75, 0.95),  # центр-низ
            (0.00, 0.70, 1.00, 0.99),  # весь низ
            (0.30, 0.40, 0.70, 0.60),  # центр экрана (loading spinner)
        ]
        
        # Циан/бирюза в HSV — широкий диапазон
        cyan_lower = np.array([60, 40, 100], dtype=np.uint8)
        cyan_upper = np.array([130, 255, 255], dtype=np.uint8)
        
        # Белый текст (тоже часто используется)
        white_lower = np.array([0, 0, 200], dtype=np.uint8)
        white_upper = np.array([180, 40, 255], dtype=np.uint8)
        
        for fr in rois:
            x0, y0, x1, y1 = roi_abs(fr)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = img[y0:y1, x0:x1]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Проверяем циановые пиксели
            mask_cyan = cv2.inRange(hsv, cyan_lower, cyan_upper)
            mask_cyan = cv2.medianBlur(mask_cyan, 3)
            kernel = np.ones((3, 3), np.uint8)
            mask_cyan = cv2.morphologyEx(mask_cyan, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            ratio_cyan = float(np.count_nonzero(mask_cyan)) / float(mask_cyan.size)
            if ratio_cyan > 0.002:
                return True
            
            # Проверяем белый текст на тёмном фоне
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            mean_bright = np.mean(gray)
            if mean_bright < 60:
                mask_white = cv2.inRange(hsv, white_lower, white_upper)
                ratio_white = float(np.count_nonzero(mask_white)) / float(mask_white.size)
                if ratio_white > 0.01:
                    return True
        
        return False
    except Exception:
        return False


def _detect_plane_screen(img: np.ndarray) -> bool:
    """Детектирует plane screen (зелёный самолёт Xbox)."""
    try:
        h, w = img.shape[:2]
        
        # Проверяем что фон тёмный
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness > 80:
            return False
        
        # Центральная зона
        x0 = int(w * 0.15)
        x1 = int(w * 0.85)
        y0 = int(h * 0.20)
        y1 = int(h * 0.80)
        roi = img[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Ярко-зелёный/салатовый диапазон
        green_lower = np.array([35, 50, 100], dtype=np.uint8)
        green_upper = np.array([95, 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        mask_green = cv2.medianBlur(mask_green, 3)
        ratio_green = float(np.count_nonzero(mask_green)) / float(mask_green.size)
        
        if ratio_green > 0.001:
            return True
        
        # Ищем характерные линии
        edges = cv2.Canny(gray[y0:y1, x0:x1], 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is not None and len(lines) > 3:
            return True
        
        return False
    except Exception:
        return False


def detect_connecting_overlay_on_page(page) -> bool:
    """Определяет, виден ли на странице оверлей CONNECTING/LOGGING IN."""
    try:
        img = capture_page_bgr(page)
        return _detect_connecting_overlay(img)
    except Exception:
        return False


def detect_plane_screen_on_page(page) -> bool:
    """Определяет, отображается ли экран с зелёным самолётиком Xbox."""
    try:
        img = capture_page_bgr(page)
        return _detect_plane_screen(img)
    except Exception:
        return False


def detect_loading_spinner(page_or_img) -> bool:
    """Определяет наличие спиннера загрузки."""
    try:
        img = _get_image(page_or_img)
        h, w = img.shape[:2]
        
        # Центральная область
        cx, cy = w // 2, h // 2
        roi_size = min(w, h) // 4
        x0 = max(0, cx - roi_size)
        y0 = max(0, cy - roi_size)
        x1 = min(w, cx + roi_size)
        y1 = min(h, cy + roi_size)
        
        roi = img[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Детектируем круги
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20,
            param1=50, param2=30,
            minRadius=15, maxRadius=100
        )
        
        return circles is not None and len(circles[0]) > 0
    except Exception:
        return False


def detect_xbox_loading_screen(page_or_img) -> bool:
    """Детектирует экран загрузки Xbox Cloud Gaming."""
    try:
        img = _get_image(page_or_img)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Xbox зелёный
        xbox_lower = np.array([35, 100, 50], dtype=np.uint8)
        xbox_upper = np.array([85, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, xbox_lower, xbox_upper)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        
        if ratio > 0.05:
            return True
        
        # Тёмный экран с белым текстом
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_bright = np.mean(gray)
        
        if mean_bright < 50:
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            white_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
            if 0.005 < white_ratio < 0.15:
                return True
        
        return False
    except Exception:
        return False


def detect_game_ready(page_or_img) -> bool:
    """Определяет, что игра загрузилась и готова к управлению."""
    try:
        img = _get_image(page_or_img)
        h, w = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_bright = np.mean(gray)
        std_bright = np.std(gray)
        
        if mean_bright < 30 or mean_bright > 250:
            return False
        
        if std_bright < 20:
            return False
        
        # Проверяем UI сверху и снизу
        top_roi = img[0:int(h*0.15), :]
        bottom_roi = img[int(h*0.85):, :]
        
        top_edges = cv2.Canny(cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY), 50, 150)
        bottom_edges = cv2.Canny(cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY), 50, 150)
        
        top_edge_ratio = float(np.count_nonzero(top_edges)) / float(top_edges.size)
        bottom_edge_ratio = float(np.count_nonzero(bottom_edges)) / float(bottom_edges.size)
        
        if top_edge_ratio > 0.02 and bottom_edge_ratio > 0.02:
            return True
        
        return False
    except Exception:
        return False


def _analyze_screen_state(img: np.ndarray) -> ScreenState:
    """Анализирует состояние экрана."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # 1. CONNECTING оверлей
    if _detect_connecting_overlay(img):
        return ScreenState.CONNECTING
    
    # 2. Plane screen
    if _detect_plane_screen(img):
        return ScreenState.PLANE_SCREEN
    
    # 3. Xbox loading
    if mean_brightness < 40:
        xbox_lower = np.array([35, 100, 50], dtype=np.uint8)
        xbox_upper = np.array([85, 255, 255], dtype=np.uint8)
        mask_xbox = cv2.inRange(hsv, xbox_lower, xbox_upper)
        if np.count_nonzero(mask_xbox) / mask_xbox.size > 0.01:
            return ScreenState.XBOX_LOADING
        
        if std_brightness < 30:
            return ScreenState.LOADING
    
    # 4. Страница входа
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([180, 30, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, white_lower, white_upper)
    white_ratio = np.count_nonzero(mask_white) / mask_white.size
    
    if white_ratio > 0.5:
        blue_lower = np.array([100, 50, 50], dtype=np.uint8)
        blue_upper = np.array([130, 255, 255], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
        if np.count_nonzero(mask_blue) / mask_blue.size > 0.005:
            return ScreenState.LOGIN_PAGE
    
    # 5. Игровой экран
    if std_brightness > 40:
        top_ui = img[0:int(h*0.12), :]
        bottom_ui = img[int(h*0.85):, :]
        
        top_edges = cv2.Canny(cv2.cvtColor(top_ui, cv2.COLOR_BGR2GRAY), 50, 150)
        bottom_edges = cv2.Canny(cv2.cvtColor(bottom_ui, cv2.COLOR_BGR2GRAY), 50, 150)
        
        top_edge_ratio = np.count_nonzero(top_edges) / top_edges.size
        bottom_edge_ratio = np.count_nonzero(bottom_edges) / bottom_edges.size
        
        if top_edge_ratio > 0.03 and bottom_edge_ratio > 0.03:
            return ScreenState.IN_GAME
        
        if top_edge_ratio > 0.01 or bottom_edge_ratio > 0.02:
            return ScreenState.LOBBY
    
    # 6. Меню/диалог
    center_roi = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    center_bright = np.mean(cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY))
    edge_bright = (np.mean(gray[:int(h*0.2), :]) + np.mean(gray[int(h*0.8):, :])) / 2
    
    if center_bright > edge_bright * 1.5 and center_bright > 80:
        return ScreenState.MENU
    
    # 7. Ошибка
    red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170, 100, 100], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, red_lower1, red_upper1),
        cv2.inRange(hsv, red_lower2, red_upper2)
    )
    
    if np.count_nonzero(mask_red) / mask_red.size > 0.01:
        return ScreenState.ERROR
    
    return ScreenState.UNKNOWN


def detect_screen_state(page_or_img, use_cache: bool = True) -> ScreenState:
    """
    Интеллектуальное определение текущего состояния экрана.
    
    Args:
        page_or_img: Playwright page или BGR изображение
        use_cache: Использовать кэш
    
    Returns:
        ScreenState
    """
    global _LAST_SCREEN_STATE, _LAST_SCREEN_STATE_TIME
    
    if use_cache and time.time() - _LAST_SCREEN_STATE_TIME < 0.5:
        return _LAST_SCREEN_STATE
    
    try:
        img = _get_image(page_or_img)
        state = _analyze_screen_state(img)
        
        _LAST_SCREEN_STATE = state
        _LAST_SCREEN_STATE_TIME = time.time()
        
        return state
    except Exception:
        return ScreenState.UNKNOWN


def wait_for_screen_state(
    page_or_img_source,
    target_states: Union[ScreenState, List[ScreenState]],
    timeout: float = 30.0,
    poll_interval: float = 0.5
) -> Optional[ScreenState]:
    """
    Ожидает появления одного из целевых состояний экрана.
    
    Args:
        page_or_img_source: Playwright page или callable
        target_states: Ожидаемые состояния
        timeout: Таймаут в секундах
        poll_interval: Интервал проверки
    
    Returns:
        ScreenState если достигнуто, None если таймаут
    """
    if isinstance(target_states, ScreenState):
        target_states = [target_states]
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            if callable(page_or_img_source):
                img = page_or_img_source()
            else:
                img = _get_image(page_or_img_source)
            
            state = detect_screen_state(img, use_cache=False)
            if state in target_states:
                return state
        except Exception:
            pass
        
        time.sleep(poll_interval)
    
    return None


def get_screen_state_info(page_or_img) -> Dict:
    """
    Возвращает детальную информацию о состоянии экрана.
    """
    try:
        img = _get_image(page_or_img)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        avg_color = img.mean(axis=(0, 1)).astype(int)
        
        top_edges = cv2.Canny(gray[:int(h*0.15), :], 50, 150)
        bottom_edges = cv2.Canny(gray[int(h*0.85):, :], 50, 150)
        has_ui = (
            np.count_nonzero(top_edges) / top_edges.size > 0.02 or
            np.count_nonzero(bottom_edges) / bottom_edges.size > 0.02
        )
        
        state = detect_screen_state(img, use_cache=False)
        
        return {
            "state": state.name,
            "state_value": state.value,
            "brightness": brightness,
            "contrast": contrast,
            "has_ui": has_ui,
            "avg_color_bgr": tuple(avg_color.tolist()),
            "width": w,
            "height": h
        }
    except Exception as e:
        return {
            "state": ScreenState.UNKNOWN.name,
            "error": str(e)
        }
