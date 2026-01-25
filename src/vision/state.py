"""
Определение состояния экрана.
"""

import os
import time
import cv2
import numpy as np
from typing import Optional, Union, List, Dict, Callable
from enum import Enum, auto
from datetime import datetime

from .capture import capture_page_bgr, capture_screen
from ..core import get_logger, ROOT_DIR

# Логгер для vision
_vision_logger = get_logger("epicbot.vision")

# Глобальный флаг для включения debug логирования
_VISION_DEBUG = False

# Папка для debug скриншотов
_DEBUG_DIR = os.path.join(ROOT_DIR, 'debugvision')


def set_vision_debug(enabled: bool):
    """Включает/выключает debug логирование vision."""
    global _VISION_DEBUG
    _VISION_DEBUG = enabled
    _vision_logger.info(f"Vision debug режим: {'ВКЛ' if enabled else 'ВЫКЛ'}")
    
    # Создаём папку для debug скриншотов
    if enabled:
        os.makedirs(_DEBUG_DIR, exist_ok=True)


def _save_debug_image(img: np.ndarray, state: str, info: dict = None):
    """Сохраняет debug скриншот с информацией о детекции."""
    if not _VISION_DEBUG:
        return
    
    try:
        # Создаём копию для рисования
        debug_img = img.copy()
        h, w = debug_img.shape[:2]
        
        # Добавляем информацию на изображение
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Состояние
        cv2.putText(debug_img, f"State: {state}", (10, 30), font, font_scale, (0, 255, 0), thickness)
        
        # Время
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(debug_img, f"Time: {timestamp}", (10, 60), font, font_scale, (0, 255, 0), thickness)
        
        # Дополнительная информация
        if info:
            y_offset = 90
            for key, value in info.items():
                text = f"{key}: {value}"
                cv2.putText(debug_img, text, (10, y_offset), font, 0.5, (255, 255, 0), 1)
                y_offset += 25
        
        # Сохраняем
        filename = f"{datetime.now().strftime('%H%M%S')}_{state}.jpg"
        filepath = os.path.join(_DEBUG_DIR, filename)
        cv2.imwrite(filepath, debug_img)
        
    except Exception as e:
        _vision_logger.debug(f"Ошибка сохранения debug изображения: {e}")


def _log_debug(msg: str):
    """Логирует если включен debug режим."""
    if _VISION_DEBUG:
        _vision_logger.debug(f"[VISION] {msg}")


def _log_info(msg: str):
    """Всегда логирует информацию."""
    _vision_logger.info(f"[VISION] {msg}")


class ScreenState(Enum):
    """Состояния экрана для умной детекции."""
    UNKNOWN = auto()
    LOADING = auto()           # Экран загрузки (тёмный, спиннер)
    CONNECTING = auto()        # CONNECTING/LOGGING IN оверлей
    PLANE_SCREEN = auto()      # Зелёный самолёт Xbox
    XBOX_LOADING = auto()      # Xbox Cloud Gaming загрузка
    XBOX_QUEUE = auto()        # Очередь Xbox Cloud Gaming (ждём место на сервере)
    LOGIN_PAGE = auto()        # Страница входа
    TITLE_SCREEN = auto()      # Титульный экран Fortnite (нажми любую клавишу)
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


def _detect_xbox_logo_screen(img: np.ndarray) -> bool:
    """
    Детектирует экран загрузки Xbox с большим логотипом XBOX по центру.
    
    Характеристики:
    - Тёмный фон
    - Логотип Xbox (шар с X) и текст "XBOX" белыми/серыми буквами по центру
    - Маленький зелёный спиннер/индикатор по центру
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Тёмный фон (яркость < 40)
        mean_brightness = np.mean(gray)
        if mean_brightness > 45:
            return False
        
        # Центральная область где должен быть логотип XBOX
        center_y0 = int(h * 0.2)
        center_y1 = int(h * 0.8)
        center_x0 = int(w * 0.3)
        center_x1 = int(w * 0.7)
        center_roi = gray[center_y0:center_y1, center_x0:center_x1]
        
        # Должны быть белые/светлые пиксели (логотип и буквы XBOX) на тёмном фоне
        # Светлые элементы: яркость > 120
        white_mask = center_roi > 120
        white_ratio = np.count_nonzero(white_mask) / white_mask.size
        
        # Логотип + буквы XBOX занимают примерно 5-25% центральной области
        if 0.03 < white_ratio < 0.30:
            # Проверяем контраст - должен быть высокий (светлый логотип на тёмном фоне)
            std_center = np.std(center_roi)
            if std_center > 30:  # Высокий контраст
                return True
        
        return False
    except Exception:
        return False


def _detect_connecting_overlay(img: np.ndarray) -> bool:
    """
    Детектирует оверлей CONNECTING/LOGGING IN на изображении.
    
    Этот оверлей появляется когда Xbox Cloud Gaming подключается к серверу.
    Характеристики: тёмный/полупрозрачный фон с текстом CONNECTING или спиннером.
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Главная страница Xbox обычно яркая (белый фон, много контента)
        # CONNECTING оверлей обычно тёмный
        mean_brightness = np.mean(gray)
        
        # Если экран слишком яркий - это не CONNECTING
        if mean_brightness > 120:
            return False
        
        # Проверяем что основная часть экрана тёмная (оверлей)
        dark_pixels = np.sum(gray < 50) / gray.size
        if dark_pixels < 0.3:  # Минимум 30% тёмных пикселей
            return False
        
        def roi_abs(fr):
            x0 = max(0, min(w - 1, int(w * fr[0])))
            y0 = max(0, min(h - 1, int(h * fr[1])))
            x1 = max(0, min(w, int(w * fr[2])))
            y1 = max(0, min(h, int(h * fr[3])))
            return x0, y0, x1, y1
        
        # Зоны поиска текста CONNECTING (обычно в центре или внизу)
        rois = [
            (0.25, 0.40, 0.75, 0.65),  # центр экрана
            (0.20, 0.75, 0.80, 0.95),  # нижняя часть
        ]
        
        # Циан/бирюза CONNECTING текста
        cyan_lower = np.array([80, 80, 120], dtype=np.uint8)
        cyan_upper = np.array([110, 255, 255], dtype=np.uint8)
        
        for fr in rois:
            x0, y0, x1, y1 = roi_abs(fr)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = img[y0:y1, x0:x1]
            roi_gray = gray[y0:y1, x0:x1]
            
            # ROI тоже должен быть тёмным
            if np.mean(roi_gray) > 80:
                continue
                
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Проверяем циановые пиксели (текст CONNECTING)
            mask_cyan = cv2.inRange(hsv, cyan_lower, cyan_upper)
            mask_cyan = cv2.medianBlur(mask_cyan, 3)
            
            ratio_cyan = float(np.count_nonzero(mask_cyan)) / float(mask_cyan.size)
            # Должно быть немного циановых пикселей (текст), но не слишком много
            if 0.005 < ratio_cyan < 0.15:
                return True
        
        # Проверяем наличие спиннера загрузки в центре
        cx, cy = w // 2, h // 2
        spinner_roi = img[cy - 50:cy + 50, cx - 50:cx + 50] if cy > 50 and cx > 50 else None
        if spinner_roi is not None and spinner_roi.size > 0:
            spinner_gray = cv2.cvtColor(spinner_roi, cv2.COLOR_BGR2GRAY)
            # Спиннер обычно имеет круговую структуру
            circles = cv2.HoughCircles(
                spinner_gray, cv2.HOUGH_GRADIENT, 1, 20,
                param1=50, param2=25,
                minRadius=10, maxRadius=40
            )
            if circles is not None and len(circles[0]) > 0:
                # Проверяем что фон тёмный
                if np.mean(spinner_gray) < 60:
                    return True
        
        return False
    except Exception:
        return False


def _detect_plane_screen(img: np.ndarray) -> bool:
    """Детектирует plane screen (зелёный самолёт Xbox с текстом 'Připravujeme')."""
    try:
        h, w = img.shape[:2]
        
        # Проверяем что фон тёмный
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness > 80:
            return False
        
        # ВАЖНО: Проверяем НЕТ ЛИ большого белого логотипа XBOX по центру
        # Если есть - это Xbox Logo Screen, НЕ plane screen
        center_y0 = int(h * 0.2)
        center_y1 = int(h * 0.7)
        center_x0 = int(w * 0.3)
        center_x1 = int(w * 0.7)
        center_roi = gray[center_y0:center_y1, center_x0:center_x1]
        
        # Белые/светлые пиксели в центре (логотип XBOX)
        white_center = center_roi > 150
        white_ratio_center = np.count_nonzero(white_center) / white_center.size
        
        # Если много белого в центре (> 5%) - это Xbox Logo Screen
        if white_ratio_center > 0.05:
            return False
        
        # Центральная зона для поиска зелёного
        x0 = int(w * 0.15)
        x1 = int(w * 0.85)
        y0 = int(h * 0.20)
        y1 = int(h * 0.80)
        roi = img[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Ярко-зелёный/салатовый диапазон (самолёт и элементы Xbox)
        green_lower = np.array([35, 50, 100], dtype=np.uint8)
        green_upper = np.array([95, 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        mask_green = cv2.medianBlur(mask_green, 3)
        ratio_green = float(np.count_nonzero(mask_green)) / float(mask_green.size)
        
        # ВАЖНО: для plane screen ОБЯЗАТЕЛЬНО должен быть ЗНАЧИТЕЛЬНЫЙ зелёный цвет (самолёт)
        # Маленький спиннер даёт ~0.001-0.003, самолёт даёт > 0.005
        if ratio_green > 0.003:
            return True
        
        return False
    except Exception:
        return False


def _detect_xbox_queue(img: np.ndarray) -> bool:
    """
    Детектирует экран очереди Xbox Game Pass.
    
    Характеристики:
    - Тёмный фон (почти чёрный)
    - Левая часть: текст "XBOX GAME PASS" белым/зелёным + реклама
    - Правая часть: серая панель (предпросмотр игры)
    - Внизу по центру: прогресс-бар очереди (белая/серая полоса)
    - Текст типа "Už to skoro je" или подобный
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Проверяем что основной фон тёмный
        mean_brightness = np.mean(gray)
        if mean_brightness > 80:
            return False
        
        # Левая часть экрана (реклама Xbox Game Pass) - должна быть тёмной с белым текстом
        left_roi = gray[int(h*0.15):int(h*0.7), 0:int(w*0.4)]
        left_mean = np.mean(left_roi)
        if left_mean > 60:
            return False
        
        # Правая часть экрана - серая панель (предпросмотр игры)
        right_roi = gray[int(h*0.1):int(h*0.8), int(w*0.75):]
        right_mean = np.mean(right_roi)
        # Правая часть должна быть заметно светлее (серая панель) или тоже тёмная
        # но с заметным контрастом
        
        # Нижняя часть - прогресс-бар очереди
        bottom_roi = img[int(h*0.85):int(h*0.98), int(w*0.3):int(w*0.7)]
        bottom_gray = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY)
        
        # Ищем горизонтальную полосу (прогресс-бар)
        # Прогресс-бар обычно белый/светло-серый на тёмном фоне
        _, thresh = cv2.threshold(bottom_gray, 100, 255, cv2.THRESH_BINARY)
        white_ratio_bottom = np.count_nonzero(thresh) / thresh.size
        
        # Должно быть немного белого (прогресс-бар и текст)
        if white_ratio_bottom < 0.01 or white_ratio_bottom > 0.5:
            return False
        
        # Проверяем наличие зелёного Xbox цвета в левой части
        left_hsv = hsv[int(h*0.15):int(h*0.6), 0:int(w*0.5)]
        xbox_green_lower = np.array([35, 50, 80], dtype=np.uint8)
        xbox_green_upper = np.array([85, 255, 255], dtype=np.uint8)
        mask_green = cv2.inRange(left_hsv, xbox_green_lower, xbox_green_upper)
        green_ratio = np.count_nonzero(mask_green) / mask_green.size
        
        # Если есть немного зелёного (логотип Xbox, иконки) - это хороший признак
        # Но не обязательно, поэтому проверяем также структуру
        
        # Проверяем белый текст в левой части ("XBOX GAME PASS", описание)
        left_img = img[int(h*0.15):int(h*0.6), 0:int(w*0.5)]
        left_gray_full = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        _, white_thresh = cv2.threshold(left_gray_full, 180, 255, cv2.THRESH_BINARY)
        white_ratio_left = np.count_nonzero(white_thresh) / white_thresh.size
        
        # Должен быть белый текст (1-20% площади)
        if white_ratio_left < 0.005 or white_ratio_left > 0.25:
            return False
        
        # Финальная проверка: тёмный фон + белый текст слева + прогресс-бар снизу
        # Либо есть зелёный Xbox цвет
        if green_ratio > 0.001 or (left_mean < 50 and white_ratio_left > 0.01):
            return True
        
        return False
    except Exception:
        return False


def _detect_title_screen(img: np.ndarray) -> bool:
    """
    Детектирует титульный экран Fortnite.
    
    Характеристики:
    - Яркое изображение (не тёмный экран загрузки)
    - Много фиолетовых/пурпурных тонов (типичная цветовая схема Fortnite)
    - Возможно надпись FORTNITE в верхней части (белая)
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mean_brightness = np.mean(gray)
        
        # Титульный экран яркий (не тёмная загрузка)
        if mean_brightness < 60:
            return False
        
        # Проверяем наличие фиолетовых/пурпурных тонов (характерно для Fortnite)
        # Фиолетовый в HSV: H=125-155, высокая насыщенность
        purple_lower = np.array([120, 30, 50], dtype=np.uint8)
        purple_upper = np.array([165, 255, 255], dtype=np.uint8)
        mask_purple = cv2.inRange(hsv, purple_lower, purple_upper)
        purple_ratio = np.count_nonzero(mask_purple) / mask_purple.size
        
        # Если много фиолетового - скорее всего титульный экран Fortnite
        if purple_ratio > 0.05:
            return True
        
        # Также проверяем на яркие разноцветные области (персонажи, эффекты)
        # Высокая насыщенность и яркость
        colorful_lower = np.array([0, 80, 100], dtype=np.uint8)
        colorful_upper = np.array([180, 255, 255], dtype=np.uint8)
        mask_colorful = cv2.inRange(hsv, colorful_lower, colorful_upper)
        colorful_ratio = np.count_nonzero(mask_colorful) / mask_colorful.size
        
        # Проверяем верхнюю часть экрана на наличие белого текста (FORTNITE)
        top_roi = img[0:int(h*0.25), int(w*0.2):int(w*0.8)]
        top_gray = cv2.cvtColor(top_roi, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(top_gray > 200) / top_gray.size
        
        # Комбинация: много цветного + белый текст сверху
        if colorful_ratio > 0.2 and white_pixels > 0.02:
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
    
    # Информация для debug изображения
    debug_info = {
        "size": f"{w}x{h}",
        "brightness": f"{mean_brightness:.1f}",
        "contrast": f"{std_brightness:.1f}",
    }
    
    _log_debug(f"Анализ экрана: размер={w}x{h}, яркость={mean_brightness:.1f}, контраст={std_brightness:.1f}")
    
    # ВАЖНО: Если экран яркий (>100) - это точно НЕ игровая загрузка
    # Это может быть страница Xbox сайта
    if mean_brightness > 100:
        debug_info["note"] = "Яркий экран - сайт Xbox"
        _log_debug(f"  Яркий экран (brightness={mean_brightness:.1f}) - пропускаем CONNECTING/LOADING")
        
        # Проверяем страницу входа Microsoft
        white_lower = np.array([0, 0, 200], dtype=np.uint8)
        white_upper = np.array([180, 30, 255], dtype=np.uint8)
        mask_white = cv2.inRange(hsv, white_lower, white_upper)
        white_ratio = np.count_nonzero(mask_white) / mask_white.size
        debug_info["white"] = f"{white_ratio:.3f}"
        
        if white_ratio > 0.5:
            blue_lower = np.array([100, 50, 50], dtype=np.uint8)
            blue_upper = np.array([130, 255, 255], dtype=np.uint8)
            mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
            blue_ratio = np.count_nonzero(mask_blue) / mask_blue.size
            debug_info["blue"] = f"{blue_ratio:.4f}"
            if blue_ratio > 0.01:
                _log_info("Состояние: LOGIN_PAGE (страница входа Microsoft)")
                _save_debug_image(img, "LOGIN_PAGE", debug_info)
                return ScreenState.LOGIN_PAGE
        
        # Яркий экран но не login - это UNKNOWN (страница сайта Xbox)
        _log_debug("  Состояние: UNKNOWN (яркий экран, вероятно сайт Xbox)")
        _save_debug_image(img, "UNKNOWN_BRIGHT", debug_info)
        return ScreenState.UNKNOWN
    
    # 1. CONNECTING оверлей (только на ОЧЕНЬ тёмном фоне < 60)
    if mean_brightness < 60:
        is_connecting = _detect_connecting_overlay(img)
        debug_info["connecting_check"] = str(is_connecting)
        _log_debug(f"  Проверка CONNECTING (яркость<60): {is_connecting}")
        if is_connecting:
            _log_info("Состояние: CONNECTING (оверлей подключения)")
            _save_debug_image(img, "CONNECTING", debug_info)
            return ScreenState.CONNECTING
    
    # 2. Xbox Queue (очередь Game Pass) - тёмный экран с рекламой и прогресс-баром
    if mean_brightness < 80:
        is_queue = _detect_xbox_queue(img)
        debug_info["queue_check"] = str(is_queue)
        _log_debug(f"  Проверка XBOX_QUEUE (яркость<80): {is_queue}")
        if is_queue:
            _log_info("Состояние: XBOX_QUEUE (очередь)")
            _save_debug_image(img, "XBOX_QUEUE", debug_info)
            return ScreenState.XBOX_QUEUE
    
    # 3. Xbox Logo Screen (большой логотип XBOX по центру) - ПРОВЕРЯЕМ ДО plane screen!
    if mean_brightness < 45:
        is_xbox_logo = _detect_xbox_logo_screen(img)
        debug_info["xbox_logo_check"] = str(is_xbox_logo)
        _log_debug(f"  Проверка XBOX_LOGO (яркость<45): {is_xbox_logo}")
        if is_xbox_logo:
            _log_info("Состояние: XBOX_LOADING (экран с логотипом XBOX)")
            _save_debug_image(img, "XBOX_LOADING", debug_info)
            return ScreenState.XBOX_LOADING
    
    # 4. Plane screen (зелёный самолёт)
    is_plane = _detect_plane_screen(img)
    debug_info["plane_check"] = str(is_plane)
    _log_debug(f"  Проверка PLANE_SCREEN: {is_plane}")
    if is_plane:
        _log_info("Состояние: PLANE_SCREEN (самолётик Xbox)")
        _save_debug_image(img, "PLANE_SCREEN", debug_info)
        return ScreenState.PLANE_SCREEN
    
    # 5. Xbox loading (зелёный или белый логотип Xbox)
    if mean_brightness < 40:
        # Проверка зелёного логотипа Xbox
        xbox_lower = np.array([35, 100, 50], dtype=np.uint8)
        xbox_upper = np.array([85, 255, 255], dtype=np.uint8)
        mask_xbox = cv2.inRange(hsv, xbox_lower, xbox_upper)
        xbox_green_ratio = np.count_nonzero(mask_xbox) / mask_xbox.size
        debug_info["xbox_green"] = f"{xbox_green_ratio:.4f}"
        
        # Проверка белого логотипа Xbox в правом нижнем углу
        # Область: правая нижняя четверть экрана
        bottom_right = img[int(h*0.5):, int(w*0.6):]
        bottom_right_gray = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2GRAY)
        # Белые пиксели (яркость > 200)
        white_mask = bottom_right_gray > 200
        white_logo_ratio = np.count_nonzero(white_mask) / white_mask.size
        debug_info["xbox_white_logo"] = f"{white_logo_ratio:.4f}"
        
        _log_debug(f"  Проверка XBOX_LOADING (яркость<40): зелёный={xbox_green_ratio:.4f}, белый_лого={white_logo_ratio:.4f}")
        
        # Зелёный логотип ИЛИ белый логотип в углу
        if xbox_green_ratio > 0.01 or (white_logo_ratio > 0.02 and white_logo_ratio < 0.3):
            _log_info("Состояние: XBOX_LOADING (загрузка Xbox)")
            _save_debug_image(img, "XBOX_LOADING", debug_info)
            return ScreenState.XBOX_LOADING
        
        # Очень тёмный экран (почти чёрный) - тоже загрузка
        if mean_brightness < 10 and std_brightness < 20:
            _log_info("Состояние: LOADING (очень тёмный экран)")
            _save_debug_image(img, "LOADING", debug_info)
            return ScreenState.LOADING
        
        if std_brightness < 30:
            _log_info("Состояние: LOADING (тёмный экран загрузки)")
            _save_debug_image(img, "LOADING", debug_info)
            return ScreenState.LOADING
    
    # 6. Титульный экран Fortnite (яркий, фиолетовый, с персонажами)
    is_title = _detect_title_screen(img)
    debug_info["title_check"] = str(is_title)
    _log_debug(f"  Проверка TITLE_SCREEN: {is_title}")
    if is_title:
        _log_info("Состояние: TITLE_SCREEN (титульный экран)")
        _save_debug_image(img, "TITLE_SCREEN", debug_info)
        return ScreenState.TITLE_SCREEN
    
    # 6. Страница входа Microsoft (белый фон с синими элементами)
    # НЕ путать с Xbox Cloud Gaming страницей которая тоже яркая!
    white_lower = np.array([0, 0, 200], dtype=np.uint8)
    white_upper = np.array([180, 30, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, white_lower, white_upper)
    white_ratio = np.count_nonzero(mask_white) / mask_white.size
    
    debug_info["white"] = f"{white_ratio:.3f}"
    _log_debug(f"  Проверка LOGIN_PAGE: белый={white_ratio:.3f}")
    if white_ratio > 0.5:
        blue_lower = np.array([100, 50, 50], dtype=np.uint8)
        blue_upper = np.array([130, 255, 255], dtype=np.uint8)
        mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
        blue_ratio = np.count_nonzero(mask_blue) / mask_blue.size
        debug_info["blue"] = f"{blue_ratio:.4f}"
        _log_debug(f"    синий={blue_ratio:.4f}")
        # Только если много белого И есть синий - это Microsoft login
        if blue_ratio > 0.01:
            _log_info("Состояние: LOGIN_PAGE (страница входа Microsoft)")
            _save_debug_image(img, "LOGIN_PAGE", debug_info)
            return ScreenState.LOGIN_PAGE
    
    # 7. Игровой экран (Fortnite лобби или в игре)
    if std_brightness > 40:
        top_ui = img[0:int(h*0.12), :]
        bottom_ui = img[int(h*0.85):, :]
        
        top_edges = cv2.Canny(cv2.cvtColor(top_ui, cv2.COLOR_BGR2GRAY), 50, 150)
        bottom_edges = cv2.Canny(cv2.cvtColor(bottom_ui, cv2.COLOR_BGR2GRAY), 50, 150)
        
        top_edge_ratio = np.count_nonzero(top_edges) / top_edges.size
        bottom_edge_ratio = np.count_nonzero(bottom_edges) / bottom_edges.size
        
        debug_info["top_edges"] = f"{top_edge_ratio:.4f}"
        debug_info["bottom_edges"] = f"{bottom_edge_ratio:.4f}"
        _log_debug(f"  Проверка IN_GAME/LOBBY (контраст>40): top_edges={top_edge_ratio:.4f}, bottom_edges={bottom_edge_ratio:.4f}")
        
        if top_edge_ratio > 0.03 and bottom_edge_ratio > 0.03:
            _log_info("Состояние: IN_GAME (в игре)")
            _save_debug_image(img, "IN_GAME", debug_info)
            return ScreenState.IN_GAME
        
        if top_edge_ratio > 0.01 or bottom_edge_ratio > 0.02:
            _log_info("Состояние: LOBBY (лобби)")
            _save_debug_image(img, "LOBBY", debug_info)
            return ScreenState.LOBBY
    
    # 8. Меню/диалог
    center_roi = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    center_bright = np.mean(cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY))
    edge_bright = (np.mean(gray[:int(h*0.2), :]) + np.mean(gray[int(h*0.8):, :])) / 2
    
    debug_info["center_bright"] = f"{center_bright:.1f}"
    debug_info["edge_bright"] = f"{edge_bright:.1f}"
    _log_debug(f"  Проверка MENU: center_bright={center_bright:.1f}, edge_bright={edge_bright:.1f}")
    if center_bright > edge_bright * 1.5 and center_bright > 80:
        _log_info("Состояние: MENU (меню/диалог)")
        _save_debug_image(img, "MENU", debug_info)
        return ScreenState.MENU
    
    # 9. Ошибка
    red_lower1 = np.array([0, 100, 100], dtype=np.uint8)
    red_upper1 = np.array([10, 255, 255], dtype=np.uint8)
    red_lower2 = np.array([170, 100, 100], dtype=np.uint8)
    red_upper2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, red_lower1, red_upper1),
        cv2.inRange(hsv, red_lower2, red_upper2)
    )
    red_ratio = np.count_nonzero(mask_red) / mask_red.size
    
    debug_info["red"] = f"{red_ratio:.4f}"
    _log_debug(f"  Проверка ERROR: красный={red_ratio:.4f}")
    if red_ratio > 0.01:
        _log_info("Состояние: ERROR (ошибка)")
        _save_debug_image(img, "ERROR", debug_info)
        return ScreenState.ERROR
    
    _log_debug("  Состояние: UNKNOWN")
    _save_debug_image(img, "UNKNOWN", debug_info)
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
