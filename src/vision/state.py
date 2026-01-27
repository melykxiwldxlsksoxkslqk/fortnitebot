"""
Определение состояния экрана.
"""

import os
import time
import cv2
import numpy as np
from typing import Optional, Union, List, Dict, Callable, Tuple
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


# ============================================================================
# TEMPLATE MATCHING СИСТЕМА
# ============================================================================

# Кэш загруженных шаблонов
_TEMPLATE_CACHE: Dict[str, np.ndarray] = {}

# Конфигурация UI элементов определяется после класса ScreenState (см. ниже)
UI_ELEMENTS_CONFIG = []  # Будет заполнено после определения ScreenState


def _load_template(name: str) -> Optional[np.ndarray]:
    """Загружает шаблон из кэша или с диска."""
    if name in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[name]
    
    template_path = os.path.join(ROOT_DIR, 'assets', name)
    if not os.path.exists(template_path):
        return None
    
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if template is not None:
        _TEMPLATE_CACHE[name] = template
    
    return template


def _find_template_multi_scale(
    img_gray: np.ndarray, 
    template: np.ndarray, 
    roi: Tuple[float, float, float, float] = None,
    threshold: float = 0.8,
    scales: List[float] = None
) -> Optional[Tuple[int, int, int, int, float]]:
    """
    Ищет шаблон с multi-scale matching.
    
    ВАЖНО: Используем ограниченный диапазон масштабов для скорости и точности.
    
    Returns:
        (x, y, w, h, confidence) или None
    """
    if scales is None:
        # Ограниченный диапазон для более точного matching
        # UI элементы обычно не сильно меняют размер
        scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    
    h, w = img_gray.shape[:2]
    
    # Применяем ROI если указан
    if roi:
        x0, y0 = int(w * roi[0]), int(h * roi[1])
        x1, y1 = int(w * roi[2]), int(h * roi[3])
        # Проверяем валидность ROI
        if x1 <= x0 or y1 <= y0:
            return None
        search_img = img_gray[y0:y1, x0:x1]
        offset_x, offset_y = x0, y0
    else:
        search_img = img_gray
        offset_x, offset_y = 0, 0
    
    # Минимальный размер области поиска
    if search_img.shape[0] < 20 or search_img.shape[1] < 20:
        return None
    
    best_val = 0
    best_result = None
    
    for scale in scales:
        resized = cv2.resize(template, None, fx=scale, fy=scale)
        th, tw = resized.shape[:2]
        
        if th > search_img.shape[0] or tw > search_img.shape[1]:
            continue
        
        # Минимальный размер шаблона
        if th < 10 or tw < 10:
            continue
        
        result = cv2.matchTemplate(search_img, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_val:
            best_val = max_val
            best_result = (
                max_loc[0] + offset_x,
                max_loc[1] + offset_y,
                tw, th,
                max_val
            )
    
    if best_val >= threshold:
        return best_result
    return None


def _detect_ui_elements(img: np.ndarray) -> List[Dict]:
    """
    Ищет все UI элементы на изображении через template matching.
    
    Включает валидацию контекста для избежания false positives.
    
    Returns:
        Список найденных элементов с координатами и confidence
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # ================================================================
    # КОНТЕКСТНАЯ ВАЛИДАЦИЯ - исключаем false positives
    # ================================================================
    mean_brightness = np.mean(gray)
    
    # Если экран очень тёмный (загрузка) - не ищем UI элементы
    # Исключение: специфичные элементы для loading экранов
    is_dark_screen = mean_brightness < 35
    
    # Проверяем есть ли характерные признаки Discover/Lobby экрана
    # (яркая верхняя панель меню)
    top_bar = gray[0:int(h*0.12), :]
    top_bar_brightness = np.mean(top_bar)
    has_menu_bar = top_bar_brightness > 80  # Меню обычно светлое
    
    found_elements = []
    
    for template_name, state, roi, threshold, priority in UI_ELEMENTS_CONFIG:
        template = _load_template(template_name)
        if template is None:
            continue
        
        # ============================================================
        # КОНТЕКСТНЫЕ ФИЛЬТРЫ
        # ============================================================
        
        # На тёмном экране загрузки НЕ ищем UI элементы
        if is_dark_screen:
            # Разрешаем только специфичные loading элементы если они будут
            _log_debug(f"Пропуск {template_name}: тёмный экран загрузки")
            continue
        
        # search_input_field требует наличия меню бара (не на загрузке)
        if 'search_input' in template_name.lower() and not has_menu_bar:
            _log_debug(f"Пропуск {template_name}: нет меню бара")
            continue
        
        # select_button должен быть ТОЛЬКО внизу экрана
        if 'select_button' in template_name.lower():
            # Проверяем что область снизу достаточно яркая (не игровой контент)
            bottom_area = gray[int(h*0.85):, :]
            if np.mean(bottom_area) < 50:
                _log_debug(f"Пропуск {template_name}: тёмная нижняя область")
                continue
        
        result = _find_template_multi_scale(gray, template, roi, threshold)
        
        if result:
            x, y, elem_w, elem_h, confidence = result
            
            # ============================================================
            # POST-DETECTION ВАЛИДАЦИЯ
            # ============================================================
            
            # Проверяем что элемент не на карточке игры (для search_input_field)
            if 'search_input' in template_name.lower():
                # search_input_field должен быть в верхней половине экрана
                if y > h * 0.55:
                    _log_debug(f"Отклонён {template_name}: слишком низко на экране")
                    continue
                # И должен быть достаточно широким (не карточка)
                if elem_w < w * 0.15:
                    _log_debug(f"Отклонён {template_name}: слишком узкий")
                    continue
            
            # select_button должен быть в нижней части
            if 'select_button' in template_name.lower():
                if y < h * 0.70:
                    _log_debug(f"Отклонён {template_name}: слишком высоко")
                    continue
            
            found_elements.append({
                'name': template_name.replace('.png', ''),
                'state': state,
                'x': x, 'y': y, 'w': elem_w, 'h': elem_h,
                'confidence': confidence,
                'priority': priority
            })
            _log_debug(f"Найден элемент: {template_name} at ({x},{y}) conf={confidence:.3f}")
    
    # Сортируем по приоритету (высший первый)
    found_elements.sort(key=lambda e: (-e['priority'], -e['confidence']))
    
    return found_elements


def _save_debug_image(img: np.ndarray, state: str, info: dict = None, elements: List[Dict] = None):
    """Сохраняет debug скриншот с информацией о детекции и найденными элементами."""
    if not _VISION_DEBUG:
        return
    
    try:
        # Создаём копию для рисования
        debug_img = img.copy()
        h, w = debug_img.shape[:2]
        
        # ============================================================
        # РИСУЕМ НАЙДЕННЫЕ UI ЭЛЕМЕНТЫ
        # ============================================================
        if elements:
            for elem in elements:
                x, y, ew, eh = elem['x'], elem['y'], elem['w'], elem['h']
                conf = elem['confidence']
                name = elem['name']
                
                # Цвет рамки в зависимости от confidence
                if conf > 0.8:
                    color = (0, 255, 0)  # Зелёный - отлично
                elif conf > 0.65:
                    color = (0, 255, 255)  # Жёлтый - хорошо
                else:
                    color = (0, 165, 255)  # Оранжевый - средне
                
                # Рамка элемента
                cv2.rectangle(debug_img, (x, y), (x + ew, y + eh), color, 2)
                
                # Подпись с названием и confidence
                label = f"{name} ({conf:.0%})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                
                # Фон для подписи
                cv2.rectangle(debug_img, (x, y - 20), (x + label_size[0] + 4, y), color, -1)
                cv2.putText(debug_img, label, (x + 2, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # ============================================================
        # ИНФОРМАЦИОННАЯ ПАНЕЛЬ
        # ============================================================
        # Добавляем полупрозрачный фон для текста
        overlay = debug_img.copy()
        panel_height = 180 + (len(elements) * 22 if elements else 0)
        panel_height = min(panel_height, h - 20)
        cv2.rectangle(overlay, (0, 0), (450, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, debug_img, 0.3, 0, debug_img)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Состояние (большим шрифтом)
        state_colors = {
            "LOBBY": (0, 255, 0),
            "DISCOVER": (255, 255, 0),
            "SEARCH_INPUT": (0, 255, 255),
            "SEARCH_RESULTS": (255, 200, 0),
            "ISLAND_PREVIEW": (255, 150, 0),
            "LOADING": (150, 150, 150),
            "XBOX_LOADING": (150, 150, 150),
            "XBOX_QUEUE": (150, 150, 150),
        }
        state_color = state_colors.get(state, (255, 255, 255))
        cv2.putText(debug_img, f"State: {state}", (10, 30), font, 0.9, state_color, 2)
        
        # Время
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cv2.putText(debug_img, f"Time: {timestamp}", (10, 55), font, 0.6, (200, 200, 200), 1)
        
        # Размер изображения
        cv2.putText(debug_img, f"Size: {w}x{h}", (10, 75), font, 0.5, (150, 150, 150), 1)
        
        # ============================================================
        # СПИСОК НАЙДЕННЫХ ЭЛЕМЕНТОВ
        # ============================================================
        y_offset = 100
        if elements:
            cv2.putText(debug_img, f"Found {len(elements)} UI elements:", (10, y_offset), 
                       font, 0.55, (100, 255, 100), 1)
            y_offset += 22
            
            for elem in elements[:8]:  # Показываем максимум 8
                conf = elem['confidence']
                color = (0, 255, 0) if conf > 0.7 else (0, 255, 255) if conf > 0.6 else (0, 165, 255)
                text = f"  • {elem['name']}: {conf:.0%} at ({elem['x']},{elem['y']})"
                cv2.putText(debug_img, text, (10, y_offset), font, 0.45, color, 1)
                y_offset += 20
        else:
            cv2.putText(debug_img, "No UI elements found", (10, y_offset), 
                       font, 0.55, (100, 100, 255), 1)
            y_offset += 22
        
        # Дополнительная информация (метрики)
        if info:
            y_offset += 5
            for key, value in list(info.items())[:5]:
                if "check" in key.lower():
                    color = (0, 255, 0) if value == "True" else (100, 100, 255)
                else:
                    color = (180, 180, 180)
                cv2.putText(debug_img, f"{key}: {value}", (10, y_offset), font, 0.45, color, 1)
                y_offset += 18
        
        # ============================================================
        # ROI ОБЛАСТИ (для debug)
        # ============================================================
        # Верхняя панель (меню)
        cv2.rectangle(debug_img, (0, 0), (w, int(h*0.12)), (50, 50, 150), 1)
        # Нижняя панель
        cv2.rectangle(debug_img, (0, int(h*0.85)), (w, h), (50, 50, 150), 1)
        
        # Сохраняем с уникальным именем
        timestamp = datetime.now().strftime('%H%M%S_%f')[:13]
        filename = f"{timestamp}_{state}.jpg"
        filepath = os.path.join(_DEBUG_DIR, filename)
        cv2.imwrite(filepath, debug_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
    except Exception as e:
        _vision_logger.debug(f"Ошибка сохранения debug изображения: {e}")


def _log_debug(msg: str):
    """Логирует если включен debug режим."""
    if _VISION_DEBUG:
        _vision_logger.debug(f"[VISION] {msg}")


def _log_info(msg: str):
    """Всегда логирует информацию."""
    _vision_logger.info(f"[VISION] {msg}")


def _log_detection_result(state: str, reason: str, metrics: dict = None):
    """
    Логирует результат детекции состояния экрана в понятном формате.
    
    Args:
        state: Название состояния
        reason: Краткое объяснение почему определено это состояние
        metrics: Дополнительные метрики (яркость, цвета и т.д.)
    """
    # Формируем понятное сообщение
    metrics_str = ""
    if metrics:
        parts = []
        for key, value in metrics.items():
            if isinstance(value, float):
                parts.append(f"{key}={value:.1f}")
            else:
                parts.append(f"{key}={value}")
        metrics_str = f" | {', '.join(parts)}"
    
    _vision_logger.info(f"[VISION] Состояние: {state} ({reason}){metrics_str}")


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
    LOBBY = auto()             # Лобби Fortnite (персонаж в центре)
    DISCOVER = auto()          # Экран Discover с Search Discover баром вверху
    SEARCH_INPUT = auto()      # Диалог ввода кода острова
    SEARCH_RESULTS = auto()    # Результаты поиска (карты островов)
    ISLAND_PREVIEW = auto()    # Превью острова (SELECT/PLAY кнопки)
    IN_GAME = auto()           # В игре
    MENU = auto()              # Меню/настройки
    ERROR = auto()             # Ошибка/диалог
    POPUP = auto()             # Popup уведомление (новости, ивенты - нужно закрыть)


# ============================================================================
# КОНФИГУРАЦИЯ UI ЭЛЕМЕНТОВ ДЛЯ TEMPLATE MATCHING
# ============================================================================
# (template_name, associated_state, roi, threshold, priority)
# roi = (x0, y0, x1, y1) в процентах от размера экрана
# priority - чем выше, тем раньше проверяется
# 
# ВАЖНО: threshold должен быть ВЫСОКИМ (0.82+) чтобы избежать false positives!
# ROI должен быть УЗКИМ и точно соответствовать месту элемента на экране!
UI_ELEMENTS_CONFIG.clear()
UI_ELEMENTS_CONFIG.extend([
    # =========================================================================
    # SEARCH INPUT DIALOG - диалог ввода кода острова (появляется по центру)
    # Это popup окно, которое появляется поверх Discover экрана
    # =========================================================================
    # Кнопка "Odeslat" (отправить) - внутри диалога, правая часть
    ('odeslat_button.png', ScreenState.SEARCH_INPUT, (0.40, 0.35, 0.70, 0.55), 0.85, 10),
    # Весь диалог поиска - по центру экрана
    ('search_dialog.png', ScreenState.SEARCH_INPUT, (0.25, 0.25, 0.75, 0.60), 0.83, 10),
    # Поле ввода внутри диалога - узкая полоса по центру
    ('search_input_field.png', ScreenState.SEARCH_INPUT, (0.30, 0.35, 0.70, 0.50), 0.85, 9),
    
    # =========================================================================
    # DISCOVER SCREEN - главный экран с поиском
    # "Search Discover" бар находится в ЛЕВОМ ВЕРХНЕМ углу
    # =========================================================================
    ('search_discover_bar.png', ScreenState.DISCOVER, (0.02, 0.06, 0.35, 0.16), 0.83, 8),
    
    # =========================================================================
    # LOBBY - главное меню с кнопкой PLAY
    # Кнопка PLAY жёлтая, находится в ЛЕВОМ ВЕРХНЕМ углу
    # =========================================================================
    ('play_button_yellow.png', ScreenState.LOBBY, (0.02, 0.03, 0.18, 0.12), 0.85, 7),
    
    # =========================================================================
    # ISLAND PREVIEW - предпросмотр острова перед запуском
    # Кнопки SELECT/PLAY находятся ВНИЗУ по центру
    # =========================================================================
    # Кнопка Select - нижняя часть экрана, по центру
    ('select_button.png', ScreenState.ISLAND_PREVIEW, (0.35, 0.80, 0.65, 0.98), 0.85, 5),
    # Кнопка Play Island - нижняя часть экрана, по центру  
    ('play_button_island.png', ScreenState.ISLAND_PREVIEW, (0.35, 0.80, 0.65, 0.98), 0.85, 5),
    
    # Like button - на карточке острова, ВЕРХНЯЯ ЛЕВАЯ область карточки
    ('like_button_empty.png', ScreenState.ISLAND_PREVIEW, (0.02, 0.15, 0.25, 0.45), 0.87, 4),
    ('like_button_filled.png', ScreenState.ISLAND_PREVIEW, (0.02, 0.15, 0.25, 0.45), 0.87, 4),
    
    # =========================================================================
    # SEARCH RESULTS - результаты поиска острова
    # Карточки островов появляются в центре экрана
    # =========================================================================
    ('island_card.png', ScreenState.SEARCH_RESULTS, (0.15, 0.20, 0.55, 0.75), 0.83, 3),
])


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
    
    ВАЖНО: НЕ путать с Search Discover диалогом!
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Тёмный фон (яркость < 40)
        mean_brightness = np.mean(gray)
        if mean_brightness > 45:
            return False
        
        # ИСКЛЮЧЕНИЕ: Проверяем нет ли яркого диалога в центре (Search Discover popup)
        # Диалог обычно серый/белый прямоугольник с высоким контрастом к тёмному фону
        dialog_area = gray[int(h*0.20):int(h*0.55), int(w*0.30):int(w*0.70)]
        dialog_mean = np.mean(dialog_area)
        dialog_bright_pixels = np.count_nonzero(dialog_area > 100) / dialog_area.size
        
        # Если в центре много ярких пикселей (>40%) - это вероятно диалог, НЕ Xbox logo
        if dialog_bright_pixels > 0.40:
            _log_debug(f"_detect_xbox_logo: исключение - яркий диалог в центре ({dialog_bright_pixels:.2%})")
            return False
        
        # ИСКЛЮЧЕНИЕ 2: Проверяем нет ли зелёной кнопки (ODESLAT) - признак Search диалога
        dialog_hsv = hsv[int(h*0.35):int(h*0.55), int(w*0.35):int(w*0.65)]
        green_lower = np.array([35, 80, 80], dtype=np.uint8)
        green_upper = np.array([85, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(dialog_hsv, green_lower, green_upper)
        green_ratio = np.count_nonzero(green_mask) / green_mask.size
        
        if green_ratio > 0.02:  # Есть зелёная кнопка
            _log_debug(f"_detect_xbox_logo: исключение - зелёная кнопка ({green_ratio:.4f})")
            return False
        
        # ИСКЛЮЧЕНИЕ 3: Проверяем нет ли белых вкладок вверху (признак Discover экрана)
        top_tabs = gray[int(h*0.06):int(h*0.12), int(w*0.45):int(w*0.95)]
        top_tabs_white = np.count_nonzero(top_tabs > 180) / top_tabs.size
        if top_tabs_white > 0.03:  # Много белого в области вкладок
            _log_debug(f"_detect_xbox_logo: исключение - белые вкладки вверху ({top_tabs_white:.4f})")
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
                _log_debug(f"_detect_xbox_logo: ОБНАРУЖЕНО (white_ratio={white_ratio:.4f}, std={std_center:.1f})")
                return True
        
        return False
    except Exception as e:
        _log_debug(f"_detect_xbox_logo error: {e}")
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
    Детектирует титульный экран Fortnite ("Press any key to continue" / "LOGGING IN...").
    
    КЛЮЧЕВЫЕ ОТЛИЧИЯ от LOBBY:
    - Большой логотип "FORTNITE" в верхней части (белые буквы)
    - Много фиолетового/пурпурного цвета (сезонная тема)
    - НЕТ верхнего меню (PLAY SHOP LOCKER...) 
    - НЕТ карусели игр внизу
    - Текст "LOGGING IN..." или "Press any key" внизу
    - Информация об аккаунте внизу слева ("Signed in as: xxx")
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mean_brightness = np.mean(gray)
        
        # Title screen яркий (50-120)
        if mean_brightness < 40 or mean_brightness > 140:
            _log_debug(f"_detect_title_screen: brightness={mean_brightness:.1f} вне диапазона - НЕТ")
            return False
        
        # ============================================================
        # ИСКЛЮЧЕНИЕ 1: Проверяем НЕТ ЛИ верхнего меню PLAY/SHOP/LOCKER
        # Это КЛЮЧЕВОЕ отличие - у Title Screen НЕТ меню!
        # ============================================================
        
        # Верхнее меню в лобби: белый текст на тёмном/прозрачном фоне
        top_menu_roi = gray[int(h*0.03):int(h*0.09), int(w*0.12):int(w*0.70)]
        top_menu_white = np.count_nonzero(top_menu_roi > 200) / top_menu_roi.size
        
        # Edges - буквы меню создают чёткие контуры
        top_edges = cv2.Canny(top_menu_roi, 100, 200)
        top_edge_ratio = np.count_nonzero(top_edges) / top_edges.size
        
        _log_debug(f"_detect_title_screen: menu_white={top_menu_white:.4f}, edges={top_edge_ratio:.4f}")
        
        # Если есть меню (белый текст + контуры) - это LOBBY, не title screen
        if top_menu_white > 0.025 and top_edge_ratio > 0.02:
            _log_debug("_detect_title_screen: НЕТ - есть верхнее меню (это LOBBY)")
            return False
        
        # ============================================================
        # ИСКЛЮЧЕНИЕ 2: НЕТ карусели игр внизу  
        # ============================================================
        bottom_roi = gray[int(h*0.75):int(h*0.95), int(w*0.05):int(w*0.95)]
        bottom_std = np.std(bottom_roi)
        bottom_bright = np.count_nonzero(bottom_roi > 120) / bottom_roi.size
        
        _log_debug(f"_detect_title_screen: bottom_std={bottom_std:.1f}, bottom_bright={bottom_bright:.4f}")
        
        # Карусель создаёт высокий контраст (std > 50) и яркие карточки (bright > 20%)
        if bottom_std > 50 and bottom_bright > 0.20:
            _log_debug("_detect_title_screen: НЕТ - есть карусель (это LOBBY)")
            return False
        
        # ============================================================
        # ПРИЗНАК 1: Большой логотип FORTNITE в верхней части
        # ============================================================
        logo_roi = gray[int(h*0.08):int(h*0.35), int(w*0.25):int(w*0.75)]
        logo_white = np.count_nonzero(logo_roi > 230) / logo_roi.size
        
        _log_debug(f"_detect_title_screen: logo_white={logo_white:.4f}")
        
        # ============================================================
        # ПРИЗНАК 2: Много фиолетового цвета (сезонная тема)
        # ============================================================
        purple_lower = np.array([120, 30, 40], dtype=np.uint8)
        purple_upper = np.array([170, 255, 255], dtype=np.uint8)
        mask_purple = cv2.inRange(hsv, purple_lower, purple_upper)
        purple_ratio = np.count_nonzero(mask_purple) / mask_purple.size
        
        _log_debug(f"_detect_title_screen: purple_ratio={purple_ratio:.4f}")
        
        # ============================================================
        # ПРИЗНАК 3: Текст внизу экрана (LOGGING IN... / Press any key)
        # ============================================================
        bottom_text_roi = gray[int(h*0.85):int(h*0.95), int(w*0.30):int(w*0.70)]
        bottom_text_white = np.count_nonzero(bottom_text_roi > 200) / bottom_text_roi.size
        
        _log_debug(f"_detect_title_screen: bottom_text_white={bottom_text_white:.4f}")
        
        # ============================================================
        # ФИНАЛЬНОЕ РЕШЕНИЕ
        # ============================================================
        
        # Главный критерий: много фиолетового И большой логотип
        if purple_ratio > 0.08 and logo_white > 0.03:
            _log_debug("_detect_title_screen: ДА - фиолетовый фон + логотип")
            return True
        
        # Альтернатива: много фиолетового + текст внизу (LOGGING IN...)
        if purple_ratio > 0.10 and bottom_text_white > 0.005:
            _log_debug("_detect_title_screen: ДА - фиолетовый + текст внизу")
            return True
        
        # Большой белый логотип + нет меню + нет карусели
        if logo_white > 0.06 and top_menu_white < 0.01 and bottom_std < 45:
            _log_debug("_detect_title_screen: ДА - большой логотип, нет меню")
            return True
        
        _log_debug("_detect_title_screen: НЕТ")
        return False
    except Exception as e:
        _log_debug(f"_detect_title_screen error: {e}")
        return False


def _detect_popup_notification(img: np.ndarray) -> bool:
    """
    Детектирует popup уведомления (новости, ивенты, промо).
    
    ПРИМЕРЫ:
    - "KENNY'S POWER HOUR THIS SATURDAY..."
    - Новости об обновлениях
    - Промо-акции
    
    КЛЮЧЕВЫЕ ПРИЗНАКИ:
    - Большая карточка слева (яркая картинка)
    - Текстовый блок справа с заголовком и описанием
    - Кнопка "PLAY NOW" или "Back" внизу
    - НЕТ верхнего меню PLAY/SHOP/LOCKER
    - Кнопка "Back" в правом нижнем углу
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mean_brightness = np.mean(gray)
        
        # Popup обычно средней яркости (40-90)
        if mean_brightness < 30 or mean_brightness > 100:
            return False
        
        # ============================================================
        # ПРИЗНАК 1: НЕТ верхнего меню PLAY/SHOP/LOCKER
        # У popup нет стандартного меню игры
        # ============================================================
        top_menu_roi = gray[int(h*0.02):int(h*0.08), int(w*0.12):int(w*0.70)]
        menu_white = np.count_nonzero(top_menu_roi > 200) / top_menu_roi.size
        
        # Если есть меню - это не popup
        if menu_white > 0.02:
            return False
        
        # ============================================================
        # ПРИЗНАК 2: Большая картинка СПРАВА (яркая, контрастная)
        # ============================================================
        right_roi = gray[int(h*0.10):int(h*0.75), int(w*0.55):int(w*0.98)]
        right_mean = np.mean(right_roi)
        right_std = np.std(right_roi)
        right_bright = np.count_nonzero(right_roi > 100) / right_roi.size
        
        # ============================================================
        # ПРИЗНАК 3: Текстовый блок СЛЕВА (белый текст на тёмном)
        # ============================================================
        left_roi = gray[int(h*0.15):int(h*0.65), int(w*0.05):int(w*0.50)]
        left_mean = np.mean(left_roi)
        left_white = np.count_nonzero(left_roi > 200) / left_roi.size
        
        _log_debug(f"_detect_popup: brightness={mean_brightness:.1f}, menu={menu_white:.4f}")
        _log_debug(f"_detect_popup: right_mean={right_mean:.1f}, right_std={right_std:.1f}, right_bright={right_bright:.4f}")
        _log_debug(f"_detect_popup: left_mean={left_mean:.1f}, left_white={left_white:.4f}")
        
        # ============================================================
        # ПРИЗНАК 4: Кнопка "Back" в правом нижнем углу
        # ============================================================
        back_roi = gray[int(h*0.88):int(h*0.98), int(w*0.85):int(w*0.98)]
        back_white = np.count_nonzero(back_roi > 180) / back_roi.size
        
        _log_debug(f"_detect_popup: back_button_white={back_white:.4f}")
        
        # ============================================================
        # ФИНАЛЬНОЕ РЕШЕНИЕ
        # ============================================================
        
        # Popup: картинка справа + текст слева + кнопка Back
        has_right_image = right_std > 40 and right_bright > 0.30
        has_left_text = left_white > 0.02 and left_mean < 80
        has_back_button = back_white > 0.02
        
        if has_right_image and has_left_text:
            _log_debug("_detect_popup: ДА - картинка справа + текст слева")
            return True
        
        if has_left_text and has_back_button and menu_white < 0.01:
            _log_debug("_detect_popup: ДА - текст слева + Back кнопка")
            return True
        
        _log_debug("_detect_popup: НЕТ")
        return False
    except Exception as e:
        _log_debug(f"_detect_popup error: {e}")
        return False


def _detect_fortnite_lobby(img: np.ndarray) -> bool:
    """
    Детектирует ОСНОВНОЕ лобби Fortnite с персонажем в центре.
    
    КЛЮЧЕВЫЕ ОТЛИЧИЯ от DISCOVER:
    - ЯРКИЙ экран (brightness > 55) - у Discover тёмный фон (< 55)
    - Персонаж в центре экрана (крупная фигура)
    - Яркий/цветной фон (небо, строения и т.д.)
    
    КЛЮЧЕВЫЕ ОТЛИЧИЯ от TITLE_SCREEN:
    - Есть верхнее меню PLAY/SHOP/LOCKER
    - Есть карусель игр внизу
    - Меньше фиолетового цвета
    """
    try:
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        
        # ============================================================
        # КЛЮЧЕВОЕ: LOBBY ЯРКОЕ (> 55)!
        # DISCOVER тёмный (< 55), это главное отличие
        # ============================================================
        if mean_brightness < 55:
            _log_debug(f"_detect_fortnite_lobby: тёмный экран ({mean_brightness:.1f}) - это DISCOVER, не LOBBY")
            return False
        
        # ============================================================
        # 1. Проверяем верхнее меню (PLAY SHOP LOCKER...)
        # ============================================================
        top_menu_roi = gray[int(h*0.03):int(h*0.10), int(w*0.12):int(w*0.70)]
        menu_white = np.count_nonzero(top_menu_roi > 200) / top_menu_roi.size
        menu_edges = cv2.Canny(top_menu_roi, 100, 200)
        menu_edge_ratio = np.count_nonzero(menu_edges) / menu_edges.size
        
        has_top_menu = menu_white > 0.025 and menu_edge_ratio > 0.015
        
        _log_debug(f"_detect_fortnite_lobby: brightness={mean_brightness:.1f}, menu_white={menu_white:.4f}, edges={menu_edge_ratio:.4f}")
        
        if not has_top_menu:
            _log_debug("_detect_fortnite_lobby: нет верхнего меню - НЕ LOBBY")
            return False
        
        # ============================================================
        # 2. Проверяем персонажа в центре (ЯРКАЯ центральная область)
        # ============================================================
        center_roi = gray[int(h*0.20):int(h*0.70), int(w*0.30):int(w*0.70)]
        center_std = np.std(center_roi)
        center_mean = np.mean(center_roi)
        
        # В LOBBY центр яркий (> 70) из-за неба/фона и персонажа
        # В DISCOVER центр тёмный (< 50) - карточки игр
        has_bright_center = center_mean > 60
        has_character = center_std > 30 and center_mean > 50
        
        _log_debug(f"_detect_fortnite_lobby: center_mean={center_mean:.1f}, center_std={center_std:.1f}")
        
        # ============================================================  
        # 3. Проверяем карусель игр внизу
        # ============================================================
        bottom_roi = gray[int(h*0.78):int(h*0.95), int(w*0.05):int(w*0.95)]
        bottom_std = np.std(bottom_roi)
        bottom_bright = np.count_nonzero(bottom_roi > 100) / bottom_roi.size
        has_carousel = bottom_std > 35 and bottom_bright > 0.15
        
        _log_debug(f"_detect_fortnite_lobby: bottom_std={bottom_std:.1f}, bottom_bright={bottom_bright:.4f}")
        
        # ============================================================
        # ФИНАЛЬНОЕ РЕШЕНИЕ
        # ============================================================
        
        # Меню + яркий центр (персонаж на фоне) = LOBBY
        if has_top_menu and has_bright_center and has_character:
            _log_debug("_detect_fortnite_lobby: ДА - меню + яркий центр + персонаж")
            return True
        
        # Меню + карусель + яркий экран = тоже LOBBY
        if has_top_menu and has_carousel and mean_brightness > 60:
            _log_debug("_detect_fortnite_lobby: ДА - меню + карусель + яркий экран")
            return True
        
        _log_debug("_detect_fortnite_lobby: НЕТ")
        return False
    except Exception as e:
        _log_debug(f"_detect_fortnite_lobby error: {e}")
        return False


def detect_fortnite_lobby_on_page(page) -> bool:
    """Определяет, находится ли игра в лобби Fortnite."""
    try:
        img = capture_page_bgr(page)
        return _detect_fortnite_lobby(img)
    except Exception:
        return False


def find_search_icon(page_or_img) -> Optional[Tuple[int, int, int, int]]:
    """
    Ищет иконку поиска (лупу) в лобби Fortnite.
    
    Лупа обычно находится в верхней левой части экрана.
    
    Args:
        page_or_img: Playwright page или BGR изображение
        
    Returns:
        (x, y, w, h) координаты иконки или None если не найдена
    """
    try:
        img = _get_image(page_or_img)
        h, w = img.shape[:2]
        
        # Область поиска: верхняя левая часть экрана (35% по ширине, 25% по высоте)
        roi = img[0:int(h*0.25), 0:int(w*0.35)]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        debug_info = {"roi_size": f"{roi.shape[1]}x{roi.shape[0]}"}
        
        # Пытаемся найти template
        search_template_path = os.path.join(ROOT_DIR, 'assets', 'search_icon.png')
        
        best_val = 0
        found = None
        
        if os.path.exists(search_template_path):
            template = cv2.imread(search_template_path, cv2.IMREAD_GRAYSCALE)
            if template is not None:
                debug_info["template_size"] = f"{template.shape[1]}x{template.shape[0]}"
                
                # Multi-scale template matching
                for scale in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6]:
                    resized_template = cv2.resize(template, None, fx=scale, fy=scale)
                    if resized_template.shape[0] > roi_gray.shape[0] or resized_template.shape[1] > roi_gray.shape[1]:
                        continue
                    
                    result = cv2.matchTemplate(roi_gray, resized_template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_val:
                        best_val = max_val
                        th, tw = resized_template.shape[:2]
                        found = (max_loc[0], max_loc[1], tw, th, scale)
                
                debug_info["best_confidence"] = f"{best_val:.3f}"
                
                if best_val > 0.45:  # Порог уверенности (снижен с 0.5)
                    x, y, found_w, found_h, scale = found
                    _log_debug(f"find_search_icon: найдена лупа с confidence={best_val:.3f} at ({x}, {y}) scale={scale}")
                    debug_info["found"] = f"({x}, {y}) {found_w}x{found_h}"
                    debug_info["scale"] = f"{scale}"
                    
                    # Сохраняем debug изображение с отмеченной областью
                    if _VISION_DEBUG:
                        debug_img = roi.copy()
                        cv2.rectangle(debug_img, (x, y), (x + found_w, y + found_h), (0, 255, 0), 2)
                        _save_debug_image(debug_img, "SEARCH_ICON_FOUND", debug_info)
                    
                    return (x, y, found_w, found_h)
                else:
                    _log_debug(f"find_search_icon: лупа не найдена, best_val={best_val:.3f}")
        else:
            debug_info["error"] = "template not found"
            _log_debug(f"find_search_icon: файл шаблона не найден: {search_template_path}")
        
        # Сохраняем debug изображение когда лупа не найдена
        if _VISION_DEBUG:
            _save_debug_image(roi, "SEARCH_ICON_NOT_FOUND", debug_info)
        
        # Альтернатива: ищем по характерному контуру круга с "ручкой"
        # Бинаризация для поиска светлых элементов
        _, binary = cv2.threshold(roi_gray, 180, 255, cv2.THRESH_BINARY)
        
        # Ищем контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 100 < area < 5000:  # Примерный размер иконки
                x, y, cnt_w, cnt_h = cv2.boundingRect(cnt)
                aspect = cnt_w / cnt_h if cnt_h > 0 else 0
                
                # Лупа примерно квадратная или слегка вытянутая (с ручкой)
                if 0.7 < aspect < 1.5:
                    # Проверяем circularity
                    perimeter = cv2.arcLength(cnt, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    if circularity > 0.3:  # Достаточно круглый
                        _log_debug(f"find_search_icon: найден кандидат лупы по контуру at ({x}, {y})")
                        return (x, y, cnt_w, cnt_h)
        
        return None
    except Exception as e:
        _log_debug(f"find_search_icon error: {e}")
        return None


def detect_search_icon_on_page(page) -> bool:
    """
    Проверяет наличие иконки поиска на странице.
    
    Используется для подтверждения что игра находится в лобби.
    
    Returns:
        True если иконка поиска найдена
    """
    result = find_search_icon(page)
    return result is not None


def _detect_discover_screen(img: np.ndarray) -> bool:
    """
    Детектирует экран Discover (с Search Discover баром вверху).
    
    КЛЮЧЕВЫЕ ОТЛИЧИЯ от LOBBY:
    - ТЁМНЫЙ фон (brightness < 55) - это ГЛАВНОЕ отличие!
    - Search Discover бар слева вверху (серая полоса с текстом)
    - Вкладки: DISCOVER, FOLLOWING, BY EPIC, RECENTS, CATEGORIES
    - Карточки игр на тёмном фоне
    - НЕТ персонажа в центре, НЕТ яркого неба
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mean_brightness = np.mean(gray)
        
        # ============================================================
        # КЛЮЧЕВОЕ: DISCOVER ТЁМНЫЙ (< 55)!
        # LOBBY яркое (> 55), это главное отличие
        # ============================================================
        if mean_brightness > 55:
            _log_debug(f"_detect_discover: слишком яркий ({mean_brightness:.1f}) - это LOBBY, не DISCOVER")
            return False
        
        if mean_brightness < 15:
            _log_debug(f"_detect_discover: слишком тёмный ({mean_brightness:.1f}) - это загрузка")
            return False
        
        # ============================================================
        # 1. Проверяем верхнее меню (PLAY SHOP LOCKER...) 
        # На Discover экране тоже есть это меню!
        # ============================================================
        top_menu_roi = gray[int(h*0.03):int(h*0.09), int(w*0.12):int(w*0.70)]
        menu_white = np.count_nonzero(top_menu_roi > 200) / top_menu_roi.size
        
        _log_debug(f"_detect_discover: brightness={mean_brightness:.1f}, menu_white={menu_white:.4f}")
        
        # Должно быть меню (белый текст)
        if menu_white < 0.015:
            _log_debug("_detect_discover: нет верхнего меню - НЕ DISCOVER")
            return False
        
        # ============================================================
        # 2. Проверяем Search Discover бар (серая полоса слева вверху)
        # ============================================================
        search_bar_roi = gray[int(h*0.07):int(h*0.13), int(w*0.05):int(w*0.45)]
        search_mean = np.mean(search_bar_roi)
        search_white = np.count_nonzero(search_bar_roi > 180) / search_bar_roi.size
        
        _log_debug(f"_detect_discover: search_bar mean={search_mean:.1f}, white={search_white:.4f}")
        
        # Search bar: серый фон (35-90) с белым текстом
        has_search_bar = (30 < search_mean < 100) and (search_white > 0.008)
        
        # ============================================================
        # 3. Проверяем вкладки справа (DISCOVER, FOLLOWING...)
        # ============================================================
        tabs_roi = gray[int(h*0.07):int(h*0.13), int(w*0.48):int(w*0.98)]
        tabs_white = np.count_nonzero(tabs_roi > 200) / tabs_roi.size
        
        _log_debug(f"_detect_discover: tabs_white={tabs_white:.4f}")
        
        has_tabs = tabs_white > 0.025
        
        # ============================================================
        # 4. Проверяем карточки игр (яркие прямоугольники на тёмном фоне)
        # ============================================================
        cards_roi = gray[int(h*0.15):int(h*0.90), int(w*0.05):int(w*0.95)]
        cards_std = np.std(cards_roi)
        cards_bright = np.count_nonzero(cards_roi > 120) / cards_roi.size
        
        _log_debug(f"_detect_discover: cards_std={cards_std:.1f}, cards_bright={cards_bright:.4f}")
        
        # Карточки создают контраст на тёмном фоне
        has_cards = cards_std > 35 and cards_bright > 0.10
        
        # ============================================================
        # 5. Центр экрана ТЁМНЫЙ (нет яркого персонажа как в LOBBY)
        # ============================================================
        center_roi = gray[int(h*0.25):int(h*0.60), int(w*0.30):int(w*0.70)]
        center_mean = np.mean(center_roi)
        
        _log_debug(f"_detect_discover: center_mean={center_mean:.1f}")
        
        # В DISCOVER центр относительно тёмный (карточки, а не яркое небо)
        has_dark_center = center_mean < 80
        
        # ============================================================
        # ФИНАЛЬНОЕ РЕШЕНИЕ
        # ============================================================
        
        # Тёмный экран + Search bar + карточки = DISCOVER
        if has_search_bar and has_cards and has_dark_center:
            _log_debug("_detect_discover: ДА - search bar + cards + тёмный центр")
            return True
        
        # Тёмный экран + вкладки + карточки = тоже DISCOVER
        if has_tabs and has_cards and has_dark_center:
            _log_debug("_detect_discover: ДА - tabs + cards + тёмный центр")
            return True
        
        # Search bar + меню + тёмный экран = DISCOVER
        if has_search_bar and menu_white > 0.02 and mean_brightness < 50:
            _log_debug("_detect_discover: ДА - search bar + меню + тёмный")
            return True
        
        _log_debug("_detect_discover: НЕТ")
        return False
    except Exception as e:
        _log_debug(f"_detect_discover error: {e}")
        return False


def _detect_search_input_dialog(img: np.ndarray) -> bool:
    """
    Детектирует диалог ввода кода острова (Search Discover popup).
    
    КЛЮЧЕВЫЕ ПРИЗНАКИ:
    - Модальное ПРЯМОУГОЛЬНОЕ окно в центре (серый прямоугольник)
    - Заголовок "Search Discover" белым текстом
    - Поле ввода (белое прямоугольное)
    - Кнопка ODESLAT (зелёная прямоугольная)
    - Затемнённый фон вокруг
    
    НЕ ПУТАТЬ С:
    - Экран загрузки с анимацией (ракета, круги)
    - TITLE_SCREEN с логотипом
    - Уведомления/popup окна
    
    ВАЖНО: Требуется ЗЕЛЁНАЯ кнопка или белое поле ввода!
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        mean_brightness = np.mean(gray)
        
        # ============================================================
        # ИСКЛЮЧЕНИЕ: Экран загрузки (тёмный с анимацией)
        # На экране загрузки brightness < 25 и нет UI элементов
        # ============================================================
        if mean_brightness < 20:
            _log_debug(f"_detect_search_input: слишком тёмный ({mean_brightness:.1f}) - это загрузка")
            return False
        
        # Проверяем есть ли верхнее меню (PLAY SHOP LOCKER...)
        # Search Input диалог появляется НА Discover экране, где есть меню
        top_menu = gray[int(h*0.03):int(h*0.09), int(w*0.12):int(w*0.70)]
        menu_white = np.count_nonzero(top_menu > 200) / top_menu.size
        
        # Если нет меню и экран очень тёмный - это загрузка
        if menu_white < 0.01 and mean_brightness < 30:
            _log_debug(f"_detect_search_input: нет меню + тёмный экран = загрузка")
            return False
        
        # ============================================================
        # ГЛАВНЫЙ ПРИЗНАК: ЗЕЛЁНАЯ КНОПКА ODESLAT
        # ============================================================
        # Кнопка ODESLAT находится в центре-нижней части диалога
        button_roi = hsv[int(h*0.38):int(h*0.55), int(w*0.35):int(w*0.65)]
        green_lower = np.array([35, 80, 80], dtype=np.uint8)
        green_upper = np.array([85, 255, 255], dtype=np.uint8)
        green_mask = cv2.inRange(button_roi, green_lower, green_upper)
        green_ratio = np.count_nonzero(green_mask) / green_mask.size
        
        _log_debug(f"_detect_search_input: brightness={mean_brightness:.1f}, menu={menu_white:.4f}, green_button={green_ratio:.4f}")
        
        # Зелёная кнопка - надёжный признак диалога
        if green_ratio > 0.03:
            _log_debug("_detect_search_input: ОБНАРУЖЕНО (зелёная кнопка ODESLAT)")
            return True
        
        # ============================================================
        # ВТОРИЧНЫЙ ПРИЗНАК: Модальный диалог (светлый центр)
        # ============================================================
        # Центральная область должна быть СВЕТЛОЙ (серый диалог)
        center_roi = gray[int(h*0.25):int(h*0.50), int(w*0.30):int(w*0.70)]
        center_mean = np.mean(center_roi)
        center_std = np.std(center_roi)
        
        # Края должны быть ТЁМНЫМИ (затемнение)
        left_edge = np.mean(gray[int(h*0.25):int(h*0.50), int(w*0.05):int(w*0.20)])
        right_edge = np.mean(gray[int(h*0.25):int(h*0.50), int(w*0.80):int(w*0.95)])
        edge_mean = (left_edge + right_edge) / 2
        
        _log_debug(f"_detect_search_input: center={center_mean:.1f}, edges={edge_mean:.1f}, diff={center_mean - edge_mean:.1f}")
        
        # Диалог: центр ЗНАЧИТЕЛЬНО светлее краёв (минимум +25)
        # И центр должен быть серым (60-150), не слишком ярким
        is_modal = (center_mean > edge_mean + 25) and (60 < center_mean < 150)
        
        if is_modal:
            # Дополнительно: ищем белое поле ввода
            input_roi = gray[int(h*0.30):int(h*0.40), int(w*0.32):int(w*0.68)]
            input_very_bright = np.count_nonzero(input_roi > 200) / input_roi.size
            
            _log_debug(f"_detect_search_input: modal detected, input_bright={input_very_bright:.4f}")
            
            # Белое поле ввода (> 5% очень ярких пикселей)
            if input_very_bright > 0.05:
                _log_debug("_detect_search_input: ОБНАРУЖЕНО (модальный диалог + поле ввода)")
                return True
        
        _log_debug("_detect_search_input: НЕ обнаружено")
        return False
    except Exception as e:
        _log_debug(f"_detect_search_input error: {e}")
        return False


def _detect_search_results(img: np.ndarray) -> bool:
    """
    Детектирует результаты поиска (карточки островов).
    
    Характерные признаки:
    - Крупные карточки островов в центре/левой части
    - Яркие изображения на тёмном фоне
    - Код острова виден на карточках
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        
        # Результаты обычно на тёмном фоне (30-80)
        if mean_brightness < 20 or mean_brightness > 100:
            return False
        
        # Область карточек (левая/центральная часть экрана)
        cards_roi = gray[int(h*0.20):int(h*0.80), int(w*0.05):int(w*0.60)]
        
        # Карточки создают высокий контраст
        cards_std = np.std(cards_roi)
        
        # Яркие пиксели (картинки на карточках)
        bright_pixels = np.count_nonzero(cards_roi > 150) / cards_roi.size
        
        _log_debug(f"_detect_search_results: cards_std={cards_std:.1f}, bright={bright_pixels:.4f}")
        
        # Высокий контраст + яркие области
        if cards_std > 45 and bright_pixels > 0.05:
            _log_debug("_detect_search_results: ОБНАРУЖЕНО")
            return True
        
        return False
    except Exception as e:
        _log_debug(f"_detect_search_results error: {e}")
        return False


def _detect_island_preview(img: np.ndarray) -> bool:
    """
    Детектирует превью острова (перед запуском).
    
    Характерные признаки НАСТОЯЩЕГО Island Preview:
    - Большое изображение острова СЛЕВА (занимает ~40-50% экрана)
    - Информация об острове СПРАВА (название, автор, описание)
    - Кнопки SELECT/PLAY внизу ПО ЦЕНТРУ
    - НЕТ верхнего меню (PLAY SHOP LOCKER...)
    - НЕТ карусели игр внизу
    - НЕТ персонажа в центре
    
    ВАЖНО: Лобби и Title Screen НЕ должны определяться как Island Preview!
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_brightness = np.mean(gray)
        
        # ИСКЛЮЧЕНИЕ 1: Проверяем НЕТ ЛИ верхнего меню (PLAY SHOP LOCKER...)
        # В лобби есть белое меню вверху, в Island Preview его нет
        top_menu_roi = gray[int(h*0.02):int(h*0.10), int(w*0.10):int(w*0.60)]
        top_white = np.count_nonzero(top_menu_roi > 200) / top_menu_roi.size
        
        if top_white > 0.03:  # Есть белое меню - это лобби, не island preview
            _log_debug(f"_detect_island_preview: есть верхнее меню (white={top_white:.4f}) - НЕ island preview")
            return False
        
        # ИСКЛЮЧЕНИЕ 2: Проверяем НЕТ ЛИ карусели игр внизу
        # В лобби внизу карусель с карточками игр (высокий контраст)
        carousel_roi = gray[int(h*0.80):, int(w*0.05):int(w*0.95)]
        carousel_std = np.std(carousel_roi)
        
        # Карусель создаёт контраст > 50, Island Preview внизу обычно однородный
        if carousel_std > 55:
            _log_debug(f"_detect_island_preview: карусель внизу (std={carousel_std:.1f}) - НЕ island preview")
            return False
        
        # ИСКЛЮЧЕНИЕ 3: Title Screen (фиолетовый с логотипом FORTNITE)
        # Проверяем фиолетовый цвет
        purple_lower = np.array([120, 30, 50], dtype=np.uint8)
        purple_upper = np.array([160, 255, 255], dtype=np.uint8)
        purple_mask = cv2.inRange(hsv, purple_lower, purple_upper)
        purple_ratio = np.count_nonzero(purple_mask) / purple_mask.size
        
        if purple_ratio > 0.10:  # Много фиолетового - это title screen
            _log_debug(f"_detect_island_preview: много фиолетового ({purple_ratio:.4f}) - НЕ island preview (title screen)")
            return False
        
        # Теперь проверяем признаки НАСТОЯЩЕГО Island Preview
        
        # 1. Левая часть - большое изображение острова
        left_roi = gray[int(h*0.15):int(h*0.70), int(w*0.05):int(w*0.45)]
        left_std = np.std(left_roi)
        left_mean = np.mean(left_roi)
        
        # 2. Правая часть - информация (текст на тёмном фоне, меньше контраста)
        right_roi = gray[int(h*0.15):int(h*0.70), int(w*0.55):int(w*0.95)]
        right_std = np.std(right_roi)
        right_mean = np.mean(right_roi)
        
        # 3. Нижняя центральная часть - кнопки SELECT/PLAY
        bottom_center = gray[int(h*0.80):int(h*0.95), int(w*0.35):int(w*0.65)]
        bottom_bright = np.count_nonzero(bottom_center > 180) / bottom_center.size
        
        _log_debug(f"_detect_island_preview: left_std={left_std:.1f}, right_std={right_std:.1f}, bottom_bright={bottom_bright:.4f}")
        
        # Island Preview: картинка слева (контраст), инфо справа (менее контрастно), кнопки внизу
        has_left_image = left_std > 35 and left_mean > 40
        has_right_info = right_std < left_std  # Справа меньше контраста чем слева
        has_bottom_buttons = bottom_bright > 0.05  # Яркие кнопки внизу по центру
        
        if has_left_image and has_right_info and has_bottom_buttons:
            _log_debug("_detect_island_preview: ОБНАРУЖЕНО (реальный island preview)")
            return True
        
        return False
    except Exception as e:
        _log_debug(f"_detect_island_preview error: {e}")
        return False


def _detect_lobby_with_character(img: np.ndarray) -> bool:
    """
    Детектирует лобби Fortnite именно с персонажем в центре.
    
    Отличие от DISCOVER: 
    - Есть крупный персонаж в центре экрана
    - ЯРКИЙ экран (brightness > 40) - у Discover тёмный фон
    """
    try:
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        mean_brightness = np.mean(gray)
        
        # КЛЮЧЕВОЕ: Лобби с персонажем ЯРКОЕ (>40)!
        # Discover тёмный (20-60), тут мы хотим яркий экран
        if mean_brightness < 40 or mean_brightness > 150:
            _log_debug(f"_detect_lobby_character: brightness={mean_brightness:.1f} - не подходит (нужно 40-150)")
            return False
        
        # Проверяем верхнее меню (PLAY, SHOP, LOCKER...)
        top_roi = gray[int(h*0.02):int(h*0.10), int(w*0.10):int(w*0.60)]
        top_white = np.count_nonzero(top_roi > 200) / top_roi.size
        top_edges = cv2.Canny(top_roi, 100, 200)
        top_edge_ratio = np.count_nonzero(top_edges) / top_edges.size
        
        has_menu = top_white > 0.02 and top_edge_ratio > 0.01
        
        if not has_menu:
            _log_debug(f"_detect_lobby_character: нет меню - НЕ лобби")
            return False
        
        # Центральная область (где персонаж)
        center_roi = gray[int(h*0.20):int(h*0.75), int(w*0.30):int(w*0.70)]
        center_std = np.std(center_roi)
        center_mean = np.mean(center_roi)
        
        # Нижняя часть (карусель игр)
        bottom_roi = gray[int(h*0.80):, int(w*0.10):int(w*0.90)]
        bottom_std = np.std(bottom_roi)
        
        _log_debug(f"_detect_lobby_character: brightness={mean_brightness:.1f}, center_std={center_std:.1f}, center_mean={center_mean:.1f}, bottom_std={bottom_std:.1f}")
        
        # Персонаж создаёт контраст в центре (std > 30) и центр не слишком тёмный (mean > 50)
        has_character = center_std > 30 and center_mean > 50
        
        # Карусель игр создаёт контраст внизу (std > 35)
        has_carousel = bottom_std > 35
        
        _log_debug(f"_detect_lobby_character: has_menu={has_menu}, has_character={has_character}, has_carousel={has_carousel}")
        
        # Нужно меню + (персонаж ИЛИ карусель)
        if has_menu and (has_character or has_carousel):
            _log_debug("_detect_lobby_character: ОБНАРУЖЕНО")
            return True
        
        return False
    except Exception as e:
        _log_debug(f"_detect_lobby_character error: {e}")
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
    """
    Анализирует состояние экрана.
    
    ГЛАВНЫЙ МЕТОД: Template matching по UI элементам из assets.
    FALLBACK: Эвристический анализ цветов/яркости.
    """
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
    
    # Базовые метрики для логирования
    base_metrics = {"brightness": mean_brightness, "contrast": std_brightness}
    
    _log_debug(f"Анализ экрана: размер={w}x{h}, яркость={mean_brightness:.1f}, контраст={std_brightness:.1f}")
    
    # ============================================================
    # МЕТОД 1: TEMPLATE MATCHING (ОСНОВНОЙ)
    # ============================================================
    # Ищем UI элементы на экране через template matching
    found_elements = _detect_ui_elements(img)
    
    if found_elements:
        # Берём состояние от элемента с наивысшим приоритетом
        best_element = found_elements[0]
        state = best_element['state']
        
        _log_debug(f"Template matching: найдено {len(found_elements)} элементов")
        _log_debug(f"  Лучший элемент: {best_element['name']} -> {state.name} (conf={best_element['confidence']:.3f})")
        
        debug_info["template_method"] = "True"
        debug_info["best_element"] = f"{best_element['name']} ({best_element['confidence']:.0%})"
        debug_info["elements_count"] = str(len(found_elements))
        
        _log_detection_result(state.name, f"найден UI элемент {best_element['name']}", 
                             {"confidence": best_element['confidence'], **base_metrics})
        _save_debug_image(img, state.name, debug_info, found_elements)
        return state
    
    _log_debug("Template matching: UI элементы не найдены, переход к эвристикам")
    debug_info["template_method"] = "False (fallback)"
    
    # ============================================================
    # МЕТОД 2: ЭВРИСТИЧЕСКИЙ АНАЛИЗ (FALLBACK)
    # ============================================================
    
    # 1. Проверяем Title Screen (фиолетовый экран БЕЗ верхнего меню)
    is_title = _detect_title_screen(img)
    debug_info["title_check"] = str(is_title)
    _log_debug(f"  Проверка TITLE_SCREEN: {is_title}")
    if is_title:
        _log_detection_result("TITLE_SCREEN", "титульный экран Fortnite", base_metrics)
        _save_debug_image(img, "TITLE_SCREEN", debug_info, found_elements)
        return ScreenState.TITLE_SCREEN
    
    # 2. Проверяем лобби с персонажем (яркий экран, меню вверху, персонаж в центре)
    is_lobby_char = _detect_lobby_with_character(img)
    debug_info["lobby_char_check"] = str(is_lobby_char)
    _log_debug(f"  Проверка LOBBY (с персонажем): {is_lobby_char}")
    if is_lobby_char:
        _log_detection_result("LOBBY", "лобби Fortnite с персонажем", base_metrics)
        _save_debug_image(img, "LOBBY", debug_info, found_elements)
        return ScreenState.LOBBY
    
    # 3. Проверяем Discover экран (тёмный фон, Search Discover бар вверху)
    # ВАЖНО: ДО общей проверки lobby! У Discover тёмный фон, у lobby яркий.
    is_discover = _detect_discover_screen(img)
    debug_info["discover_check"] = str(is_discover)
    _log_debug(f"  Проверка DISCOVER: {is_discover}")
    if is_discover:
        _log_detection_result("DISCOVER", "экран Discover с Search баром", base_metrics)
        _save_debug_image(img, "DISCOVER", debug_info, found_elements)
        return ScreenState.DISCOVER
    
    # 4. Общая проверка лобби (меню + play button)
    is_lobby = _detect_fortnite_lobby(img)
    debug_info["lobby_check"] = str(is_lobby)
    _log_debug(f"  Проверка FORTNITE_LOBBY: {is_lobby}")
    if is_lobby:
        _log_detection_result("LOBBY", "лобби Fortnite", base_metrics)
        _save_debug_image(img, "LOBBY", debug_info, found_elements)
        return ScreenState.LOBBY
    
    # 5. Проверяем Search Input Dialog (диалог ввода кода)
    is_search_input = _detect_search_input_dialog(img)
    debug_info["search_input_check"] = str(is_search_input)
    _log_debug(f"  Проверка SEARCH_INPUT: {is_search_input}")
    if is_search_input:
        _log_detection_result("SEARCH_INPUT", "диалог ввода кода острова", base_metrics)
        _save_debug_image(img, "SEARCH_INPUT", debug_info, found_elements)
        return ScreenState.SEARCH_INPUT
    
    # 6. Проверяем Search Results (результаты поиска с карточками островов)
    is_search_results = _detect_search_results(img)
    debug_info["search_results_check"] = str(is_search_results)
    _log_debug(f"  Проверка SEARCH_RESULTS: {is_search_results}")
    if is_search_results:
        _log_detection_result("SEARCH_RESULTS", "результаты поиска островов", base_metrics)
        _save_debug_image(img, "SEARCH_RESULTS", debug_info, found_elements)
        return ScreenState.SEARCH_RESULTS
    
    # 7. Проверяем Island Preview (превью острова перед запуском)
    # ПОСЛЕДНИМ из специфичных экранов Fortnite
    is_island_preview = _detect_island_preview(img)
    debug_info["island_preview_check"] = str(is_island_preview)
    _log_debug(f"  Проверка ISLAND_PREVIEW: {is_island_preview}")
    if is_island_preview:
        _log_detection_result("ISLAND_PREVIEW", "превью острова (SELECT/PLAY)", base_metrics)
        _save_debug_image(img, "ISLAND_PREVIEW", debug_info, found_elements)
        return ScreenState.ISLAND_PREVIEW
    
    # ВАЖНО: Если экран ОЧЕНЬ яркий (>130) и это НЕ лобби - это может быть страница Xbox
    if mean_brightness > 130:
        debug_info["note"] = "Очень яркий экран - возможно сайт Xbox"
        _log_debug(f"  Очень яркий экран (brightness={mean_brightness:.1f})")
        
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
                _log_detection_result("LOGIN_PAGE", "страница входа Microsoft", {"white": white_ratio*100, "blue": blue_ratio*100})
                _save_debug_image(img, "LOGIN_PAGE", debug_info, found_elements)
                return ScreenState.LOGIN_PAGE
        
        # Очень яркий экран но не login и не лобби - UNKNOWN
        _log_debug("  Состояние: UNKNOWN (очень яркий экран)")
        _save_debug_image(img, "UNKNOWN_BRIGHT", debug_info, found_elements)
        return ScreenState.UNKNOWN
    
    # 1. CONNECTING оверлей (только на ОЧЕНЬ тёмном фоне < 60)
    if mean_brightness < 60:
        is_connecting = _detect_connecting_overlay(img)
        debug_info["connecting_check"] = str(is_connecting)
        _log_debug(f"  Проверка CONNECTING (яркость<60): {is_connecting}")
        if is_connecting:
            _log_detection_result("CONNECTING", "оверлей подключения", base_metrics)
            _save_debug_image(img, "CONNECTING", debug_info, found_elements)
            return ScreenState.CONNECTING
    
    # 2. Xbox Queue (очередь Game Pass) - тёмный экран с рекламой и прогресс-баром
    if mean_brightness < 80:
        is_queue = _detect_xbox_queue(img)
        debug_info["queue_check"] = str(is_queue)
        _log_debug(f"  Проверка XBOX_QUEUE (яркость<80): {is_queue}")
        if is_queue:
            _log_detection_result("XBOX_QUEUE", "очередь Xbox Cloud Gaming", base_metrics)
            _save_debug_image(img, "XBOX_QUEUE", debug_info, found_elements)
            return ScreenState.XBOX_QUEUE
    
    # 3. Xbox Logo Screen (большой логотип XBOX по центру) - ПРОВЕРЯЕМ ДО plane screen!
    if mean_brightness < 45:
        is_xbox_logo = _detect_xbox_logo_screen(img)
        debug_info["xbox_logo_check"] = str(is_xbox_logo)
        _log_debug(f"  Проверка XBOX_LOGO (яркость<45): {is_xbox_logo}")
        if is_xbox_logo:
            _log_detection_result("XBOX_LOADING", "экран с логотипом XBOX", base_metrics)
            _save_debug_image(img, "XBOX_LOADING", debug_info, found_elements)
            return ScreenState.XBOX_LOADING
    
    # 4. Plane screen (зелёный самолёт)
    is_plane = _detect_plane_screen(img)
    debug_info["plane_check"] = str(is_plane)
    _log_debug(f"  Проверка PLANE_SCREEN: {is_plane}")
    if is_plane:
        _log_detection_result("PLANE_SCREEN", "зелёный самолётик Xbox", base_metrics)
        _save_debug_image(img, "PLANE_SCREEN", debug_info, found_elements)
        return ScreenState.PLANE_SCREEN
    
    # 5. Xbox loading (зелёный или белый логотип Xbox)
    # ВАЖНО: Исключаем случаи когда есть диалог (Search Discover popup)
    if mean_brightness < 40:
        # ИСКЛЮЧЕНИЕ: Проверяем нет ли яркого диалога в центре экрана
        dialog_center = gray[int(h*0.20):int(h*0.55), int(w*0.30):int(w*0.70)]
        dialog_bright = np.count_nonzero(dialog_center > 100) / dialog_center.size
        
        # Проверяем нет ли зелёной кнопки (ODESLAT) в области диалога
        dialog_hsv = hsv[int(h*0.35):int(h*0.55), int(w*0.35):int(w*0.65)]
        green_btn_lower = np.array([35, 80, 80], dtype=np.uint8)
        green_btn_upper = np.array([85, 255, 255], dtype=np.uint8)
        green_btn_mask = cv2.inRange(dialog_hsv, green_btn_lower, green_btn_upper)
        green_btn_ratio = np.count_nonzero(green_btn_mask) / green_btn_mask.size
        
        has_dialog = dialog_bright > 0.35 or green_btn_ratio > 0.015
        
        debug_info["dialog_bright"] = f"{dialog_bright:.4f}"
        debug_info["green_btn"] = f"{green_btn_ratio:.4f}"
        
        if has_dialog:
            _log_debug(f"  XBOX_LOADING пропущен - обнаружен диалог (bright={dialog_bright:.4f}, green_btn={green_btn_ratio:.4f})")
        else:
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
                _log_detection_result("XBOX_LOADING", "загрузка Xbox Cloud Gaming", {"green": xbox_green_ratio*100, "white_logo": white_logo_ratio*100})
                _save_debug_image(img, "XBOX_LOADING", debug_info, found_elements)
                return ScreenState.XBOX_LOADING
            
            # Очень тёмный экран (почти чёрный) - тоже загрузка
            if mean_brightness < 10 and std_brightness < 20:
                _log_detection_result("LOADING", "очень тёмный экран", base_metrics)
                _save_debug_image(img, "LOADING", debug_info, found_elements)
                return ScreenState.LOADING
            
            if std_brightness < 30:
                _log_detection_result("LOADING", "тёмный экран загрузки", base_metrics)
                _save_debug_image(img, "LOADING", debug_info, found_elements)
                return ScreenState.LOADING
    
    # 6. Титульный экран Fortnite (яркий, фиолетовый, с персонажами)
    is_title = _detect_title_screen(img)
    debug_info["title_check"] = str(is_title)
    _log_debug(f"  Проверка TITLE_SCREEN: {is_title}")
    if is_title:
        _log_detection_result("TITLE_SCREEN", "титульный экран Fortnite", base_metrics)
        _save_debug_image(img, "TITLE_SCREEN", debug_info, found_elements)
        return ScreenState.TITLE_SCREEN
    
    # 6.5. Лобби Fortnite (жёлтая кнопка PLAY, персонаж, UI вкладки)
    is_lobby = _detect_fortnite_lobby(img)
    debug_info["lobby_check"] = str(is_lobby)
    _log_debug(f"  Проверка FORTNITE_LOBBY: {is_lobby}")
    if is_lobby:
        _log_detection_result("LOBBY", "обнаружена кнопка PLAY и UI", base_metrics)
        _save_debug_image(img, "LOBBY", debug_info, found_elements)
        return ScreenState.LOBBY
    
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
            _log_detection_result("LOGIN_PAGE", "страница входа Microsoft", {"white": white_ratio*100, "blue": blue_ratio*100})
            _save_debug_image(img, "LOGIN_PAGE", debug_info, found_elements)
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
            _log_detection_result("IN_GAME", "в игре (много UI элементов)", {"top_ui": top_edge_ratio*100, "bottom_ui": bottom_edge_ratio*100})
            _save_debug_image(img, "IN_GAME", debug_info, found_elements)
            return ScreenState.IN_GAME
        
        if top_edge_ratio > 0.01 or bottom_edge_ratio > 0.02:
            _log_detection_result("LOBBY", "обнаружен UI сверху/снизу", {"top_ui": top_edge_ratio*100, "bottom_ui": bottom_edge_ratio*100})
            _save_debug_image(img, "LOBBY", debug_info, found_elements)
            return ScreenState.LOBBY
    
    # 8. Меню/диалог
    center_roi = img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
    center_bright = np.mean(cv2.cvtColor(center_roi, cv2.COLOR_BGR2GRAY))
    edge_bright = (np.mean(gray[:int(h*0.2), :]) + np.mean(gray[int(h*0.8):, :])) / 2
    
    debug_info["center_bright"] = f"{center_bright:.1f}"
    debug_info["edge_bright"] = f"{edge_bright:.1f}"
    _log_debug(f"  Проверка MENU: center_bright={center_bright:.1f}, edge_bright={edge_bright:.1f}")
    if center_bright > edge_bright * 1.5 and center_bright > 80:
        _log_detection_result("MENU", "открыто меню/диалог", {"center": center_bright, "edge": edge_bright})
        _save_debug_image(img, "MENU", debug_info, found_elements)
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
        _log_detection_result("ERROR", "обнаружен красный цвет ошибки", {"red": red_ratio*100})
        _save_debug_image(img, "ERROR", debug_info, found_elements)
        return ScreenState.ERROR
    
    # Не удалось определить состояние
    _log_detection_result("UNKNOWN", "не удалось определить", base_metrics)
    _save_debug_image(img, "UNKNOWN", debug_info, found_elements)
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
