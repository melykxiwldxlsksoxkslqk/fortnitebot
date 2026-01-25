"""
Vision модуль - компьютерное зрение и детекция.

Содержит:
- detection: Детекция элементов на экране
- templates: Работа с шаблонами изображений
- state: Определение состояния экрана
- yolo: YOLO детекция объектов
- capture: Захват экрана и видео
- utils: Утилиты для работы с изображениями
"""

from .detection import (
    find_template,
    find_template_multi,
    wait_for_template,
    detect_button,
    detect_text_region,
    detect_color_region,
    smart_find_element,
)
from .state import (
    ScreenState,
    detect_screen_state,
    wait_for_screen_state,
    get_screen_state_info,
    detect_connecting_overlay_on_page,
    detect_plane_screen_on_page,
    detect_loading_spinner,
    detect_xbox_loading_screen,
    detect_game_ready,
    set_vision_debug,
)
from .capture import (
    capture_screen,
    capture_page_bgr,
)
from .templates import (
    load_template,
    resolve_asset_path,
    clear_template_cache,
)
from .yolo_detector import (
    yolo_load_model,
    yolo_detect,
    yolo_detect_best,
)

__all__ = [
    # Detection
    'find_template',
    'find_template_multi',
    'wait_for_template',
    'detect_button',
    'detect_text_region',
    'detect_color_region',
    'smart_find_element',
    # State
    'ScreenState',
    'detect_screen_state',
    'wait_for_screen_state',
    'get_screen_state_info',
    'detect_connecting_overlay_on_page',
    'detect_plane_screen_on_page',
    'detect_loading_spinner',
    'detect_xbox_loading_screen',
    'detect_game_ready',
    'set_vision_debug',
    # Capture
    'capture_screen',
    'capture_page_bgr',
    # Templates
    'load_template',
    'resolve_asset_path',
    'clear_template_cache',
    # YOLO
    'yolo_load_model',
    'yolo_detect',
    'yolo_detect_best',
]
