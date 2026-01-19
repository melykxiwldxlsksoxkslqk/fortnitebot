"""
Browser модуль - работа с браузером.

Содержит:
- manager: Управление браузером
- camoufox: Анти-детект браузер
- playwright: Базовый Playwright браузер
- input: Эмуляция ввода
"""

from .manager import (
    BrowserManager,
    create_browser,
    close_browser,
)
from .input import (
    press_key,
    press_action,
    click_at,
    move_mouse,
    type_text,
    hold_key,
    click_canvas,
    focus_canvas,
)

__all__ = [
    # Manager
    'BrowserManager',
    'create_browser',
    'close_browser',
    # Input
    'press_key',
    'press_action',
    'click_at',
    'move_mouse',
    'type_text',
    'hold_key',
    'click_canvas',
    'focus_canvas',
]
