"""
Эмуляция ввода для браузера.
"""

import os
import json
import time
from typing import Dict, List, Optional

from ..core import get_logger, ROOT_DIR

logger = get_logger(__name__)

# Маппинг клавиш геймпада
DEFAULT_INPUT_MAP: Dict[str, List[str]] = {
    "A": ["Enter", "KeyA"],
    "B": ["Escape", "Backspace", "KeyB"],
    "X": ["KeyX"],
    "Y": ["Slash", "KeyY"],
    "UP": ["ArrowUp"],
    "DOWN": ["ArrowDown"],
    "LEFT": ["ArrowLeft"],
    "RIGHT": ["ArrowRight"],
    "LB": ["BracketLeft"],
    "RB": ["BracketRight"],
    "LT": ["Minus"],
    "RT": ["Equal"],
    "MENU": ["KeyM", "Tab"],
    "VIEW": ["KeyV", "F1"],
    "NEXUS": ["KeyN"],
}

_INPUT_MAP: Dict[str, List[str]] = {}


def _load_input_map() -> Dict[str, List[str]]:
    """Загружает маппинг клавиш из конфига."""
    global _INPUT_MAP
    if _INPUT_MAP:
        return _INPUT_MAP
    
    cfg_path = os.path.join(ROOT_DIR, 'config', 'input_map.json')
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                mapped: Dict[str, List[str]] = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        mapped[str(k).upper()] = [str(x) for x in v]
                    else:
                        mapped[str(k).upper()] = [str(v)]
                _INPUT_MAP = {**DEFAULT_INPUT_MAP, **mapped}
                return _INPUT_MAP
    except Exception:
        pass
    
    _INPUT_MAP = DEFAULT_INPUT_MAP
    return _INPUT_MAP


def press_key(page, key: str, delay: int = 50) -> None:
    """
    Нажимает клавишу на странице.
    
    Args:
        page: Playwright page
        key: Код клавиши (например, 'Enter', 'KeyA')
        delay: Задержка после нажатия (мс)
    """
    try:
        page.keyboard.press(key)
        if delay > 0:
            page.wait_for_timeout(delay)
    except Exception as e:
        logger.debug(f"Ошибка нажатия клавиши {key}: {e}")


def press_action(page, action: str) -> None:
    """
    Нажимает действие из маппинга.
    
    Args:
        page: Playwright page
        action: Имя действия (A, B, X, Y, UP, DOWN, etc.)
    """
    mapping = _load_input_map()
    keys = mapping.get(str(action).upper())
    if not keys:
        return
    
    for key in keys:
        try:
            page.keyboard.press(key)
            page.wait_for_timeout(60)
        except Exception:
            pass


def click_at(page, x: int, y: int, button: str = 'left', delay: int = 100) -> None:
    """
    Кликает в указанной позиции.
    
    Args:
        page: Playwright page
        x, y: Координаты
        button: 'left', 'right', 'middle'
        delay: Задержка после клика (мс)
    """
    try:
        page.mouse.click(x, y, button=button)
        if delay > 0:
            page.wait_for_timeout(delay)
    except Exception as e:
        logger.debug(f"Ошибка клика в ({x}, {y}): {e}")


def move_mouse(page, x: int, y: int) -> None:
    """Перемещает курсор."""
    try:
        page.mouse.move(x, y)
    except Exception as e:
        logger.debug(f"Ошибка перемещения мыши: {e}")


def type_text(page, text: str, delay: int = 50) -> None:
    """
    Вводит текст.
    
    Args:
        page: Playwright page
        text: Текст для ввода
        delay: Задержка между символами (мс)
    """
    try:
        page.keyboard.type(text, delay=delay)
    except Exception as e:
        logger.debug(f"Ошибка ввода текста: {e}")


def hold_key(page, key: str, duration_ms: int = 500) -> None:
    """
    Удерживает клавишу.
    
    Args:
        page: Playwright page
        key: Код клавиши
        duration_ms: Длительность удержания (мс)
    """
    try:
        page.keyboard.down(key)
        page.wait_for_timeout(duration_ms)
        page.keyboard.up(key)
    except Exception as e:
        logger.debug(f"Ошибка удержания клавиши {key}: {e}")


def press_key_combo(page, *keys: str, delay: int = 50) -> None:
    """
    Нажимает комбинацию клавиш.
    
    Args:
        page: Playwright page
        keys: Клавиши (например, 'Control', 'Shift', 'KeyA')
        delay: Задержка (мс)
    """
    try:
        # Зажимаем все кроме последней
        for key in keys[:-1]:
            page.keyboard.down(key)
        
        # Нажимаем последнюю
        if keys:
            page.keyboard.press(keys[-1])
        
        # Отпускаем в обратном порядке
        for key in reversed(keys[:-1]):
            page.keyboard.up(key)
        
        if delay > 0:
            page.wait_for_timeout(delay)
    except Exception as e:
        logger.debug(f"Ошибка комбинации клавиш: {e}")


def scroll(page, delta_y: int = -300) -> None:
    """
    Прокручивает страницу.
    
    Args:
        page: Playwright page
        delta_y: Величина прокрутки (отрицательная = вверх)
    """
    try:
        page.mouse.wheel(0, delta_y)
    except Exception as e:
        logger.debug(f"Ошибка прокрутки: {e}")


# Canvas-специфичные функции

def _get_canvas_locator(page):
    """Находит canvas на странице или во фреймах."""
    try:
        loc = page.locator('canvas')
        if loc.count() > 0 and loc.first.is_visible():
            return loc.first
    except Exception:
        pass
    
    try:
        for fr in page.frames:
            try:
                loc = fr.locator('canvas')
                if loc.count() > 0 and loc.first.is_visible():
                    return loc.first
            except Exception:
                continue
    except Exception:
        pass
    
    return None


def click_canvas(page, x: int, y: int, button: str = 'left') -> bool:
    """
    Кликает по canvas элементу.
    
    Args:
        page: Playwright page
        x, y: Координаты относительно canvas
        button: Кнопка мыши
    
    Returns:
        True если успешно
    """
    try:
        canvas = _get_canvas_locator(page)
        if canvas:
            canvas.click(position={'x': x, 'y': y}, button=button)
            return True
    except Exception as e:
        logger.debug(f"Ошибка клика по canvas: {e}")
    
    # Fallback на обычный клик
    click_at(page, x, y, button)
    return True


def focus_canvas(page) -> bool:
    """
    Фокусирует canvas для ввода с клавиатуры.
    
    Returns:
        True если canvas найден и сфокусирован
    """
    try:
        canvas = _get_canvas_locator(page)
        if canvas:
            canvas.click()
            return True
    except Exception:
        pass
    return False
