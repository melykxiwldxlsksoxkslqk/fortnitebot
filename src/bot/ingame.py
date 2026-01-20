"""
In-game actions module for Fortnite.
Handles active gameplay actions, lobby waiting, and game state management.
"""

import time
import random
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from playwright.sync_api import Page

from ..core.logger import get_logger

logger = get_logger(__name__)


def _emit_status(message: str, status_callback: Optional[Callable] = None) -> None:
    """Emit status message if callback is provided."""
    logger.info(message)
    if status_callback:
        try:
            status_callback(message)
        except Exception:
            pass


def lock_mouse_into_stream(page: "Page") -> bool:
    """
    Lock mouse into the game stream area.
    
    Args:
        page: Playwright page object
        
    Returns:
        True if successful
    """
    try:
        # Find stream canvas/video
        stream_selectors = [
            'canvas#StreamCanvas',
            'video',
            '[data-testid="stream-video"]',
            '.stream-container',
        ]
        
        for sel in stream_selectors:
            try:
                el = page.locator(sel).first
                if el and el.is_visible(timeout=2000):
                    box = el.bounding_box()
                    if box:
                        # Click center of stream
                        cx = int(box['x'] + box['width'] / 2)
                        cy = int(box['y'] + box['height'] / 2)
                        page.mouse.click(cx, cy)
                        logger.info(f"Mouse locked to stream at ({cx}, {cy})")
                        return True
            except Exception:
                continue
                
        # Fallback: click center of viewport
        viewport = page.viewport_size
        if viewport:
            page.mouse.click(viewport['width'] // 2, viewport['height'] // 2)
            return True
            
    except Exception as e:
        logger.error(f"Failed to lock mouse: {e}")
        
    return False


def ensure_stream_focus(page: "Page") -> bool:
    """
    Ensure the game stream has focus for input.
    
    Args:
        page: Playwright page object
        
    Returns:
        True if focus established
    """
    try:
        # Click on stream area
        lock_mouse_into_stream(page)
        
        # Verify focus
        page.wait_for_timeout(200)
        return True
        
    except Exception as e:
        logger.error(f"Failed to ensure focus: {e}")
        return False


def do_active_ingame_actions(
    page: "Page",
    duration: int = 60000,
    status_callback: Optional[Callable] = None
) -> None:
    """
    Perform active in-game actions to keep the session alive.
    
    Args:
        page: Playwright page object
        duration: Duration of active actions in ms
        status_callback: Optional callback for status updates
    """
    _emit_status("Выполняю внутриигровые действия", status_callback)
    
    actions = [
        ('move_forward', lambda: page.keyboard.press('w')),
        ('move_back', lambda: page.keyboard.press('s')),
        ('move_left', lambda: page.keyboard.press('a')),
        ('move_right', lambda: page.keyboard.press('d')),
        ('look_around', lambda: _look_around(page)),
        ('jump', lambda: page.keyboard.press('Space')),
    ]
    
    elapsed = 0
    action_interval = 2000
    
    while elapsed < duration:
        try:
            # Pick random action
            action_name, action_func = random.choice(actions)
            action_func()
            logger.debug(f"Performed action: {action_name}")
        except Exception as e:
            logger.debug(f"Action failed: {e}")
            
        page.wait_for_timeout(action_interval)
        elapsed += action_interval
        
        # Status update every 30 seconds
        if elapsed % 30000 == 0:
            _emit_status(f"Активность... {elapsed // 1000}с", status_callback)


def _look_around(page: "Page") -> None:
    """Perform a random look around motion."""
    try:
        viewport = page.viewport_size
        if viewport:
            cx = viewport['width'] // 2
            cy = viewport['height'] // 2
            
            # Random offset
            dx = random.randint(-100, 100)
            dy = random.randint(-50, 50)
            
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx + dx, cy + dy, steps=5)
            page.mouse.up()
    except Exception:
        pass


def wait_for_game_load(
    page: "Page",
    status_callback: Optional[Callable] = None,
    timeout: int = 180000
) -> bool:
    """
    Wait for the game to fully load after island launch.
    
    Args:
        page: Playwright page object
        status_callback: Optional callback for status updates
        timeout: Maximum wait time in ms
        
    Returns:
        True if game loaded
    """
    _emit_status("Ожидаю загрузку игры", status_callback)
    
    # In-game indicators (things we should NOT see in loading)
    loading_indicators = [
        'Loading',
        'Connecting',
        'Please wait',
    ]
    
    elapsed = 0
    check_interval = 5000
    stable_count = 0
    required_stable = 3  # Need 3 consecutive checks without loading
    
    while elapsed < timeout:
        is_loading = False
        
        # Check for loading text
        for indicator in loading_indicators:
            try:
                el = page.get_by_text(indicator).first
                if el and el.is_visible(timeout=500):
                    is_loading = True
                    break
            except Exception:
                continue
                
        if is_loading:
            stable_count = 0
            _emit_status("Загрузка...", status_callback)
        else:
            stable_count += 1
            if stable_count >= required_stable:
                _emit_status("Игра загружена", status_callback)
                return True
                
        page.wait_for_timeout(check_interval)
        elapsed += check_interval
        
    _emit_status("Таймаут загрузки игры", status_callback)
    return False


def handle_game_menu(page: "Page") -> bool:
    """
    Handle any in-game menus that might appear.
    
    Args:
        page: Playwright page object
        
    Returns:
        True if menu was handled
    """
    menu_close_selectors = [
        'button:has-text("Close")',
        'button:has-text("OK")',
        'button:has-text("Continue")',
        '[data-testid="close-button"]',
        '.close-button',
    ]
    
    for sel in menu_close_selectors:
        try:
            el = page.locator(sel).first
            if el and el.is_visible(timeout=1000):
                el.click()
                page.wait_for_timeout(500)
                return True
        except Exception:
            continue
            
    # Try Escape key
    try:
        page.keyboard.press('Escape')
        return True
    except Exception:
        pass
        
    return False


def perform_afk_prevention(
    page: "Page",
    interval: int = 30000,
    status_callback: Optional[Callable] = None
) -> None:
    """
    Perform minimal actions to prevent AFK kick.
    
    Args:
        page: Playwright page object
        interval: Time between actions in ms
        status_callback: Optional callback for status updates
    """
    _emit_status("AFK-защита активна", status_callback)
    
    afk_actions = [
        lambda: page.mouse.move(640, 360),
        lambda: page.mouse.move(641, 361),
        lambda: page.keyboard.press('w'),
        lambda: page.keyboard.press('s'),
    ]
    
    action_idx = 0
    while True:
        try:
            action = afk_actions[action_idx % len(afk_actions)]
            action()
            action_idx += 1
        except Exception as e:
            logger.debug(f"AFK action failed: {e}")
            
        page.wait_for_timeout(interval)


def open_search(page: "Page") -> bool:
    """
    Open the in-game search interface.
    
    Args:
        page: Playwright page object
        
    Returns:
        True if search opened
    """
    try:
        # Try gamepad/keyboard shortcut first
        page.keyboard.press('/')
        page.wait_for_timeout(300)
        return True
    except Exception:
        pass
        
    # Try clicking search icon
    search_selectors = [
        '[aria-label*="Search"]',
        '.search-icon',
        'button:has-text("Search")',
    ]
    
    for sel in search_selectors:
        try:
            el = page.locator(sel).first
            if el and el.is_visible(timeout=2000):
                el.click()
                page.wait_for_timeout(300)
                return True
        except Exception:
            continue
            
    return False


def press_key_sequence(page: "Page", keys: list, delay: int = 100) -> None:
    """
    Press a sequence of keys with delay.
    
    Args:
        page: Playwright page object
        keys: List of keys to press
        delay: Delay between keys in ms
    """
    for key in keys:
        try:
            page.keyboard.press(key)
            page.wait_for_timeout(delay)
        except Exception as e:
            logger.debug(f"Key press failed for {key}: {e}")


def type_with_delay(page: "Page", text: str, delay: int = 50) -> None:
    """
    Type text with delay between characters.
    
    Args:
        page: Playwright page object
        text: Text to type
        delay: Delay between characters in ms
    """
    try:
        page.keyboard.type(text, delay=delay)
    except Exception as e:
        logger.error(f"Type failed: {e}")


def game_input_move(
    page: "Page",
    direction: str,
    duration: int = 500
) -> None:
    """
    Move in a direction for a duration.
    
    Args:
        page: Playwright page object
        direction: Direction ('forward', 'back', 'left', 'right')
        duration: Hold duration in ms
    """
    key_map = {
        'forward': 'w',
        'back': 's',
        'left': 'a',
        'right': 'd',
        'up': 'w',
        'down': 's',
    }
    
    key = key_map.get(direction.lower(), 'w')
    
    try:
        page.keyboard.down(key)
        page.wait_for_timeout(duration)
        page.keyboard.up(key)
    except Exception as e:
        logger.error(f"Move failed: {e}")


def game_input_look(
    page: "Page",
    dx: int = 0,
    dy: int = 0
) -> None:
    """
    Look/rotate the camera.
    
    Args:
        page: Playwright page object
        dx: Horizontal delta
        dy: Vertical delta
    """
    try:
        viewport = page.viewport_size
        if viewport:
            cx = viewport['width'] // 2
            cy = viewport['height'] // 2
            
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.mouse.move(cx + dx, cy + dy, steps=5)
            page.mouse.up()
    except Exception as e:
        logger.error(f"Look failed: {e}")
