"""
Island search and launch module for Fortnite.
Handles island code input and map selection in the game lobby.
"""

import re
from typing import TYPE_CHECKING, Optional, Callable, Tuple

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


def wait_for_lobby_ui(
    page: "Page",
    status_callback: Optional[Callable] = None,
    timeout: int = 120000
) -> bool:
    """
    Wait for Fortnite lobby UI to appear.
    
    Args:
        page: Playwright page object
        status_callback: Optional callback for status updates
        timeout: Maximum wait time in ms
        
    Returns:
        True if lobby UI detected
    """
    _emit_status("Ожидаю загрузку лобби", status_callback)
    
    # Lobby detection strategies
    lobby_indicators = [
        # Text-based
        ('text', 'PLAY'),
        ('text', 'BATTLE ROYALE'),
        ('text', 'CREATIVE'),
        ('text', 'SAVE THE WORLD'),
        # Selector-based
        ('selector', 'button:has-text("PLAY")'),
        ('selector', '[data-testid="play-button"]'),
    ]
    
    elapsed = 0
    check_interval = 2000
    
    while elapsed < timeout:
        # Check for lobby indicators
        for indicator_type, indicator in lobby_indicators:
            try:
                if indicator_type == 'text':
                    el = page.get_by_text(re.compile(f'^{indicator}$', re.I)).first
                else:
                    el = page.locator(indicator).first
                    
                if el and el.is_visible(timeout=500):
                    _emit_status("Лобби загружено", status_callback)
                    page.wait_for_timeout(1000)  # Small delay for stability
                    return True
            except Exception:
                continue
                
        page.wait_for_timeout(check_interval)
        elapsed += check_interval
        
        if elapsed % 10000 == 0:
            _emit_status(f"Ожидание лобби... {elapsed // 1000}с", status_callback)
            
    _emit_status("Таймаут ожидания лобби", status_callback)
    return False


def open_search_panel(
    page: "Page",
    status_callback: Optional[Callable] = None
) -> bool:
    """
    Open the island search panel.
    
    Args:
        page: Playwright page object
        status_callback: Optional callback for status updates
        
    Returns:
        True if search panel opened
    """
    _emit_status("Открываю панель поиска", status_callback)
    
    # Search button selectors
    search_selectors = [
        'button[aria-label*="Search"]',
        'button:has-text("Search")',
        '[data-testid="search-button"]',
        'input[placeholder*="Search"]',
        '.search-icon',
    ]
    
    for sel in search_selectors:
        try:
            el = page.locator(sel).first
            if el and el.is_visible(timeout=3000):
                el.click()
                page.wait_for_timeout(500)
                _emit_status("Панель поиска открыта", status_callback)
                return True
        except Exception:
            continue
            
    # Try hotkey
    try:
        page.keyboard.press('/')
        page.wait_for_timeout(500)
        _emit_status("Поиск открыт через хоткей", status_callback)
        return True
    except Exception:
        pass
        
    _emit_status("Не удалось открыть поиск", status_callback)
    return False


def find_search_input(page: "Page") -> Optional[Tuple[int, int]]:
    """
    Find the search input field.
    
    Returns:
        Tuple of (x, y) coordinates or None if not found
    """
    input_selectors = [
        'input[placeholder*="Search for Islands or Creators"]',
        'input[placeholder*="Search for Islands"]',
        'input[placeholder*="Search"]',
        'input[type="search"]',
        'input[type="text"]',
        '[role="textbox"]',
    ]
    
    for sel in input_selectors:
        try:
            el = page.locator(sel).first
            if el and el.is_visible(timeout=2000):
                box = el.bounding_box()
                if box:
                    cx = int(box['x'] + box['width'] / 2)
                    cy = int(box['y'] + box['height'] / 2)
                    return (cx, cy)
        except Exception:
            continue
            
    # Fallback: find largest input in top half of page
    try:
        candidates = page.locator('input, textarea, [role="textbox"]')
        best_el = None
        best_width = 0
        
        count = candidates.count()
        for i in range(min(count, 20)):
            try:
                el = candidates.nth(i)
                box = el.bounding_box()
                if box and box['y'] < 400 and box['width'] > best_width and el.is_visible():
                    best_width = box['width']
                    best_el = el
            except Exception:
                continue
                
        if best_el:
            box = best_el.bounding_box()
            if box:
                return (int(box['x'] + box['width'] / 2), int(box['y'] + box['height'] / 2))
    except Exception:
        pass
        
    return None


def search_and_open_island_dom(
    page: "Page",
    code: str,
    status_callback: Optional[Callable] = None
) -> bool:
    """
    Search for island using DOM selectors.
    
    Args:
        page: Playwright page object
        code: Island code to search for
        status_callback: Optional callback for status updates
        
    Returns:
        True if search successful
    """
    _emit_status(f"Ищу остров: {code}", status_callback)
    
    # Try to find and use search input
    try:
        inp = page.get_by_placeholder(re.compile(r"Search.*Islands|Search.*Creators|Search", re.I)).first
        if inp and inp.is_visible(timeout=3000):
            inp.click()
            inp.fill("")
            inp.type(code, delay=50)
            page.keyboard.press('Enter')
            page.wait_for_timeout(1500)
            return True
    except Exception:
        pass
        
    # Fallback to coordinate-based input
    coords = find_search_input(page)
    if coords:
        try:
            page.mouse.click(coords[0], coords[1])
            page.wait_for_timeout(200)
            page.keyboard.type(code, delay=50)
            page.keyboard.press('Enter')
            page.wait_for_timeout(1500)
            return True
        except Exception:
            pass
            
    _emit_status("Не удалось найти поле поиска", status_callback)
    return False


def select_map_and_play(
    page: "Page",
    status_callback: Optional[Callable] = None
) -> bool:
    """
    Select the found map and click Play.
    
    Args:
        page: Playwright page object
        status_callback: Optional callback for status updates
        
    Returns:
        True if successful
    """
    _emit_status("Выбираю карту и нажимаю Play", status_callback)
    
    # Click SELECT button
    select_selectors = [
        'button:has-text("SELECT")',
        'button:has-text("Select")',
        'button:has-text("Выбрать")',
        '[data-testid="select-button"]',
    ]
    
    for sel in select_selectors:
        try:
            btn = page.locator(sel).first
            if btn and btn.is_visible(timeout=5000):
                btn.click(timeout=10000)
                page.wait_for_timeout(1500)
                break
        except Exception:
            continue
            
    # Click PLAY button
    play_selectors = [
        'button:has-text("PLAY")',
        'button:has-text("Play")',
        'button:has-text("Играть")',
        '[data-testid="play-button"]',
    ]
    
    for sel in play_selectors:
        try:
            btn = page.locator(sel).first
            if btn and btn.is_visible(timeout=5000):
                btn.scroll_into_view_if_needed()
                btn.click(timeout=15000)
                _emit_status("Play нажата", status_callback)
                return True
        except Exception:
            continue
            
    # Fallback: press Enter
    try:
        page.keyboard.press('Enter')
        _emit_status("Enter как fallback", status_callback)
        return True
    except Exception:
        pass
        
    _emit_status("Не удалось нажать Play", status_callback)
    return False


def search_and_launch_island_unified(
    page: "Page",
    code: str,
    status_callback: Optional[Callable] = None,
    use_vision: bool = False
) -> bool:
    """
    Unified island search and launch flow.
    
    This function tries multiple strategies to search for and launch an island:
    1. DOM-based search (most reliable for web UI)
    2. Vision-based search (fallback for canvas-based UI)
    3. Hotkey-based search (last resort)
    
    Args:
        page: Playwright page object
        code: Island code to search for
        status_callback: Optional callback for status updates
        use_vision: Whether to use vision module as fallback
        
    Returns:
        True if island was launched successfully
    """
    _emit_status(f"Запускаю остров: {code}", status_callback)
    
    # Clean up code
    code = code.strip()
    if not code:
        _emit_status("Код острова не указан", status_callback)
        return False
        
    # Strategy 1: Open search and enter code via DOM
    if open_search_panel(page, status_callback):
        page.wait_for_timeout(500)
        if search_and_open_island_dom(page, code, status_callback):
            page.wait_for_timeout(2000)
            if select_map_and_play(page, status_callback):
                return True
                
    # Strategy 2: Direct input (search might already be open)
    _emit_status("Пробую прямой ввод", status_callback)
    try:
        page.keyboard.type(code, delay=50)
        page.keyboard.press('Enter')
        page.wait_for_timeout(2000)
        if select_map_and_play(page, status_callback):
            return True
    except Exception:
        pass
        
    # Strategy 3: Escape and retry
    _emit_status("Перезапускаю поиск", status_callback)
    try:
        page.keyboard.press('Escape')
        page.wait_for_timeout(500)
        page.keyboard.press('Escape')
        page.wait_for_timeout(1000)
    except Exception:
        pass
        
    if open_search_panel(page, status_callback):
        page.wait_for_timeout(500)
        if search_and_open_island_dom(page, code, status_callback):
            page.wait_for_timeout(2000)
            if select_map_and_play(page, status_callback):
                return True
                
    # Strategy 4: Vision-based (optional)
    if use_vision:
        try:
            from ..vision import detection
            _emit_status("Пробую поиск через vision", status_callback)
            # This would use template matching to find UI elements
            # Implementation depends on vision module capabilities
        except Exception as e:
            logger.debug(f"Vision fallback failed: {e}")
            
    _emit_status("Не удалось запустить остров", status_callback)
    return False


def search_and_launch_island_canvas(
    page: "Page",
    code: str,
    vision_module,
    stream_input_module,
    status_callback: Optional[Callable] = None
) -> bool:
    """
    Search and launch island using canvas-based UI with vision.
    
    This is used when the game UI is rendered on canvas and DOM selectors don't work.
    
    Args:
        page: Playwright page object
        code: Island code to search for
        vision_module: Vision module for template matching
        stream_input_module: Stream input module for game input
        status_callback: Optional callback for status updates
        
    Returns:
        True if successful
    """
    _emit_status(f"Canvas поиск острова: {code}", status_callback)
    
    try:
        # Ensure focus on stream
        stream_input_module.ensure_stream_focus(page)
        
        # Step 1: Open search via vision
        _emit_status("Шаг 1: Открываю поиск", status_callback)
        if not stream_input_module.open_search(page):
            # Fallback: hotkey
            page.keyboard.press('/')
            page.wait_for_timeout(200)
            
        # Step 2: Find input field via vision
        _emit_status("Шаг 2: Ищу поле ввода", status_callback)
        input_pos = vision_module.find_image_on_page(
            page,
            'assets/island_code_input_field.png',
            confidence=0.66,
            timeout=3,
            roi=(0.05, 0.08, 0.95, 0.40),
            scales=[0.6, 0.8, 1.0, 1.2],
        )
        
        if not input_pos:
            _emit_status("Поле ввода не найдено", status_callback)
            return False
            
        # Step 3: Click and type code
        _emit_status("Шаг 3: Ввожу код", status_callback)
        page.mouse.click(input_pos[0], input_pos[1])
        page.wait_for_timeout(100)
        page.keyboard.type(code, delay=50)
        page.keyboard.press('Enter')
        page.wait_for_timeout(1500)
        
        # Step 4: Find and click Play button
        _emit_status("Шаг 4: Ищу кнопку Play", status_callback)
        
        # Try yellow play button first
        play_pos = vision_module.find_image_on_page(
            page,
            'assets/play_button_yellow.png',
            confidence=0.70,
            timeout=10,
            scales=[0.75, 0.9, 1.0, 1.2],
        )
        
        if play_pos:
            page.mouse.click(play_pos[0], play_pos[1])
            _emit_status("Play нажата (желтая)", status_callback)
            return True
            
        # Try regular play button
        play_pos = vision_module.find_image_on_page(
            page,
            'assets/play_button.png',
            confidence=0.75,
            timeout=5,
            scales=[0.75, 0.9, 1.0, 1.2],
        )
        
        if play_pos:
            page.mouse.click(play_pos[0], play_pos[1])
            _emit_status("Play нажата", status_callback)
            return True
            
        # Fallback: Enter key
        page.keyboard.press('Enter')
        _emit_status("Enter вместо Play", status_callback)
        return True
        
    except Exception as e:
        logger.error(f"Canvas search failed: {e}")
        _emit_status(f"Ошибка canvas поиска: {e}", status_callback)
        return False


def skip_trailer(page: "Page") -> bool:
    """Skip any intro/trailer video."""
    skip_selectors = [
        'button:has-text("Skip")',
        'button:has-text("SKIP")',
        '[data-testid="skip-button"]',
        '.skip-button',
    ]
    
    for sel in skip_selectors:
        try:
            el = page.locator(sel).first
            if el and el.is_visible(timeout=2000):
                el.click()
                return True
        except Exception:
            continue
            
    # Try pressing Escape or Space
    try:
        page.keyboard.press('Escape')
        page.wait_for_timeout(500)
        page.keyboard.press('Space')
        return True
    except Exception:
        pass
        
    return False
