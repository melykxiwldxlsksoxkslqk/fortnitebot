"""
Xbox Cloud Gaming navigation module.
Handles browser setup, Xbox navigation, and Play button clicking.
"""

import os
import re
from typing import TYPE_CHECKING, Optional, Callable, Dict, Any

if TYPE_CHECKING:
    from playwright.sync_api import Page, Browser, BrowserContext, Playwright

from ..core.config import load_settings
from ..core.logger import get_logger

logger = get_logger(__name__)

# Xbox Cloud Gaming URLs
XBOX_CLOUD_URL = "https://www.xbox.com/play"
FORTNITE_LAUNCH_URL = "https://www.xbox.com/play/launch/fortnite/BT5P2X999VH2"

# Extensions directory
EXTENSIONS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "extensions")


def _emit_status(message: str, status_callback: Optional[Callable] = None) -> None:
    """Emit status message if callback is provided."""
    logger.info(message)
    if status_callback:
        try:
            status_callback(message)
        except Exception:
            pass


def get_chrome_extensions() -> list:
    """Get list of extension paths to load."""
    extensions = []
    if os.path.exists(EXTENSIONS_DIR):
        for item in os.listdir(EXTENSIONS_DIR):
            ext_path = os.path.join(EXTENSIONS_DIR, item)
            if os.path.isdir(ext_path):
                manifest = os.path.join(ext_path, "manifest.json")
                if os.path.exists(manifest):
                    extensions.append(ext_path)
    return extensions


def open_browser(
    playwright: "Playwright",
    headless: bool = True,
    proxy: Optional[Dict[str, str]] = None,
    user_data_dir: Optional[str] = None,
    status_callback: Optional[Callable] = None
) -> tuple:
    """
    Open a browser with Xbox Cloud Gaming configuration.
    
    Args:
        playwright: Playwright instance
        headless: Whether to run headless
        proxy: Optional proxy configuration
        user_data_dir: Optional user data directory for persistent context
        status_callback: Optional callback for status updates
        
    Returns:
        Tuple of (browser, context, page)
    """
    _emit_status("Открываю браузер", status_callback)
    
    settings = load_settings()
    
    # Browser arguments
    args = [
        "--disable-blink-features=AutomationControlled",
        "--disable-infobars",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-popup-blocking",
        "--disable-translate",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-features=TranslateUI",
        "--lang=en-US",
    ]
    
    # Add extensions if available
    extensions = get_chrome_extensions()
    if extensions and not headless:
        args.append(f"--disable-extensions-except={','.join(extensions)}")
        for ext in extensions:
            args.append(f"--load-extension={ext}")
    
    # Viewport settings
    viewport = {
        "width": settings.get("viewport_width", 1280),
        "height": settings.get("viewport_height", 720),
    }
    
    # Launch options
    launch_opts = {
        "headless": headless,
        "args": args,
    }
    
    if proxy:
        launch_opts["proxy"] = proxy
        
    # Use persistent context if user_data_dir provided
    if user_data_dir:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir,
            **launch_opts,
            viewport=viewport,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        browser = None
        page = context.pages[0] if context.pages else context.new_page()
    else:
        browser = playwright.chromium.launch(**launch_opts)
        context = browser.new_context(
            viewport=viewport,
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        )
        page = context.new_page()
    
    _emit_status("Браузер открыт", status_callback)
    return browser, context, page


def navigate_to_xbox(
    page: "Page",
    fortnite_direct: bool = True,
    status_callback: Optional[Callable] = None
) -> bool:
    """
    Navigate to Xbox Cloud Gaming.
    
    Args:
        page: Playwright page object
        fortnite_direct: If True, navigate directly to Fortnite launch page
        status_callback: Optional callback for status updates
        
    Returns:
        True if navigation successful
    """
    url = FORTNITE_LAUNCH_URL if fortnite_direct else XBOX_CLOUD_URL
    _emit_status(f"Перехожу на Xbox Cloud Gaming", status_callback)
    
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(2000)
        
        # Check if we landed on login page
        current_url = page.url.lower()
        if 'login' in current_url or 'signin' in current_url:
            _emit_status("Требуется авторизация", status_callback)
            return False
            
        _emit_status("Навигация на Xbox выполнена", status_callback)
        return True
        
    except Exception as e:
        logger.error(f"Navigation error: {e}")
        _emit_status(f"Ошибка навигации: {e}", status_callback)
        return False


def click_play_button(
    page: "Page",
    status_callback: Optional[Callable] = None,
    max_retries: int = 5
) -> bool:
    """
    Click the Play button on Xbox Cloud Gaming.
    
    Args:
        page: Playwright page object
        status_callback: Optional callback for status updates
        max_retries: Maximum number of click attempts
        
    Returns:
        True if Play button was clicked successfully
    """
    _emit_status("Ищу кнопку Play", status_callback)
    
    play_selectors = [
        # Primary selectors
        'button[aria-label="Play"]',
        'button[aria-label="Play game"]',
        'button:has-text("Play")',
        'button:has-text("PLAY")',
        '[data-testid="play-button"]',
        # Secondary selectors
        'button.play-button',
        '.PlayButton',
        'a[href*="play/launch"]',
        # Fortnite specific
        'button[aria-label*="Fortnite"]',
    ]
    
    for attempt in range(max_retries):
        logger.info(f"Play button click attempt {attempt + 1}/{max_retries}")
        
        for sel in play_selectors:
            try:
                el = page.locator(sel).first
                if el and el.is_visible(timeout=3000):
                    el.scroll_into_view_if_needed()
                    page.wait_for_timeout(500)
                    el.click(timeout=5000)
                    _emit_status("Нажата кнопка Play", status_callback)
                    page.wait_for_timeout(2000)
                    return True
            except Exception as e:
                logger.debug(f"Selector {sel} failed: {e}")
                continue
                
        # Try keyboard shortcut
        try:
            page.keyboard.press('Enter')
            page.wait_for_timeout(1000)
        except Exception:
            pass
            
        page.wait_for_timeout(2000)
        
    _emit_status("Не удалось найти кнопку Play", status_callback)
    return False


def click_play_with_retries(
    page: "Page",
    status_callback: Optional[Callable] = None,
    max_retries: int = 3,
    retry_delay: int = 5000
) -> bool:
    """
    Click Play button with retries and page refresh.
    
    Args:
        page: Playwright page object
        status_callback: Optional callback for status updates
        max_retries: Maximum number of retry cycles
        retry_delay: Delay between retries in ms
        
    Returns:
        True if successful
    """
    for cycle in range(max_retries):
        if click_play_button(page, status_callback):
            return True
            
        _emit_status(f"Попытка {cycle + 1} не удалась, обновляю страницу", status_callback)
        page.wait_for_timeout(retry_delay)
        
        try:
            page.reload(wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(3000)
        except Exception as e:
            logger.error(f"Reload failed: {e}")
            
    return False


def wait_for_stream_connected(
    page: "Page",
    status_callback: Optional[Callable] = None,
    timeout: int = 120000
) -> bool:
    """
    Wait for the game stream to connect.
    
    Args:
        page: Playwright page object
        status_callback: Optional callback for status updates
        timeout: Maximum wait time in ms
        
    Returns:
        True if stream connected
    """
    _emit_status("Жду подключения к стриму", status_callback)
    
    stream_selectors = [
        'video',
        'canvas#StreamCanvas',
        '[data-testid="stream-video"]',
        '.stream-container video',
        '#game-stream',
    ]
    
    elapsed = 0
    check_interval = 2000
    
    while elapsed < timeout:
        for sel in stream_selectors:
            try:
                el = page.locator(sel).first
                if el and el.is_visible(timeout=1000):
                    # Check if video is playing
                    try:
                        is_playing = page.evaluate(f'''
                            (() => {{
                                const v = document.querySelector('{sel}');
                                return v && !v.paused && v.readyState >= 2;
                            }})()
                        ''')
                        if is_playing:
                            _emit_status("Стрим подключен", status_callback)
                            return True
                    except Exception:
                        # For canvas, just check visibility
                        if 'canvas' in sel.lower():
                            _emit_status("Стрим подключен (canvas)", status_callback)
                            return True
            except Exception:
                continue
                
        page.wait_for_timeout(check_interval)
        elapsed += check_interval
        
        # Status update every 10 seconds
        if elapsed % 10000 == 0:
            _emit_status(f"Ожидание стрима... {elapsed // 1000}с", status_callback)
            
    _emit_status("Таймаут ожидания стрима", status_callback)
    return False


def keep_stream_open(
    page: "Page",
    duration: int = 0,
    status_callback: Optional[Callable] = None
) -> None:
    """
    Keep the stream open by preventing timeout.
    
    Args:
        page: Playwright page object
        duration: How long to keep open in ms (0 = indefinitely)
        status_callback: Optional callback for status updates
    """
    _emit_status("Поддерживаю стрим активным", status_callback)
    
    elapsed = 0
    keep_alive_interval = 30000  # 30 seconds
    
    while duration == 0 or elapsed < duration:
        try:
            # Move mouse slightly to prevent idle timeout
            page.mouse.move(640, 360)
            page.wait_for_timeout(100)
            page.mouse.move(641, 361)
            
            # Press a neutral key
            page.keyboard.press('F15')  # Usually doesn't do anything in games
        except Exception as e:
            logger.debug(f"Keep alive action failed: {e}")
            
        page.wait_for_timeout(keep_alive_interval)
        elapsed += keep_alive_interval


def handle_cookie_consent(page: "Page") -> bool:
    """Handle cookie consent dialog if present."""
    consent_selectors = [
        'button:has-text("Accept")',
        'button:has-text("Accept all")',
        'button:has-text("I accept")',
        'button#onetrust-accept-btn-handler',
        '[data-testid="cookie-accept"]',
    ]
    
    for sel in consent_selectors:
        try:
            el = page.locator(sel).first
            if el and el.is_visible(timeout=2000):
                el.click()
                page.wait_for_timeout(1000)
                return True
        except Exception:
            continue
            
    return False
