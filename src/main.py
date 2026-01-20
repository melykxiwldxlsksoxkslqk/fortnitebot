"""
Main entry point for the Fortnite bot.

This module provides the main entry point and orchestrates
the bot workflow using the modular microservices architecture.
"""

import os
import threading
from typing import Optional, Dict, Any, List
from playwright.sync_api import sync_playwright, Browser

# Core modules
from .core.config import load_settings, load_accounts, load_island_code
from .core.logger import get_logger
from .core.db import Database
from .core.exceptions import BadCredentialsError, CodeRequiredError

# Bot modules
from .bot.auth import microsoft_login, try_login_flow
from .bot.xbox import (
    open_browser,
    navigate_to_xbox,
    click_play_button,
    click_play_with_retries,
    wait_for_stream_connected,
    handle_cookie_consent,
)
from .bot.island import (
    wait_for_lobby_ui,
    search_and_launch_island_unified,
    skip_trailer,
)
from .bot.ingame import (
    do_active_ingame_actions,
    lock_mouse_into_stream,
    ensure_stream_focus,
)
from .bot.canvas import (
    CanvasNavigator,
    ScreenState,
    create_navigator,
)

# Vision module (optional)
try:
    from .vision import detection as vision
    VISION_AVAILABLE = True
except ImportError:
    vision = None
    VISION_AVAILABLE = False

logger = get_logger(__name__)

# Active browsers registry
_ACTIVE_BROWSERS: Dict[str, Browser] = {}
_ACTIVE_BROWSERS_LOCK = threading.Lock()

# Manual control event (for external control)
manual_lobby_event: Optional[threading.Event] = None


def _emit_status(message: str) -> None:
    """Emit status message via IPC if available."""
    logger.info(message)
    try:
        from .ipc.server import broadcast_status
        broadcast_status(message)
    except Exception:
        pass


def run_bot(
    account: Dict[str, str],
    island_code: str,
    headless: bool = True,
    proxy: Optional[Dict[str, str]] = None,
) -> bool:
    """
    Run the bot for a single account.
    
    Args:
        account: Account credentials dict with 'email' and 'password'
        island_code: Fortnite island code to join
        headless: Whether to run browser headless
        proxy: Optional proxy configuration
        
    Returns:
        True if bot completed successfully
    """
    account_login = account.get('email', 'unknown')
    logger.info(f"Starting bot for account: {account_login}")
    _emit_status(f"Запуск бота для {account_login}")
    
    result_success = False
    browser = None
    
    try:
        with sync_playwright() as playwright:
            # Step 1: Open browser
            _emit_status("Шаг 1: Открываю браузер")
            browser, context, page = open_browser(
                playwright,
                headless=headless,
                proxy=proxy,
            )
            
            # Register browser
            with _ACTIVE_BROWSERS_LOCK:
                _ACTIVE_BROWSERS[account_login] = browser
            
            # Step 2: Navigate to Xbox
            _emit_status("Шаг 2: Перехожу на Xbox Cloud Gaming")
            navigate_to_xbox(page, fortnite_direct=True)
            
            # Handle cookie consent
            handle_cookie_consent(page)
            page.wait_for_timeout(2000)
            
            # Step 3: Check if login required
            if 'login' in page.url.lower() or 'signin' in page.url.lower():
                _emit_status("Шаг 3: Авторизация")
                login_success = try_login_flow(
                    page,
                    account.get('email', ''),
                    account.get('password', ''),
                    status_callback=_emit_status,
                )
                if not login_success:
                    _emit_status("Ошибка авторизации")
                    return False
                    
                # Re-navigate after login
                page.wait_for_timeout(2000)
                navigate_to_xbox(page, fortnite_direct=True)
            else:
                _emit_status("Шаг 3: Авторизация не требуется")
            
            # Step 4: Click Play button
            _emit_status("Шаг 4: Нажимаю Play")
            if not click_play_with_retries(page, status_callback=_emit_status):
                _emit_status("Не удалось нажать Play")
                return False
            
            # Step 5: Wait for stream
            _emit_status("Шаг 5: Ожидаю подключение стрима")
            if not wait_for_stream_connected(page, status_callback=_emit_status):
                _emit_status("Стрим не подключился")
                return False
            
            # Step 6: Skip any intro
            _emit_status("Шаг 6: Пропускаю интро")
            skip_trailer(page)
            page.wait_for_timeout(2000)
            
            # Step 7: Create Canvas Navigator for smart canvas interaction
            _emit_status("Шаг 7: Инициализация Canvas Navigator")
            navigator = create_navigator(page, status_callback=_emit_status)
            
            # Step 8: Ensure focus and wait for lobby
            _emit_status("Шаг 8: Фокус на стриме и ожидание лобби")
            navigator.ensure_focus()
            navigator.wait_for_state(ScreenState.LOBBY, timeout=120000)
            
            # Step 9: Manual wait if enabled
            if manual_lobby_event is not None:
                _emit_status("Ожидание ручной команды 'Лобби готово'")
                try:
                    manual_lobby_event.wait(timeout=600)
                    if manual_lobby_event.is_set():
                        _emit_status("Команда получена")
                        manual_lobby_event.clear()
                except Exception:
                    pass
            
            # Step 10: Search and launch island using Canvas Navigator
            _emit_status(f"Шаг 10: Запуск острова {island_code}")
            if not navigator.search_and_launch_island(island_code):
                _emit_status("Canvas Navigator не смог запустить остров, пробую fallback")
                # Fallback to old method
                if not search_and_launch_island_unified(
                    page,
                    island_code,
                    status_callback=_emit_status,
                ):
                    _emit_status("Не удалось запустить остров")
            
            # Step 11: Wait for game load
            _emit_status("Шаг 11: Ожидаю загрузку игры")
            navigator.wait_for_state(ScreenState.IN_GAME, timeout=180000)
            
            # Step 12: Active gameplay with AFK prevention
            _emit_status("Шаг 12: Активная игра")
            navigator.run_afk_prevention(duration_ms=60000, interval_ms=15000)
            
            _emit_status("Бот завершил работу успешно")
            result_success = True
            
    except BadCredentialsError as e:
        logger.error(f"Bad credentials for {account_login}: {e}")
        _emit_status(f"Неверные учётные данные: {e}")
    except CodeRequiredError as e:
        logger.error(f"Code required for {account_login}: {e}")
        _emit_status(f"Требуется код подтверждения")
    except Exception as e:
        logger.error(f"Error for {account_login}: {e}")
        _emit_status(f"Ошибка: {e}")
    finally:
        # Cleanup
        logger.info(f"Closing browser for {account_login}")
        try:
            with _ACTIVE_BROWSERS_LOCK:
                if _ACTIVE_BROWSERS.get(account_login) is browser:
                    _ACTIVE_BROWSERS.pop(account_login, None)
        except Exception:
            pass
            
        if browser:
            try:
                browser.close()
            except Exception:
                pass
    
    return result_success


def close_all_active_browsers() -> None:
    """Close all active browsers safely."""
    items = []
    try:
        with _ACTIVE_BROWSERS_LOCK:
            items = list(_ACTIVE_BROWSERS.items())
            _ACTIVE_BROWSERS.clear()
    except Exception:
        items = []
        
    for login, browser in items:
        try:
            browser.close()
            logger.info(f"Closed browser for {login}")
        except Exception:
            pass


def main() -> None:
    """Main entry point for the application."""
    logger.info("Starting Fortnite Bot")
    
    # Load configuration
    accounts = load_accounts()
    if not accounts:
        logger.error("No accounts found. Exiting.")
        print("No accounts found. Exiting.")
        return
    
    settings = load_settings()
    island_code = settings.get('island_code') or load_island_code()
    headless = settings.get('headless', True)
    
    # Validate island code
    code_clean = (island_code or "").strip()
    placeholder_codes = {"", "1234-5678-9012", "0000-0000-0000"}
    if not code_clean or code_clean in placeholder_codes:
        logger.error("Island code not set. Bots will not start.")
        print("Код острова не установлен. Боты не будут запущены.")
        return
    
    logger.info(f"Island code: {code_clean}")
    logger.info(f"Headless mode: {headless}")
    logger.info(f"Accounts to process: {len(accounts)}")
    
    # Disable OS input if vision is available
    if VISION_AVAILABLE:
        try:
            vision.set_disable_os_input(True)
        except Exception:
            pass
    
    # Run bot for each account
    for account in accounts:
        try:
            run_bot(account, code_clean, headless=headless)
        except Exception as e:
            logger.error(f"Failed to run bot for account: {e}")
            continue
    
    logger.info("All bots completed")


if __name__ == "__main__":
    main()
