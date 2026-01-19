"""
Управление браузером.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Dict, Any, Tuple, List, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from playwright.sync_api import Browser, BrowserContext, Page

# Playwright
try:
    from playwright.sync_api import sync_playwright
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    sync_playwright = None
    PlaywrightTimeoutError = Exception

# Camoufox (опционально)
try:
    from camoufox.sync_api import Camoufox
    CAMOUFOX_AVAILABLE = True
except ImportError:
    CAMOUFOX_AVAILABLE = False
    Camoufox = None

from ..core import get_logger, BROWSER, ROOT_DIR

logger = get_logger(__name__)

# Путь к профилю браузера
BROWSER_PROFILE_DIR = os.path.join(ROOT_DIR, 'browser-profile')

# Camoufox конфигурация
CAMOUFOX_CONFIG: Dict[str, Any] = {
    'geoip': True,
    'humanize': True,
    'webgl_config': {
        'renderer': 'ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0)',
        'vendor': 'Google Inc. (Intel)',
    },
    'locale': 'en-US',
    'timezone': 'America/New_York',
    'block_images': False,
    'block_webrtc': False,
}

# Chromium аргументы
CHROMIUM_ARGS: List[str] = [
    '--start-maximized',
    '--disable-blink-features=AutomationControlled',
    '--disable-infobars',
    '--disable-dev-shm-usage',
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-gpu-sandbox',
    '--disable-web-security',
    '--allow-running-insecure-content',
]


def create_browser_camoufox(
    headless: bool = False,
    proxy: Optional[Dict[str, str]] = None,
    profile_dir: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """Создаёт браузер через Camoufox."""
    if not CAMOUFOX_AVAILABLE:
        raise ImportError("Camoufox не установлен")
    
    logger.info("Запуск Camoufox браузера...")
    
    config = CAMOUFOX_CONFIG.copy()
    
    if proxy and proxy.get('server'):
        config['proxy'] = {'server': proxy['server']}
        if proxy.get('username'):
            config['proxy']['username'] = proxy['username']
        if proxy.get('password'):
            config['proxy']['password'] = proxy['password']
    
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
        config['persistent_context'] = profile_dir
    
    try:
        cf = Camoufox(headless=headless, **config)
        page = cf.new_page()
        logger.info("Camoufox браузер запущен")
        return cf, cf, page
    except Exception as e:
        logger.error(f"Ошибка запуска Camoufox: {e}")
        raise


def create_browser_playwright(
    headless: bool = False,
    proxy: Optional[Dict[str, str]] = None,
    profile_dir: Optional[str] = None,
    browser_type: str = 'chromium',
    extensions: Optional[List[str]] = None,
) -> Tuple["Browser", "BrowserContext", "Page"]:
    """Создаёт браузер через Playwright."""
    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError("Playwright не установлен")
    
    logger.info(f"Запуск Playwright {browser_type}...")
    
    pw = sync_playwright().start()
    
    if browser_type == 'firefox':
        engine = pw.firefox
    elif browser_type == 'webkit':
        engine = pw.webkit
    else:
        engine = pw.chromium
    
    launch_kwargs: Dict[str, Any] = {'headless': headless}
    
    if browser_type == 'chromium':
        args = CHROMIUM_ARGS.copy()
        if extensions:
            valid_extensions = [e for e in extensions if os.path.exists(e)]
            if valid_extensions:
                ext_paths = ','.join(valid_extensions)
                args.extend([
                    f'--disable-extensions-except={ext_paths}',
                    f'--load-extension={ext_paths}',
                ])
        launch_kwargs['args'] = args
    
    if proxy and proxy.get('server'):
        launch_kwargs['proxy'] = {'server': proxy['server']}
        if proxy.get('username'):
            launch_kwargs['proxy']['username'] = proxy['username']
        if proxy.get('password'):
            launch_kwargs['proxy']['password'] = proxy['password']
    
    try:
        if profile_dir:
            os.makedirs(profile_dir, exist_ok=True)
            context = engine.launch_persistent_context(
                user_data_dir=profile_dir,
                **launch_kwargs,
                no_viewport=True,
            )
            browser = context.browser
            page = context.new_page() if not context.pages else context.pages[0]
        else:
            browser = engine.launch(**launch_kwargs)
            context = browser.new_context(no_viewport=True)
            page = context.new_page()
        
        logger.info(f"Playwright {browser_type} запущен")
        return browser, context, page
    except Exception as e:
        logger.error(f"Ошибка запуска Playwright: {e}")
        try:
            pw.stop()
        except Exception:
            pass
        raise


def create_browser(
    headless: bool = False,
    proxy: Optional[Dict[str, str]] = None,
    profile_dir: Optional[str] = None,
    prefer_camoufox: bool = True,
    browser_type: str = 'chromium',
    extensions: Optional[List[str]] = None,
) -> Tuple[Any, Any, "Page"]:
    """
    Универсальная функция создания браузера.
    
    Args:
        headless: Режим без интерфейса
        proxy: Прокси конфигурация
        profile_dir: Путь к профилю
        prefer_camoufox: Предпочитать Camoufox
        browser_type: Тип браузера для Playwright
        extensions: Расширения (только Chromium)
    
    Returns:
        Tuple[browser, context, page]
    """
    profile = profile_dir or BROWSER_PROFILE_DIR
    
    if prefer_camoufox and CAMOUFOX_AVAILABLE:
        try:
            return create_browser_camoufox(headless, proxy, profile)
        except Exception as e:
            logger.warning(f"Camoufox недоступен: {e}")
    
    return create_browser_playwright(headless, proxy, profile, browser_type, extensions)


def close_browser(browser, context=None, page=None) -> None:
    """Закрывает браузер и освобождает ресурсы."""
    try:
        if page:
            try:
                page.close()
            except Exception:
                pass
        if context and context != browser:
            try:
                context.close()
            except Exception:
                pass
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        logger.debug("Браузер закрыт")
    except Exception as e:
        logger.error(f"Ошибка при закрытии браузера: {e}")


class BrowserManager:
    """
    Менеджер браузера для удобного использования.
    
    Example:
        with BrowserManager(headless=False) as bm:
            bm.page.goto('https://xbox.com/play')
    """
    
    def __init__(
        self,
        headless: bool = False,
        proxy: Optional[Dict[str, str]] = None,
        profile_dir: Optional[str] = None,
        prefer_camoufox: bool = True,
        browser_type: str = 'chromium',
    ):
        self.headless = headless
        self.proxy = proxy
        self.profile_dir = profile_dir
        self.prefer_camoufox = prefer_camoufox
        self.browser_type = browser_type
        
        self.browser = None
        self.context = None
        self.page = None
        self._is_camoufox = False
    
    def __enter__(self) -> 'BrowserManager':
        self.browser, self.context, self.page = create_browser(
            headless=self.headless,
            proxy=self.proxy,
            profile_dir=self.profile_dir,
            prefer_camoufox=self.prefer_camoufox,
            browser_type=self.browser_type,
        )
        self._is_camoufox = CAMOUFOX_AVAILABLE and self.prefer_camoufox
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def close(self):
        """Закрывает браузер."""
        close_browser(self.browser, self.context, self.page)
    
    def new_page(self) -> "Page":
        """Создаёт новую страницу."""
        if self._is_camoufox:
            return self.browser.new_page()
        return self.context.new_page()
    
    def goto(self, url: str, **kwargs) -> None:
        """Переход по URL."""
        self.page.goto(url, **kwargs)
    
    def screenshot(self, path: str) -> None:
        """Сохраняет скриншот."""
        self.page.screenshot(path=path)


def check_browser_availability() -> Dict[str, bool]:
    """Проверяет доступность браузеров."""
    return {
        'playwright': PLAYWRIGHT_AVAILABLE,
        'camoufox': CAMOUFOX_AVAILABLE,
    }


def get_recommended_browser() -> str:
    """Возвращает рекомендуемый браузер."""
    return 'camoufox' if CAMOUFOX_AVAILABLE else 'playwright'
