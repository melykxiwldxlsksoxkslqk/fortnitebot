"""
Модуль для работы с браузером через Playwright + Camoufox.

Camoufox — анти-детект браузер на основе Firefox с улучшенной защитой
от фингерпринтинга и обнаружения ботов.

Использование:
    from src.browser import create_browser, BrowserManager
    
    # Простой запуск
    browser, context, page = create_browser(headless=False)
    
    # Или через менеджер
    with BrowserManager(headless=False) as bm:
        page = bm.page
        page.goto("https://xbox.com/play")
"""

import os
import sys
from typing import Optional, Tuple, Dict, Any, List
from contextlib import contextmanager

# Playwright
try:
    from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Browser = None
    BrowserContext = None
    Page = None

# Camoufox (опционально)
try:
    from camoufox.sync_api import Camoufox
    CAMOUFOX_AVAILABLE = True
except ImportError:
    CAMOUFOX_AVAILABLE = False
    Camoufox = None

from .logger import get_logger
from .config import BROWSER, ROOT_DIR

logger = get_logger(__name__)


# ============================================================================
# КОНФИГУРАЦИЯ БРАУЗЕРА
# ============================================================================

# Путь к профилю браузера
BROWSER_PROFILE_DIR = os.path.join(ROOT_DIR, 'browser-profile')

# Camoufox конфигурация для анти-детекта
CAMOUFOX_CONFIG: Dict[str, Any] = {
    # Геолокация (можно менять под нужный регион)
    'geoip': True,  # Автоопределение по IP
    
    # Человекоподобное поведение
    'humanize': True,
    
    # WebGL спуфинг
    'webgl_config': {
        'renderer': 'ANGLE (Intel, Intel(R) UHD Graphics 620 Direct3D11 vs_5_0 ps_5_0)',
        'vendor': 'Google Inc. (Intel)',
    },
    
    # Локаль и язык
    'locale': 'en-US',
    'timezone': 'America/New_York',
    
    # Отключаем некоторые фичи для стелса
    'block_images': False,
    'block_webrtc': False,  # Xbox Cloud требует WebRTC
}

# Стандартные аргументы для Chromium (fallback)
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


# ============================================================================
# ФУНКЦИИ СОЗДАНИЯ БРАУЗЕРА
# ============================================================================

def create_browser_camoufox(
    headless: bool = False,
    proxy: Optional[Dict[str, str]] = None,
    profile_dir: Optional[str] = None,
) -> Tuple[Any, Any, Any]:
    """
    Создаёт браузер через Camoufox (рекомендуется).
    
    Args:
        headless: Режим без интерфейса
        proxy: Прокси {'server': 'http://...', 'username': '...', 'password': '...'}
        profile_dir: Путь к профилю для сохранения сессии
    
    Returns:
        Tuple[Camoufox, BrowserContext, Page]
    """
    if not CAMOUFOX_AVAILABLE:
        raise ImportError(
            "Camoufox не установлен. Установите: pip install camoufox && camoufox fetch"
        )
    
    logger.info("Запуск Camoufox браузера...")
    
    config = CAMOUFOX_CONFIG.copy()
    
    # Добавляем прокси
    if proxy and proxy.get('server'):
        config['proxy'] = {
            'server': proxy['server'],
        }
        if proxy.get('username'):
            config['proxy']['username'] = proxy['username']
        if proxy.get('password'):
            config['proxy']['password'] = proxy['password']
        logger.debug(f"Прокси: {proxy['server']}")
    
    # Профиль для сохранения cookies/localStorage
    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
        config['persistent_context'] = profile_dir
    
    try:
        # Camoufox сам управляет браузером и контекстом
        cf = Camoufox(
            headless=headless,
            **config
        )
        
        # Получаем страницу
        page = cf.new_page()
        
        # Camoufox не возвращает отдельный browser/context в стандартном виде
        # Возвращаем cf как "browser" для совместимости
        logger.info("Camoufox браузер запущен успешно")
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
) -> Tuple[Browser, BrowserContext, Page]:
    """
    Создаёт браузер через стандартный Playwright (fallback).
    
    Args:
        headless: Режим без интерфейса
        proxy: Прокси-конфигурация
        profile_dir: Путь к профилю
        browser_type: 'chromium', 'firefox', или 'webkit'
        extensions: Список путей к расширениям (только Chromium)
    
    Returns:
        Tuple[Browser, BrowserContext, Page]
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise ImportError("Playwright не установлен. Установите: pip install playwright")
    
    logger.info(f"Запуск Playwright {browser_type}...")
    
    pw = sync_playwright().start()
    
    # Выбор движка
    if browser_type == 'firefox':
        engine = pw.firefox
    elif browser_type == 'webkit':
        engine = pw.webkit
    else:
        engine = pw.chromium
    
    # Параметры запуска
    launch_kwargs: Dict[str, Any] = {
        'headless': headless,
    }
    
    # Аргументы для Chromium
    if browser_type == 'chromium':
        args = CHROMIUM_ARGS.copy()
        
        # Расширения
        if extensions:
            valid_extensions = [e for e in extensions if os.path.exists(e)]
            if valid_extensions:
                ext_paths = ','.join(valid_extensions)
                args.extend([
                    f'--disable-extensions-except={ext_paths}',
                    f'--load-extension={ext_paths}',
                ])
                logger.debug(f"Загружены расширения: {valid_extensions}")
        
        launch_kwargs['args'] = args
    
    # Прокси
    if proxy and proxy.get('server'):
        launch_kwargs['proxy'] = {
            'server': proxy['server'],
        }
        if proxy.get('username'):
            launch_kwargs['proxy']['username'] = proxy['username']
        if proxy.get('password'):
            launch_kwargs['proxy']['password'] = proxy['password']
    
    try:
        # Persistent context для сохранения сессии
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
        
        logger.info(f"Playwright {browser_type} запущен успешно")
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
) -> Tuple[Any, Any, Page]:
    """
    Универсальная функция создания браузера.
    
    Пробует Camoufox (если доступен и prefer_camoufox=True),
    иначе fallback на Playwright.
    
    Args:
        headless: Режим без интерфейса
        proxy: Прокси {'server': 'http://host:port', 'username': '...', 'password': '...'}
        profile_dir: Путь к профилю браузера
        prefer_camoufox: Использовать Camoufox если доступен
        browser_type: Тип браузера для Playwright ('chromium', 'firefox', 'webkit')
        extensions: Расширения (только Chromium)
    
    Returns:
        Tuple[browser, context, page]
    
    Example:
        browser, context, page = create_browser(
            headless=False,
            proxy={'server': 'http://proxy.example.com:8080'},
            prefer_camoufox=True
        )
        page.goto('https://xbox.com/play')
    """
    profile = profile_dir or BROWSER_PROFILE_DIR
    
    # Пробуем Camoufox
    if prefer_camoufox and CAMOUFOX_AVAILABLE:
        try:
            return create_browser_camoufox(
                headless=headless,
                proxy=proxy,
                profile_dir=profile,
            )
        except Exception as e:
            logger.warning(f"Camoufox недоступен, переключаемся на Playwright: {e}")
    
    # Fallback на Playwright
    return create_browser_playwright(
        headless=headless,
        proxy=proxy,
        profile_dir=profile,
        browser_type=browser_type,
        extensions=extensions,
    )


# ============================================================================
# BROWSER MANAGER (КОНТЕКСТНЫЙ МЕНЕДЖЕР)
# ============================================================================

class BrowserManager:
    """
    Менеджер браузера для удобного использования с context manager.
    
    Example:
        with BrowserManager(headless=False) as bm:
            bm.page.goto('https://xbox.com/play')
            # ... работа со страницей
        # браузер автоматически закроется
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
        """Закрывает браузер и освобождает ресурсы."""
        try:
            if self._is_camoufox and self.browser:
                # Camoufox
                try:
                    self.browser.close()
                except Exception:
                    pass
            else:
                # Playwright
                if self.page:
                    try:
                        self.page.close()
                    except Exception:
                        pass
                if self.context:
                    try:
                        self.context.close()
                    except Exception:
                        pass
                if self.browser:
                    try:
                        self.browser.close()
                    except Exception:
                        pass
            
            logger.debug("Браузер закрыт")
        except Exception as e:
            logger.error(f"Ошибка при закрытии браузера: {e}")
    
    def new_page(self) -> Page:
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


# ============================================================================
# УТИЛИТЫ
# ============================================================================

def check_browser_availability() -> Dict[str, bool]:
    """
    Проверяет доступность браузерных движков.
    
    Returns:
        {'playwright': bool, 'camoufox': bool}
    """
    result = {
        'playwright': PLAYWRIGHT_AVAILABLE,
        'camoufox': CAMOUFOX_AVAILABLE,
    }
    
    logger.info(f"Доступность браузеров: {result}")
    return result


def install_camoufox() -> bool:
    """
    Устанавливает Camoufox браузер.
    
    Returns:
        True если успешно
    """
    import subprocess
    
    try:
        logger.info("Установка Camoufox...")
        
        # Установка пакета
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'camoufox'],
            check=True,
            capture_output=True,
        )
        
        # Загрузка браузера
        subprocess.run(
            ['camoufox', 'fetch'],
            check=True,
            capture_output=True,
        )
        
        logger.info("Camoufox установлен успешно")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка установки Camoufox: {e}")
        return False
    except FileNotFoundError:
        logger.error("camoufox CLI не найден. Попробуйте: pip install camoufox")
        return False


def get_recommended_browser() -> str:
    """
    Возвращает рекомендуемый браузер.
    
    Returns:
        'camoufox' или 'playwright'
    """
    if CAMOUFOX_AVAILABLE:
        return 'camoufox'
    return 'playwright'
