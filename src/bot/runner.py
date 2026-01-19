"""
Запуск и управление ботами.
"""

import time
import threading
from typing import Optional, Dict, Callable, Any

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

from ..core import get_logger, BadCredentialsError, BrowserClosedError
from ..core import init_db, get_settings, get_setting, fetch_accounts
from ..browser import create_browser, close_browser
from .. import vision

logger = get_logger(__name__)


class BotRunner:
    """Запуск бота для одного аккаунта."""
    
    def __init__(
        self,
        account: Dict[str, str],
        island_code: str,
        headless: bool = False,
        proxy: Optional[Dict[str, str]] = None,
        manual_lobby_event: Optional[threading.Event] = None,
        status_callback: Optional[Callable[[str], None]] = None
    ):
        self.account = account
        self.island_code = island_code
        self.headless = headless
        self.proxy = proxy
        self.manual_lobby_event = manual_lobby_event
        self.status_callback = status_callback
        
        self.browser = None
        self.context = None
        self.page = None
        
        self._login = account.get('login', 'unknown')
    
    def _log(self, message: str):
        """Логирует и отправляет статус."""
        logger.info(f"[{self._login}] {message}")
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass
    
    def run(self) -> bool:
        """
        Запускает бота.
        
        Returns:
            True если успешно
        """
        try:
            self._log("Запуск браузера...")
            self.browser, self.context, self.page = create_browser(
                headless=self.headless,
                proxy=self.proxy,
            )
            
            self._log("Переход на Xbox Cloud Gaming...")
            self.page.goto("https://www.xbox.com/play", wait_until="domcontentloaded", timeout=30000)
            
            # Логин
            if not self._handle_login():
                return False
            
            # Ожидание игры
            if not self._wait_for_game():
                return False
            
            # Навигация к острову
            if not self._navigate_to_island():
                return False
            
            # Время на острове
            time_on_island = int(get_setting('time_on_island_min', 15))
            self._log(f"На острове. Ожидание {time_on_island} мин...")
            time.sleep(time_on_island * 60)
            
            self._log("Успешно завершено!")
            return True
            
        except BadCredentialsError:
            self._log("Неверный логин/пароль")
            raise
        except BrowserClosedError:
            self._log("Браузер закрыт")
            raise
        except Exception as e:
            self._log(f"Ошибка: {e}")
            return False
        finally:
            self._cleanup()
    
    def _handle_login(self) -> bool:
        """Обрабатывает логин в Microsoft."""
        try:
            self._log("Проверка авторизации...")
            time.sleep(2)
            
            # Проверяем нужна ли авторизация
            if "login" in self.page.url.lower() or "signin" in self.page.url.lower():
                self._log("Требуется вход...")
                
                # Email
                email_input = self.page.locator('input[type="email"]')
                if email_input.count() > 0:
                    email_input.fill(self.account.get('login', ''))
                    self.page.keyboard.press('Enter')
                    time.sleep(2)
                
                # Password
                pass_input = self.page.locator('input[type="password"]')
                if pass_input.count() > 0:
                    pass_input.fill(self.account.get('password', ''))
                    self.page.keyboard.press('Enter')
                    time.sleep(3)
                
                # Проверяем ошибки
                error_elem = self.page.locator('[id*="error"]')
                if error_elem.count() > 0 and error_elem.is_visible():
                    raise BadCredentialsError("Неверные учетные данные")
            
            return True
        except BadCredentialsError:
            raise
        except Exception as e:
            self._log(f"Ошибка логина: {e}")
            return False
    
    def _wait_for_game(self) -> bool:
        """Ожидает загрузки игры."""
        self._log("Ожидание загрузки игры...")
        
        start = time.time()
        timeout = 120  # 2 минуты
        
        while time.time() - start < timeout:
            try:
                # Проверяем состояние экрана
                state = vision.detect_screen_state(self.page)
                
                if state == vision.ScreenState.LOBBY:
                    self._log("Лобби обнаружено!")
                    return True
                
                if state == vision.ScreenState.IN_GAME:
                    self._log("Игра готова!")
                    return True
                
                # Ручной сигнал
                if self.manual_lobby_event and self.manual_lobby_event.is_set():
                    self._log("Ручное подтверждение лобби")
                    return True
                
                self._log(f"Состояние: {state.name}")
                time.sleep(2)
                
            except Exception:
                time.sleep(1)
        
        self._log("Таймаут ожидания игры")
        return False
    
    def _navigate_to_island(self) -> bool:
        """Навигация к острову."""
        self._log(f"Навигация к острову: {self.island_code}")
        
        try:
            # Creative Mode
            self._log("Поиск Creative Mode...")
            result = vision.wait_for_template(
                self.page, 
                'assets/creative_mode_button.png',
                timeout=30,
                confidence=0.7
            )
            
            if result:
                x, y, w, h = result
                self.page.mouse.click(x + w // 2, y + h // 2)
                time.sleep(2)
            
            # Island Code
            self._log("Поиск Island Code...")
            result = vision.wait_for_template(
                self.page,
                'assets/island_code_button.png',
                timeout=15,
                confidence=0.7
            )
            
            if result:
                x, y, w, h = result
                self.page.mouse.click(x + w // 2, y + h // 2)
                time.sleep(2)
            
            # Ввод кода
            self._log("Ввод кода острова...")
            result = vision.wait_for_template(
                self.page,
                'assets/island_code_input_field.png',
                timeout=15,
                confidence=0.7
            )
            
            if result:
                x, y, w, h = result
                self.page.mouse.click(x + w // 2, y + h // 2)
                time.sleep(0.5)
                self.page.keyboard.type(self.island_code, delay=100)
                self.page.keyboard.press('Enter')
                time.sleep(2)
            
            # Launch Island
            self._log("Запуск острова...")
            result = vision.wait_for_template(
                self.page,
                'assets/launch_island_button.png',
                timeout=15,
                confidence=0.7
            )
            
            if result:
                x, y, w, h = result
                self.page.mouse.click(x + w // 2, y + h // 2)
            
            self._log("Навигация завершена")
            return True
            
        except Exception as e:
            self._log(f"Ошибка навигации: {e}")
            return False
    
    def _cleanup(self):
        """Закрывает ресурсы."""
        close_browser(self.browser, self.context, self.page)


def run_bot(
    account: Dict[str, str],
    island_code: str,
    headless: bool = False,
    proxy: Optional[Dict[str, str]] = None,
    manual_lobby_event: Optional[threading.Event] = None,
    status_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Запускает бота для одного аккаунта.
    
    Args:
        account: Словарь с 'login' и 'password'
        island_code: Код острова
        headless: Режим без интерфейса
        proxy: Прокси конфигурация
        manual_lobby_event: Событие ручного подтверждения лобби
        status_callback: Колбэк для статуса
    
    Returns:
        True если успешно
    """
    runner = BotRunner(
        account=account,
        island_code=island_code,
        headless=headless,
        proxy=proxy,
        manual_lobby_event=manual_lobby_event,
        status_callback=status_callback,
    )
    return runner.run()
