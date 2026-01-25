"""
Основная логика работы бота.
"""

import time
import threading
import asyncio
from typing import Optional, Callable, Dict, Any

from ..core import get_logger
from .. import vision

logger = get_logger(__name__)


class BotLogic:
    """
    Класс управления ботом.
    
    Используется GUI для запуска, остановки и мониторинга бота.
    """
    
    def __init__(
        self,
        account: Dict[str, str],
        proxy: Optional[Dict[str, str]],
        config: Dict[str, Any],
        update_status_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Инициализация бота.
        
        Args:
            account: Словарь с 'login' и 'password'
            proxy: Опциональный словарь с настройками прокси
            config: Конфигурация бота
            update_status_callback: Колбэк для обновления статуса в UI
        """
        self.account = account
        self.proxy = proxy
        self.config = config
        self.update_status = update_status_callback
        self.playwright = None
        self.browser = None
        self._runner = None  # Ссылка на BotRunner для принудительного закрытия
        self.stop_requested = False
        self.manual_lobby_event = threading.Event()
        
        self._login = account.get('login', 'unknown')
        logger.info(f"Создан бот для аккаунта: {self._login}")

    def _log(self, message: str) -> None:
        """Логирование с отправкой статуса в UI."""
        try:
            if self.update_status:
                self.update_status(self._login, str(message))
        except Exception as e:
            logger.debug(f"Ошибка отправки статуса: {e}")
        
        logger.info(f"[{self._login}] {message}")

    def signal_lobby_ready(self):
        """Устанавливает ручной флаг: лобби готово."""
        try:
            self.manual_lobby_event.set()
            try:
                vision.set_vision_debug(True)
            except Exception:
                pass
            self._log("Получен сигнал: лобби готово (ручной)")
        except Exception as e:
            logger.error(f"Ошибка при сигнале лобби: {e}")

    def request_stop(self):
        """Запрашивает остановку бота."""
        self.stop_requested = True
        self._log("Получен запрос на остановку...")
        
        # Принудительно закрываем браузер если есть runner
        if self._runner:
            try:
                self._runner._close_browser()
            except Exception:
                pass

    def _is_stop_requested(self) -> bool:
        """Возвращает True если запрошена остановка."""
        return self.stop_requested
    
    def _set_runner(self, runner) -> None:
        """Сохраняет ссылку на runner для принудительного закрытия."""
        self._runner = runner

    async def run(self):
        """Запускает бота."""
        self._log("Бот запущен...")
        try:
            from .runner import run_bot
            from ..core import get_setting, BadCredentialsError, BrowserClosedError

            island_code = self.config.get('island_code') or get_setting('island_code', '1234-5678-9012')
            headless = bool(self.config.get('headless', True))

            try:
                def _forward(msg: str):
                    if self.update_status:
                        self.update_status(self.account.get('login', 'unknown'), msg)
                
                success = await asyncio.to_thread(
                    run_bot, 
                    self.account, 
                    island_code, 
                    headless, 
                    self.proxy, 
                    self.manual_lobby_event, 
                    _forward,
                    self._is_stop_requested,  # передаём функцию проверки остановки
                    self._set_runner  # передаём колбэк для сохранения runner
                )
            except BadCredentialsError:
                if self.update_status:
                    self.update_status(self._login, "Неверный логин/пароль")
                return
            except BrowserClosedError:
                if self.update_status:
                    self.update_status(self._login, "Браузер закрыт пользователем")
                return

            if self.stop_requested:
                self._log("Бот остановлен по запросу.")
                return

            self._log("Бот завершил работу.")
            if self.update_status:
                if success:
                    self.update_status(self._login, "Успех")
                else:
                    self.update_status(self._login, "Не удалось загрузить карту")
        except Exception as e:
            self._log(f"Произошла ошибка: {e}")
            if self.update_status:
                self.update_status(self._login, f"Ошибка: {e.__class__.__name__}")
        finally:
            self._runner = None
            self._log("Бот выключается.")
