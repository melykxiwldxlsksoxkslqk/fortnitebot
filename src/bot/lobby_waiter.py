"""
Модуль ожидания лобби Fortnite.

Отвечает за ожидание загрузки игры и подтверждение готовности лобби.
Поддерживает две стратегии:
  - MANUAL:    только ручное подтверждение через кнопку «Лобби готово» в UI
  - AUTO:      автоматическая детекция лобби по DOM-признакам (+ ручной override)

По умолчанию используется MANUAL, чтобы бот НЕ начинал навигацию
без явного разрешения пользователя.
"""

import time
from enum import Enum, auto
from typing import Optional, Callable, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from playwright.sync_api import Page

from ..core import get_logger, get_setting

logger = get_logger(__name__)


class LobbyConfirmMode(Enum):
    """Режим подтверждения готовности лобби."""
    MANUAL = auto()   # Только ручная кнопка «Лобби готово»
    AUTO = auto()     # Авто-детекция через DOM (с возможностью ручного override)


class _ScreenState(Enum):
    """Внутренние состояния экрана (DOM-based)."""
    UNKNOWN = "unknown"
    LOADING = "loading"
    STREAM_ACTIVE = "stream_active"
    ERROR = "error"
    POPUP = "popup"
    LOBBY = "lobby"


class LobbyWaiter:
    """
    Ожидание загрузки игры и подтверждение лобби.

    Принцип единственной ответственности (SRP):
    этот класс отвечает ТОЛЬКО за определение момента,
    когда лобби готово к навигации на остров.

    Attributes:
        page:                 Playwright-страница с игровым стримом
        mode:                 Режим подтверждения (MANUAL / AUTO)
        manual_lobby_event:   threading.Event — сигнал от UI-кнопки «Лобби готово»
        stop_check:           Функция проверки запроса на остановку бота
        status_callback:      Колбэк для логирования / отправки статуса в UI
        timeout:              Максимальное время ожидания (сек), по умолчанию 600
    """

    # Сколько последовательных обнаружений лобби нужно для авто-подтверждения
    AUTO_CONFIRM_THRESHOLD = 3

    # Пауза между проверками DOM (сек)
    POLL_INTERVAL = 2.0

    # Интервал логирования URL для отладки (сек)
    URL_LOG_INTERVAL = 30.0

    # Максимальное время ожидания появления стрима (сек)
    STREAM_WAIT_FALLBACK = 90

    def __init__(
        self,
        page: "Page",
        mode: LobbyConfirmMode = LobbyConfirmMode.MANUAL,
        manual_lobby_event: Optional[threading.Event] = None,
        stop_check: Optional[Callable[[], bool]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        timeout: int = 600,
        login: str = "unknown",
    ):
        self._page = page
        self._mode = mode
        self._manual_event = manual_lobby_event
        self._stop_check = stop_check or (lambda: False)
        self._log_fn = status_callback
        self._timeout = timeout
        self._login = login

        # Внутренние счётчики
        self._lobby_detected_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wait(self) -> bool:
        """
        Блокирующее ожидание готовности лобби.

        Returns:
            True — лобби подтверждено (ручной сигнал или авто-детекция).
            False — таймаут или запрошена остановка.
        """
        self._log("Ожидание загрузки игры и подтверждения лобби...")

        if self._mode == LobbyConfirmMode.MANUAL:
            self._log("Режим: РУЧНОЙ. Нажмите кнопку «Лобби готово» в UI.")
        else:
            self._log("Режим: АВТО. Бот сам определит лобби (или нажмите «Лобби готово»).")

        # Полноэкранный режим
        self._press_fullscreen()

        start = time.time()
        last_state = None
        last_log_time = 0.0
        last_url_log_time = 0.0
        last_stream_log_time = 0.0
        stream_wait_start = 0.0
        controller_checks = 0

        self._log("Нажмите кнопку «Лобби готово» в UI когда персонаж появится в лобби")
        if self._mode == LobbyConfirmMode.AUTO:
            self._log("(Также бот автоматически обнаружит лобби по состоянию экрана)")

        while time.time() - start < self._timeout:
            # 1. Остановка
            if self._should_stop():
                self._log("Остановлен во время ожидания лобби")
                return False

            # 2. Ручной сигнал (приоритет при любом режиме)
            if self._check_manual_signal():
                self._log("Получен сигнал: ЛОББИ ГОТОВО! (ручной)")
                return True

            # 3. Диалоги контроллера / fullscreen (первые 30 сек)
            if time.time() - start < 30 and controller_checks < 10:
                self._handle_popups()
                controller_checks += 1

            # 4. Проверяем URL (не закрылась ли игра)
            url_ok = self._check_url(last_url_log_time, start)
            if url_ok is False:
                # Игра закрылась — пробуем перезапустить
                self._try_restart_game()
                stream_wait_start = 0.0
                controller_checks = 0
                continue
            if isinstance(url_ok, float):
                last_url_log_time = url_ok

            # 5. Проверяем наличие стрима
            if not self._has_game_stream():
                if stream_wait_start == 0.0:
                    stream_wait_start = time.time()
                elapsed_no_stream = time.time() - stream_wait_start
                if elapsed_no_stream < self.STREAM_WAIT_FALLBACK:
                    now = time.time()
                    if now - last_stream_log_time > 15:
                        self._log(f"Ожидание запуска игрового стрима... ({int(elapsed_no_stream)}с)")
                        last_stream_log_time = now
                    time.sleep(self.POLL_INTERVAL)
                    continue
                else:
                    if elapsed_no_stream < self.STREAM_WAIT_FALLBACK + 5:
                        self._log("⚠️ Стрим не обнаружен >90 сек, продолжаю ожидание...")
            else:
                stream_wait_start = 0.0

            # 6. DOM: определяем состояние экрана
            try:
                state = self._detect_screen_state()
            except Exception as e:
                self._log(f"Ошибка детекции DOM: {e}")
                time.sleep(1)
                continue

            # Логируем при смене или каждые 10 сек
            now = time.time()
            if state != last_state or now - last_log_time > 10:
                self._log(f"Состояние экрана: {state.name}")
                last_state = state
                last_log_time = now

            # 7. Обработка конкретных состояний
            handled = self._handle_dom_state(state)
            if handled is True:
                # Авто-подтверждение сработало (только в AUTO режиме)
                return True

            time.sleep(self.POLL_INTERVAL)

        self._log(f"Таймаут ожидания лобби ({self._timeout // 60} мин)")
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _log(self, message: str) -> None:
        logger.info(f"[{self._login}] {message}")
        if self._log_fn:
            try:
                self._log_fn(message)
            except Exception:
                pass

    def _should_stop(self) -> bool:
        try:
            return self._stop_check()
        except Exception:
            return False

    def _check_manual_signal(self) -> bool:
        """Проверяет, был ли нажат ручной сигнал «Лобби готово»."""
        if self._manual_event and self._manual_event.is_set():
            self._manual_event.clear()
            return True
        return False

    def _press_fullscreen(self) -> None:
        try:
            self._page.keyboard.press('F11')
            self._log("Нажат F11 для полноэкранного режима")
        except Exception:
            pass

    def _has_game_stream(self) -> bool:
        """Проверяет наличие видимого игрового стрима на странице."""
        try:
            game_stream = self._page.locator(
                'video, '
                'canvas#game-stream, '
                'canvas[class*="game"], '
                'canvas[class*="stream"], '
                'div[class*="stream"] canvas, '
                'div[class*="stream"] video, '
                'div[class*="GameStreamContainer"], '
                'div[data-testid*="stream"], '
                'iframe[src*="stream"]'
            )
            return game_stream.count() > 0 and game_stream.first.is_visible(timeout=500)
        except Exception:
            return False

    def _check_url(self, last_url_log_time: float, start_time: float):
        """
        Проверяет URL — не закрылась ли игра.

        Returns:
            False  — игра закрылась (нужен перезапуск)
            float  — обновлённый last_url_log_time
            None   — всё OK, URL не нужно логировать
        """
        try:
            current_url = self._page.url.lower()
            now = time.time()

            if now - last_url_log_time > self.URL_LOG_INTERVAL:
                self._log(f"[DEBUG] URL: {current_url[:80]}")
                last_url_log_time = now

            is_game_active = (
                "fortnite" in current_url
                or "/games/" in current_url
                or "/launch/" in current_url
            )
            is_main_page = (
                "xbox.com" in current_url
                and "/play" in current_url
                and not is_game_active
            )
            if is_main_page:
                self._log("ВНИМАНИЕ: Игра закрылась! Xbox вернул на главную страницу.")
                return False

            return last_url_log_time
        except Exception:
            return None

    def _try_restart_game(self) -> None:
        """Пытается перезапустить Fortnite после закрытия."""
        try:
            self._log("Пробуем перезапустить Fortnite...")
            self._page.goto(
                "https://www.xbox.com/play/games/fortnite/BT5P2X999VH2",
                wait_until="domcontentloaded",
                timeout=30000,
            )
            time.sleep(3)
            play_btn = self._page.locator(
                'button:has-text("Play"), button:has-text("Hrát"), button:has-text("Играть")'
            ).first
            if play_btn.is_visible():
                self._log("Нажимаем Play для перезапуска...")
                play_btn.click(force=True)
                time.sleep(2)
        except Exception:
            pass

    def _handle_popups(self) -> None:
        """Закрывает диалог контроллера / fullscreen popup."""
        # Делегируем вызывающему коду — мы только уведомляем
        # (в BotRunner это вызывается из _handle_controller_dialog / _handle_fullscreen_popup)
        pass

    def _detect_screen_state(self) -> _ScreenState:
        """
        Определяет состояние экрана по DOM-элементам.

        Returns:
            _ScreenState — текущее состояние.
        """
        try:
            # Ошибки / диалоги
            error_sel = (
                'div[class*="error" i], '
                'div[class*="Error"], '
                'div[role="alert"], '
                '[data-testid*="error" i]'
            )
            err = self._page.locator(error_sel)
            if err.count() > 0 and err.first.is_visible(timeout=300):
                return _ScreenState.ERROR

            # Popup / модальные диалоги
            popup_sel = (
                'div[role="dialog"], '
                'div[class*="modal" i], '
                'div[class*="popup" i], '
                'div[class*="overlay" i][class*="dialog" i]'
            )
            pop = self._page.locator(popup_sel)
            if pop.count() > 0 and pop.first.is_visible(timeout=300):
                return _ScreenState.POPUP

            # Стрим активен — считаем как лобби/игра
            if self._has_game_stream():
                return _ScreenState.STREAM_ACTIVE

        except Exception:
            pass

        return _ScreenState.UNKNOWN

    def _handle_dom_state(self, state: _ScreenState) -> Optional[bool]:
        """
        Обрабатывает определённое DOM-состояние.

        Returns:
            True  — лобби подтверждено (авто-детекция)
            None  — продолжаем ждать
        """
        # Popup — закрываем, НЕ сбрасываем счётчик
        if state == _ScreenState.POPUP:
            self._log("Popup обнаружен — закрываем через B...")
            self._press_b()
            return None

        # Error — пробуем закрыть
        if state == _ScreenState.ERROR:
            self._log("Ошибка — пробуем закрыть...")
            self._press_b()
            return None

        # STREAM_ACTIVE — стрим виден, считаем как лобби
        if state == _ScreenState.STREAM_ACTIVE:
            if self._mode == LobbyConfirmMode.AUTO:
                self._lobby_detected_count += 1
                self._log(
                    f"Стрим активен — проверка "
                    f"{self._lobby_detected_count}/{self.AUTO_CONFIRM_THRESHOLD}"
                )
                if self._lobby_detected_count >= self.AUTO_CONFIRM_THRESHOLD:
                    self._log(
                        f"ЛОББИ ПОДТВЕРЖДЕНО автоматически! "
                        f"(стрим стабильно виден {self.AUTO_CONFIRM_THRESHOLD} раз)"
                    )
                    return True
            else:
                # В MANUAL режиме — только информируем, НЕ подтверждаем
                if self._lobby_detected_count == 0:
                    self._log(
                        "Стрим обнаружен. "
                        "Нажмите «Лобби готово» для продолжения."
                    )
                self._lobby_detected_count += 1
            return None

        # Любое другое состояние — сброс счётчика
        if self._lobby_detected_count > 0:
            self._log(f"Состояние изменилось на {state.value}, сброс счётчика лобби")
        self._lobby_detected_count = 0
        return None

    # --- Gamepad shortcuts (минимальные, без зависимости на полный VirtualGamepad) ---

    def _press_a(self) -> None:
        """Нажимает A (Enter) через клавиатуру как fallback."""
        try:
            self._page.keyboard.press("Enter")
        except Exception:
            pass

    def _press_b(self) -> None:
        """Нажимает B (Escape) через клавиатуру как fallback."""
        try:
            self._page.keyboard.press("Escape")
        except Exception:
            pass
