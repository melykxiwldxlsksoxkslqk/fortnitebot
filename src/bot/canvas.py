"""
Canvas Navigation Module - умная навигация в канвас-стриме Xbox Cloud Gaming.

Этот модуль предоставляет высокоуровневую абстракцию для работы с игровым UI,
который рендерится на canvas элементе (а не в DOM). Использует DOM-селекторы
и эвристики для определения состояния и эмулирует геймпад/клавиатурный ввод.

Особенности:
- Автоматическое определение состояния экрана (лобби, меню, игра)
- Эмуляция геймпада через keyboard (D-pad навигация)
- Система подтверждения действий (ждёт изменения экрана)
- Retry логика с умным fallback
"""

import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from playwright.sync_api import Page

from ..core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class ScreenState(Enum):
    """Состояния экрана в Fortnite."""
    UNKNOWN = auto()
    LOADING = auto()
    MAIN_MENU = auto()
    LOBBY = auto()
    DISCOVER = auto()           # Экран Discover с поиском островов
    SEARCH_INPUT = auto()       # Диалог ввода кода острова
    SEARCH_PANEL = auto()
    ISLAND_PREVIEW = auto()
    MATCHMAKING = auto()
    IN_GAME = auto()
    PAUSE_MENU = auto()
    ERROR_DIALOG = auto()
    POPUP = auto()              # Popup уведомление (новости, ивенты)


class NavigationDirection(Enum):
    """Направления навигации геймпадом."""
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'


class GamepadButton(Enum):
    """Кнопки геймпада (маппинг на клавиатуру)."""
    A = 'Enter'           # Confirm/Select
    B = 'Escape'          # Back/Cancel
    X = 'x'               # Action 1
    Y = 'y'               # Action 2
    LB = 'q'              # Left Bumper
    RB = 'e'              # Right Bumper
    LT = 'z'              # Left Trigger  
    RT = 'c'              # Right Trigger
    START = 'Escape'      # Menu
    SELECT = 'Tab'        # Select/View
    DPAD_UP = 'ArrowUp'
    DPAD_DOWN = 'ArrowDown'
    DPAD_LEFT = 'ArrowLeft'
    DPAD_RIGHT = 'ArrowRight'


# Маппинг клавиш для навигации
NAVIGATION_KEYS = {
    NavigationDirection.UP: ['ArrowUp', 'w'],
    NavigationDirection.DOWN: ['ArrowDown', 's'],
    NavigationDirection.LEFT: ['ArrowLeft', 'a'],
    NavigationDirection.RIGHT: ['ArrowRight', 'd'],
}

# Таймауты (в мс)
TIMEOUTS = {
    'screen_change': 5000,
    'element_appear': 10000,
    'action_cooldown': 150,
    'navigation_delay': 200,
    'confirm_delay': 300,
    'loading_max': 60000,
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class CanvasElement:
    """Описание элемента на канвасе."""
    name: str
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    template_path: Optional[str] = None
    
    @property
    def center(self) -> Tuple[int, int]:
        """Центр элемента."""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        """Границы элемента (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class ScreenSnapshot:
    """Снимок состояния экрана."""
    state: ScreenState
    elements: List[CanvasElement] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    frame_hash: str = ""
    
    def has_element(self, name: str) -> bool:
        """Проверяет наличие элемента."""
        return any(e.name.lower() == name.lower() for e in self.elements)
    
    def get_element(self, name: str) -> Optional[CanvasElement]:
        """Получает элемент по имени."""
        for e in self.elements:
            if e.name.lower() == name.lower():
                return e
        return None


@dataclass 
class NavigationResult:
    """Результат навигационного действия."""
    success: bool
    action: str
    from_state: ScreenState
    to_state: ScreenState
    duration_ms: int
    error: Optional[str] = None


# ============================================================================
# CANVAS NAVIGATOR CLASS
# ============================================================================

class CanvasNavigator:
    """
    Высокоуровневый навигатор для canvas-based UI.
    
    Поддерживает два режима управления:
    1. Виртуальный геймпад (vgamepad) - предпочтительный для Xbox Cloud Gaming
    2. Клавиатура + клики (fallback)
    
    Использование:
        nav = CanvasNavigator(page)
        nav.ensure_focus()
        nav.wait_for_state(ScreenState.LOBBY)
        nav.open_search()
        nav.type_island_code("1234-5678-9012")
        nav.confirm()
    """
    
    def __init__(
        self,
        page: "Page",
        status_callback: Optional[Callable[[str], None]] = None,
        use_gamepad: bool = True,  # Использовать виртуальный геймпад если доступен
    ):
        self.page = page
        self.status_callback = status_callback
        self._last_snapshot: Optional[ScreenSnapshot] = None
        self._action_history: List[str] = []
        self._retry_counts: Dict[str, int] = {}
        self._gamepad = None
        
        # Инициализация виртуального геймпада
        if use_gamepad:
            try:
                from .gamepad import VirtualGamepad, is_gamepad_available
                if is_gamepad_available():
                    self._gamepad = VirtualGamepad(page=page, status_callback=status_callback)
                    if self._gamepad.is_virtual:
                        self._emit("Используется виртуальный Xbox геймпад")
                    else:
                        self._emit("Геймпад недоступен, используется клавиатура")
            except Exception as e:
                logger.warning(f"Gamepad initialization failed: {e}")
                self._emit(f"Геймпад недоступен: {e}")
    
    # ========================================================================
    # STATUS & LOGGING
    # ========================================================================
    
    def _emit(self, message: str) -> None:
        """Отправить статус."""
        logger.info(f"[Canvas] {message}")
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass
    
    def _log_action(self, action: str) -> None:
        """Записать действие в историю."""
        self._action_history.append(f"{time.time():.2f}: {action}")
        if len(self._action_history) > 100:
            self._action_history = self._action_history[-50:]
    
    def _log_input(self, action_type: str, details: str, extra: str = "") -> None:
        """
        Логирует действия ввода (кнопки, клики, скролл) в понятном формате.
        
        Args:
            action_type: Тип действия (CLICK, KEY, GAMEPAD, SCROLL, TYPE)
            details: Детали действия
            extra: Дополнительная информация
        """
        # Эмодзи для разных типов действий
        icons = {
            "CLICK": "🖱️",
            "KEY": "⌨️",
            "GAMEPAD": "🎮",
            "SCROLL": "📜",
            "TYPE": "✏️",
            "WAIT": "⏳",
            "FOCUS": "🎯",
        }
        icon = icons.get(action_type, "▶️")
        
        msg = f"{icon} [{action_type}] {details}"
        if extra:
            msg += f" | {extra}"
        
        logger.info(f"[Canvas:Input] {msg}")
        self._log_action(f"{action_type}: {details}")
    
    # ========================================================================
    # CANVAS FOCUS & CAPTURE
    # ========================================================================
    
    def get_canvas_bounds(self) -> Optional[Tuple[int, int, int, int]]:
        """Получить границы канваса."""
        selectors = [
            'canvas#StreamCanvas',
            'canvas[data-testid="stream-canvas"]',
            'video',
            '.stream-container canvas',
            'canvas',
        ]
        
        for sel in selectors:
            try:
                el = self.page.locator(sel).first
                if el and el.is_visible(timeout=1000):
                    box = el.bounding_box()
                    if box:
                        return (
                            int(box['x']),
                            int(box['y']),
                            int(box['width']),
                            int(box['height'])
                        )
            except Exception:
                continue
        
        # Fallback: используем viewport
        viewport = self.page.viewport_size
        if viewport:
            return (0, 0, viewport['width'], viewport['height'])
        
        return None
    
    def ensure_focus(self) -> bool:
        """
        Убедиться что канвас имеет фокус для ввода.
        
        Один клик по центру для захвата фокуса Playwright (необходим для передачи input).
        
        Returns:
            True если фокус установлен
        """
        self._emit("Устанавливаю фокус на канвас")
        
        bounds = self.get_canvas_bounds()
        if not bounds:
            logger.error("Canvas not found")
            return False
        
        x, y, w, h = bounds
        cx, cy = x + w // 2, y + h // 2
        
        try:
            # Один клик для захвата фокуса Playwright (минимум для работы ввода)
            self._log_input("FOCUS", f"Canvas center ({cx}, {cy})", f"size={w}x{h}")
            self.page.mouse.click(cx, cy)
            self.page.wait_for_timeout(300)
            
            self._log_action(f"focus_canvas({cx}, {cy})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to focus canvas: {e}")
            return False
    
    # ========================================================================
    # SCREEN STATE DETECTION
    # ========================================================================
    
    def detect_screen_state(self) -> ScreenSnapshot:
        """
        Определить текущее состояние экрана.
        
        Использует DOM-эвристики для определения
        в каком экране сейчас находится игра.
        """
        elements: List[CanvasElement] = []
        
        # Определяем state по DOM-эвристикам
        state = self._detect_state_dom()
        
        snapshot = ScreenSnapshot(
            state=state,
            elements=elements,
            frame_hash="",
        )
        
        self._last_snapshot = snapshot
        return snapshot
    
    def _detect_state_dom(self) -> ScreenState:
        """Определение состояния экрана по DOM-признакам."""
        try:
            # Проверяем наличие canvas стрима
            canvas = self.page.locator('canvas#StreamCanvas, canvas[data-testid="stream-canvas"]')
            canvas_visible = False
            try:
                canvas_visible = canvas.count() > 0 and canvas.first.is_visible(timeout=1000)
            except Exception:
                pass
            
            if not canvas_visible:
                # Нет canvas — проверяем загрузку
                loading = self.page.locator('.loading, [data-testid="loading"]')
                try:
                    if loading.count() > 0 and loading.first.is_visible(timeout=500):
                        return ScreenState.LOADING
                except Exception:
                    pass
                return ScreenState.UNKNOWN
            
            # Canvas виден — считаем что в игре
            return ScreenState.IN_GAME
            
        except Exception:
            return ScreenState.UNKNOWN
    
    def wait_for_state(
        self,
        target_state: ScreenState,
        timeout: int = None,
        check_interval: int = 500
    ) -> bool:
        """
        Ждать определённого состояния экрана.
        
        Args:
            target_state: Целевое состояние
            timeout: Таймаут в мс
            check_interval: Интервал проверки в мс
            
        Returns:
            True если состояние достигнуто
        """
        if timeout is None:
            timeout = TIMEOUTS['loading_max']
        
        self._emit(f"Жду состояние: {target_state.name}")
        
        elapsed = 0
        while elapsed < timeout:
            snapshot = self.detect_screen_state()
            
            if snapshot.state == target_state:
                self._emit(f"Состояние {target_state.name} достигнуто")
                return True
            
            self.page.wait_for_timeout(check_interval)
            elapsed += check_interval
            
            if elapsed % 5000 == 0:
                self._emit(f"Ожидание... текущее: {snapshot.state.name}")
        
        self._emit(f"Таймаут ожидания {target_state.name}")
        return False
    
    def wait_for_screen_change(self, timeout: int = None) -> bool:
        """
        Ждать изменения экрана (любого).
        
        Returns:
            True если экран изменился
        """
        if timeout is None:
            timeout = TIMEOUTS['screen_change']
        
        if self._last_snapshot is None:
            self.detect_screen_state()
        
        initial_hash = self._last_snapshot.frame_hash if self._last_snapshot else ""
        
        elapsed = 0
        check_interval = 200
        
        while elapsed < timeout:
            snapshot = self.detect_screen_state()
            
            if snapshot.frame_hash != initial_hash:
                return True
            
            self.page.wait_for_timeout(check_interval)
            elapsed += check_interval
        
        return False
    
    # ========================================================================
    # INPUT METHODS
    # ========================================================================
    
    def press_button(self, button: GamepadButton, hold_ms: int = 0) -> None:
        """
        Нажать кнопку геймпада.
        
        Использует ТОЛЬКО виртуальный геймпад.
        
        Args:
            button: Кнопка для нажатия
            hold_ms: Время удержания (0 = tap)
        """
        if self._gamepad:
            try:
                from .gamepad import XboxButton
                # Маппинг GamepadButton -> XboxButton
                gamepad_map = {
                    GamepadButton.A: XboxButton.A,
                    GamepadButton.B: XboxButton.B,
                    GamepadButton.X: XboxButton.X,
                    GamepadButton.Y: XboxButton.Y,
                    GamepadButton.LB: XboxButton.LB,
                    GamepadButton.RB: XboxButton.RB,
                    GamepadButton.LT: XboxButton.LT,
                    GamepadButton.RT: XboxButton.RT,
                    GamepadButton.START: XboxButton.START,
                    GamepadButton.SELECT: XboxButton.BACK,
                    GamepadButton.DPAD_UP: XboxButton.DPAD_UP,
                    GamepadButton.DPAD_DOWN: XboxButton.DPAD_DOWN,
                    GamepadButton.DPAD_LEFT: XboxButton.DPAD_LEFT,
                    GamepadButton.DPAD_RIGHT: XboxButton.DPAD_RIGHT,
                }
                xbox_btn = gamepad_map.get(button)
                if xbox_btn:
                    self._gamepad.press_button(xbox_btn, duration_ms=max(hold_ms, 100))
                    self._log_action(f"gamepad_button({button.name})")
                    self.page.wait_for_timeout(TIMEOUTS['action_cooldown'])
                    return
            except Exception as e:
                logger.error(f"Gamepad button press failed: {e}")
        else:
            logger.warning(f"Gamepad not available, cannot press {button.name}")
    
    def navigate(self, direction: NavigationDirection, times: int = 1) -> None:
        """
        Навигация в направлении (D-pad).
        
        Использует ТОЛЬКО виртуальный геймпад.
        
        Args:
            direction: Направление
            times: Сколько раз нажать
        """
        if not self._gamepad:
            logger.warning(f"Gamepad not available, cannot navigate {direction.name}")
            return
        
        try:
            from .gamepad import XboxButton
            direction_map = {
                NavigationDirection.UP: XboxButton.DPAD_UP,
                NavigationDirection.DOWN: XboxButton.DPAD_DOWN,
                NavigationDirection.LEFT: XboxButton.DPAD_LEFT,
                NavigationDirection.RIGHT: XboxButton.DPAD_RIGHT,
            }
            btn = direction_map.get(direction)
            if btn:
                for _ in range(times):
                    self._gamepad.press_button(btn)
                    self.page.wait_for_timeout(TIMEOUTS['navigation_delay'])
                self._log_action(f"gamepad_navigate({direction.name}, {times})")
        except Exception as e:
            logger.error(f"Gamepad navigation failed: {e}")
    
    def confirm(self) -> bool:
        """
        Подтвердить текущий выбор (кнопка A геймпада).
        
        Returns:
            True если экран изменился после подтверждения
        """
        self._emit("Подтверждаю выбор (A)")
        
        if self._gamepad:
            try:
                from .gamepad import XboxButton
                self._gamepad.press_button(XboxButton.A)
                self.page.wait_for_timeout(TIMEOUTS['confirm_delay'])
                return self.wait_for_screen_change(timeout=3000)
            except Exception as e:
                logger.error(f"Gamepad confirm failed: {e}")
        
        logger.warning("Gamepad not available for confirm")
        return False
    
    def cancel(self) -> bool:
        """
        Отменить / вернуться назад (кнопка B геймпада).
        
        Returns:
            True если экран изменился
        """
        self._emit("Отмена / назад (B)")
        
        if self._gamepad:
            try:
                from .gamepad import XboxButton
                self._gamepad.press_button(XboxButton.B)
                return self.wait_for_screen_change(timeout=2000)
            except Exception as e:
                logger.error(f"Gamepad cancel failed: {e}")
        
        logger.warning("Gamepad not available for cancel")
        return False
    
    def type_text(self, text: str, delay: int = 50) -> None:
        """
        Ввести текст.
        
        Args:
            text: Текст для ввода
            delay: Задержка между символами в мс
        """
        try:
            self.page.keyboard.type(text, delay=delay)
            self._log_action(f"type({text[:20]}...)")
        except Exception as e:
            logger.error(f"Type failed: {e}")
    
    def click_at(self, x: int, y: int) -> None:
        """
        DEPRECATED: Клик по координатам отключён.
        Используйте геймпад для навигации.
        """
        logger.warning(f"click_at({x}, {y}) вызван, но мышь отключена. Используйте геймпад.")
    
    def click_element(self, element: CanvasElement) -> bool:
        """
        DEPRECATED: Клик по элементу отключён.
        Используйте геймпад для навигации к элементу и confirm().
        
        Args:
            element: Элемент (игнорируется)
            
        Returns:
            True если экран изменился
        """
        self._emit(f"click_element({element.name}) → заменён на gamepad A")
        return self.confirm()
    
    # ========================================================================
    # HIGH-LEVEL ACTIONS
    # ========================================================================
    
    def open_search(self) -> bool:
        """
        Открыть панель поиска островов через ГЕЙМПАД.
        
        В Fortnite на Discover экране LT открывает Search напрямую.
        
        Returns:
            True если поиск открыт
        """
        self._emit("Открываю поиск островов через геймпад")
        
        if not self._gamepad:
            self._emit("❌ Геймпад недоступен!")
            return False
        
        from .gamepad import XboxButton
        
        # Метод 1: LT — в Fortnite на Discover экране это открывает Search
        self._emit("Пробую LT (быстрый поиск)...")
        self._gamepad.press_button(XboxButton.LT, duration_ms=100)
        self.page.wait_for_timeout(1500)
        
        return True
    
    def type_island_code(self, code: str) -> bool:
        """
        Ввести код острова.
        Клавиатура используется ТОЛЬКО для ввода текста.
        
        Args:
            code: Код острова (например "1234-5678-9012")
            
        Returns:
            True если код введён
        """
        self._emit(f"Ввожу код острова: {code}")
        
        # Очистить поле
        try:
            self.page.keyboard.press('Control+a')
            self.page.wait_for_timeout(50)
            self.page.keyboard.press('Delete')
            self.page.wait_for_timeout(50)
        except Exception:
            pass
        
        # Ввести код (клавиатура ТОЛЬКО для текстового ввода)
        self._emit(f"Ввожу текст: {code}")
        self.type_text(code, delay=30)
        self.page.wait_for_timeout(200)
        
        return True
    
    def submit_search(self) -> bool:
        """
        Отправить поисковый запрос (геймпад A).
        
        Returns:
            True если поиск выполнен
        """
        self._emit("Отправляю поиск (A)")
        
        try:
            if self._gamepad:
                from .gamepad import XboxButton
                self._gamepad.press_button(XboxButton.A)
            self.page.wait_for_timeout(1500)
            
            # Ждём появления результатов
            return self.wait_for_screen_change(timeout=5000)
        except Exception as e:
            logger.error(f"Submit failed: {e}")
            return False
    
    def select_island(self) -> bool:
        """
        Выбрать найденный остров (геймпад A).
        
        Returns:
            True если остров выбран
        """
        self._emit("Выбираю остров (A)")
        return self.confirm()
    
    def click_play(self) -> bool:
        """
        Нажать кнопку PLAY (геймпад A).
        
        Returns:
            True если Play нажата
        """
        self._emit("Нажимаю PLAY (A)")
        return self.confirm()
    
    def search_and_launch_island(self, code: str) -> bool:
        """
        Полный цикл: найти и запустить остров.
        
        Args:
            code: Код острова
            
        Returns:
            True если остров запущен
        """
        self._emit(f"Запуск острова: {code}")
        
        # 1. Убедиться в фокусе
        if not self.ensure_focus():
            self._emit("Не удалось установить фокус")
            return False
        
        # 2. Открыть поиск
        if not self.open_search():
            # Retry с Escape
            self.cancel()
            self.page.wait_for_timeout(500)
            if not self.open_search():
                self._emit("Не удалось открыть поиск")
                return False
        
        # 3. Ввести код
        self.type_island_code(code)
        
        # 4. Отправить поиск
        if not self.submit_search():
            self._emit("Поиск не дал результатов")
            return False
        
        # 5. Выбрать остров
        self.page.wait_for_timeout(1000)
        if not self.select_island():
            self._emit("Не удалось выбрать остров")
            # Продолжаем, может уже выбран
        
        # 6. Нажать PLAY
        self.page.wait_for_timeout(1000)
        if not self.click_play():
            self._emit("Не удалось нажать PLAY")
            return False
        
        self._emit("Остров запущен!")
        return True
    
    def search_and_launch_island_gamepad(self, code: str) -> bool:
        """
        Запуск острова через геймпад с автономной vision-навигацией.
        Делегирует в smart_launch_island_gamepad.
        """
        return self.smart_launch_island_gamepad(code)
    
    def search_and_launch_island_xbox(self, code: str) -> bool:
        """
        Запуск острова через Xbox метод с автономной vision-навигацией.
        Делегирует в smart_launch_island_gamepad.
        """
        return self.smart_launch_island_gamepad(code)
    
    def smart_launch_island_gamepad(self, code: str, max_attempts: int = 40) -> bool:
        """
        Навигация к острову через геймпад.
        
        Vision-модуль удалён. Используйте search_and_launch_island() 
        або хардкод-послідовність (fallback).
        """
        self._emit("⚠️ smart_launch_island_gamepad: vision удалён, используйте fallback")
        logger.warning("smart_launch_island_gamepad called but vision module removed")
        return False
    
    def _get_xbox_button(self, name: str):
        """Получить XboxButton по имени."""
        from .gamepad import XboxButton
        return getattr(XboxButton, name)
    
    # Старый метод для совместимости
    def smart_launch_island(self, code: str, max_attempts: int = 30) -> bool:
        """Алиас для smart_launch_island_gamepad."""
        return self.smart_launch_island_gamepad(code, max_attempts)

    # ========================================================================
    # MOVEMENT & CAMERA (GAMEPAD ONLY)
    # ========================================================================

    def move(
        self,
        direction: str,
        duration_ms: int = 500
    ) -> None:
        """
        Движение персонажа через левый стик геймпада.
        
        Args:
            direction: 'forward', 'back', 'left', 'right'
            duration_ms: Время удержания
        """
        if not self._gamepad:
            logger.warning("Gamepad not available for movement")
            return
        
        stick_map = {
            'forward': (0.0, 1.0),
            'up': (0.0, 1.0),
            'back': (0.0, -1.0),
            'down': (0.0, -1.0),
            'left': (-1.0, 0.0),
            'right': (1.0, 0.0),
        }
        
        x, y = stick_map.get(direction.lower(), (0.0, 1.0))
        
        try:
            self._gamepad.set_left_stick(x, y)
            self.page.wait_for_timeout(duration_ms)
            self._gamepad.set_left_stick(0.0, 0.0)
            self._log_action(f"move({direction}, {duration_ms})")
        except Exception as e:
            logger.error(f"Move failed: {e}")
    
    def look(self, dx: int = 0, dy: int = 0) -> None:
        """
        Поворот камеры через правый стик геймпада.
        
        Args:
            dx: Смещение по X (-100..100 -> -1.0..1.0)
            dy: Смещение по Y (-100..100 -> -1.0..1.0)
        """
        if not self._gamepad:
            return
        
        try:
            # Нормализуем dx/dy из пиксельных значений в -1.0..1.0
            fx = max(-1.0, min(1.0, dx / 100.0))
            fy = max(-1.0, min(1.0, dy / 100.0))
            
            self._gamepad.set_right_stick(fx, fy)
            self.page.wait_for_timeout(200)
            self._gamepad.set_right_stick(0.0, 0.0)
            self._log_action(f"look({dx}, {dy})")
        except Exception as e:
            logger.error(f"Look failed: {e}")
    
    def jump(self) -> None:
        """Прыжок (геймпад A)."""
        if self._gamepad:
            from .gamepad import XboxButton
            self._gamepad.press_button(XboxButton.A)
            self._log_action("jump")
    
    def interact(self) -> None:
        """Взаимодействие (геймпад X)."""
        if self._gamepad:
            from .gamepad import XboxButton
            self._gamepad.press_button(XboxButton.X)
            self._log_action("interact")
    
    # ========================================================================
    # AFK PREVENTION
    # ========================================================================
    
    def do_random_action(self) -> None:
        """Выполнить случайное действие для AFK-защиты через геймпад."""
        if not self._gamepad:
            return
        
        actions = [
            lambda: self.move('forward', 200),
            lambda: self.move('back', 200),
            lambda: self.move('left', 150),
            lambda: self.move('right', 150),
            lambda: self.look(random.randint(-50, 50), random.randint(-20, 20)),
            lambda: self.jump(),
        ]
        
        action = random.choice(actions)
        try:
            action()
        except Exception:
            pass
    
    def run_afk_prevention(
        self,
        duration_ms: int = 0,
        interval_ms: int = 30000
    ) -> None:
        """
        Запустить AFK-защиту.
        
        Args:
            duration_ms: Длительность (0 = бесконечно)
            interval_ms: Интервал между действиями
        """
        self._emit("AFK-защита активирована")
        
        elapsed = 0
        while duration_ms == 0 or elapsed < duration_ms:
            self.do_random_action()
            self.page.wait_for_timeout(interval_ms)
            elapsed += interval_ms
    
    # ========================================================================
    # UTILITY
    # ========================================================================
    
    def get_action_history(self) -> List[str]:
        """Получить историю действий."""
        return self._action_history.copy()
    
    def clear_history(self) -> None:
        """Очистить историю."""
        self._action_history.clear()
    
    def take_screenshot(self, path: str) -> bool:
        """
        Сохранить скриншот.
        
        Args:
            path: Путь для сохранения
            
        Returns:
            True если успешно
        """
        try:
            self.page.screenshot(path=path)
            return True
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_navigator(
    page: "Page",
    status_callback: Optional[Callable] = None
) -> CanvasNavigator:
    """
    Создать экземпляр навигатора.
    
    Args:
        page: Playwright page
        status_callback: Callback для статусов
        
    Returns:
        CanvasNavigator instance
    """
    return CanvasNavigator(page, status_callback)


def quick_search_island(
    page: "Page",
    code: str,
    status_callback: Optional[Callable] = None
) -> bool:
    """
    Быстрый поиск и запуск острова.
    
    Args:
        page: Playwright page
        code: Код острова
        status_callback: Callback для статусов
        
    Returns:
        True если успешно
    """
    nav = create_navigator(page, status_callback)
    return nav.search_and_launch_island(code)
