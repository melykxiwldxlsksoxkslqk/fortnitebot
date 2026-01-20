"""
Canvas Navigation Module - умная навигация в канвас-стриме Xbox Cloud Gaming.

Этот модуль предоставляет высокоуровневую абстракцию для работы с игровым UI,
который рендерится на canvas элементе (а не в DOM). Использует computer vision
для детекции элементов и эмулирует геймпад/клавиатурный ввод.

Особенности:
- Автоматическое определение состояния экрана (лобби, меню, игра)
- Адаптивное обнаружение UI элементов через template matching
- Эмуляция геймпада через keyboard (D-pad навигация)
- Система подтверждения действий (ждёт изменения экрана)
- Retry логика с умным fallback
"""

import time
import random
import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Callable, Dict, List, Tuple, Any, Union
from functools import lru_cache

if TYPE_CHECKING:
    from playwright.sync_api import Page

import numpy as np

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
    SEARCH_PANEL = auto()
    ISLAND_PREVIEW = auto()
    MATCHMAKING = auto()
    IN_GAME = auto()
    PAUSE_MENU = auto()
    ERROR_DIALOG = auto()


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
    raw_image: Optional[np.ndarray] = None
    
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
        vision_module = None,
    ):
        self.page = page
        self.status_callback = status_callback
        self._vision = vision_module
        self._last_snapshot: Optional[ScreenSnapshot] = None
        self._action_history: List[str] = []
        self._retry_counts: Dict[str, int] = {}
        
        # Загрузить vision если не передан
        if self._vision is None:
            try:
                from ..vision import detection
                self._vision = detection
            except ImportError:
                logger.warning("Vision module not available")
    
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
            # Несколько кликов для надёжности
            self.page.mouse.click(cx, cy)
            self.page.wait_for_timeout(100)
            self.page.mouse.click(cx, cy)
            self.page.wait_for_timeout(100)
            
            # Проверяем что элемент активен
            self.page.keyboard.press('F13')  # Neutral key
            
            self._log_action(f"focus_canvas({cx}, {cy})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to focus canvas: {e}")
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Захватить текущий кадр канваса."""
        if self._vision is None:
            return None
        
        try:
            return self._vision.capture_page_bgr(self.page)
        except Exception as e:
            logger.error(f"Frame capture failed: {e}")
            return None
    
    def _compute_frame_hash(self, img: np.ndarray) -> str:
        """Вычислить хеш кадра для сравнения."""
        if img is None:
            return ""
        # Уменьшаем и хешируем
        small = img[::10, ::10].tobytes()
        return hashlib.md5(small).hexdigest()[:16]
    
    # ========================================================================
    # SCREEN STATE DETECTION
    # ========================================================================
    
    def detect_screen_state(self) -> ScreenSnapshot:
        """
        Определить текущее состояние экрана.
        
        Использует template matching и эвристики для определения
        в каком экране сейчас находится игра.
        """
        img = self.capture_frame()
        elements: List[CanvasElement] = []
        state = ScreenState.UNKNOWN
        
        if img is None:
            return ScreenSnapshot(state=state)
        
        frame_hash = self._compute_frame_hash(img)
        
        # Список шаблонов для поиска с их state-маппингом
        templates_to_check = [
            # (template_path, element_name, associated_state, roi)
            ('assets/play_button_yellow.png', 'play_button', ScreenState.LOBBY, None),
            ('assets/play_button.png', 'play_button_alt', ScreenState.LOBBY, None),
            ('assets/search_icon.png', 'search_icon', ScreenState.LOBBY, (0, 0, 0.3, 0.2)),
            ('assets/island_code_input_field.png', 'search_input', ScreenState.SEARCH_PANEL, (0, 0, 1, 0.5)),
            ('assets/select_button.png', 'select_button', ScreenState.ISLAND_PREVIEW, None),
            ('assets/loading_spinner.png', 'loading', ScreenState.LOADING, None),
            ('assets/error_dialog.png', 'error', ScreenState.ERROR_DIALOG, None),
        ]
        
        if self._vision:
            for template_path, elem_name, assoc_state, roi in templates_to_check:
                try:
                    result = self._vision.find_template(
                        img, 
                        template_path, 
                        confidence=0.65,
                        roi=roi
                    )
                    if result:
                        x, y, w, h = result
                        elements.append(CanvasElement(
                            name=elem_name,
                            x=x, y=y, width=w, height=h,
                            template_path=template_path
                        ))
                        # Определяем state по найденным элементам
                        if state == ScreenState.UNKNOWN:
                            state = assoc_state
                except Exception as e:
                    logger.debug(f"Template check failed for {template_path}: {e}")
        
        # Эвристики если template matching не дал результата
        if state == ScreenState.UNKNOWN:
            state = self._detect_state_heuristic(img)
        
        snapshot = ScreenSnapshot(
            state=state,
            elements=elements,
            frame_hash=frame_hash,
            raw_image=img
        )
        
        self._last_snapshot = snapshot
        return snapshot
    
    def _detect_state_heuristic(self, img: np.ndarray) -> ScreenState:
        """Эвристическое определение состояния по цветам/паттернам."""
        if img is None:
            return ScreenState.UNKNOWN
        
        h, w = img.shape[:2]
        
        # Анализируем среднюю яркость и цвета
        try:
            # Центральная область
            center_roi = img[h//4:3*h//4, w//4:3*w//4]
            avg_brightness = np.mean(center_roi)
            
            # Верхняя область (обычно UI)
            top_roi = img[0:h//5, :]
            top_brightness = np.mean(top_roi)
            
            # Очень тёмный экран = загрузка
            if avg_brightness < 30:
                return ScreenState.LOADING
            
            # Яркий верх с тёмным центром = меню/лобби
            if top_brightness > 100 and avg_brightness < 80:
                return ScreenState.LOBBY
            
            # Относительно равномерная яркость = в игре
            if 50 < avg_brightness < 150:
                return ScreenState.IN_GAME
                
        except Exception:
            pass
        
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
        
        Args:
            button: Кнопка для нажатия
            hold_ms: Время удержания (0 = tap)
        """
        key = button.value
        
        try:
            if hold_ms > 0:
                self.page.keyboard.down(key)
                self.page.wait_for_timeout(hold_ms)
                self.page.keyboard.up(key)
            else:
                self.page.keyboard.press(key)
            
            self._log_action(f"button({button.name})")
            self.page.wait_for_timeout(TIMEOUTS['action_cooldown'])
            
        except Exception as e:
            logger.error(f"Button press failed: {e}")
    
    def navigate(self, direction: NavigationDirection, times: int = 1) -> None:
        """
        Навигация в направлении (D-pad).
        
        Args:
            direction: Направление
            times: Сколько раз нажать
        """
        keys = NAVIGATION_KEYS[direction]
        key = keys[0]  # Используем Arrow keys
        
        for _ in range(times):
            try:
                self.page.keyboard.press(key)
                self.page.wait_for_timeout(TIMEOUTS['navigation_delay'])
            except Exception as e:
                logger.error(f"Navigation failed: {e}")
        
        self._log_action(f"navigate({direction.name}, {times})")
    
    def confirm(self) -> bool:
        """
        Подтвердить текущий выбор (A / Enter).
        
        Returns:
            True если экран изменился после подтверждения
        """
        self._emit("Подтверждаю выбор")
        self.press_button(GamepadButton.A)
        self.page.wait_for_timeout(TIMEOUTS['confirm_delay'])
        return self.wait_for_screen_change(timeout=3000)
    
    def cancel(self) -> bool:
        """
        Отменить / вернуться назад (B / Escape).
        
        Returns:
            True если экран изменился
        """
        self._emit("Отмена / назад")
        self.press_button(GamepadButton.B)
        return self.wait_for_screen_change(timeout=2000)
    
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
        Клик по координатам на канвасе.
        
        Args:
            x, y: Координаты
        """
        bounds = self.get_canvas_bounds()
        if bounds:
            # Убеждаемся что координаты внутри канваса
            cx, cy, cw, ch = bounds
            x = max(cx, min(x, cx + cw))
            y = max(cy, min(y, cy + ch))
        
        try:
            self.page.mouse.click(x, y)
            self._log_action(f"click({x}, {y})")
            self.page.wait_for_timeout(TIMEOUTS['action_cooldown'])
        except Exception as e:
            logger.error(f"Click failed: {e}")
    
    def click_element(self, element: CanvasElement) -> bool:
        """
        Клик по найденному элементу.
        
        Args:
            element: Элемент для клика
            
        Returns:
            True если экран изменился
        """
        cx, cy = element.center
        self._emit(f"Клик по {element.name} ({cx}, {cy})")
        self.click_at(cx, cy)
        return self.wait_for_screen_change(timeout=3000)
    
    # ========================================================================
    # HIGH-LEVEL ACTIONS
    # ========================================================================
    
    def open_search(self) -> bool:
        """
        Открыть панель поиска островов.
        
        Пробует несколько методов:
        1. Клик по иконке поиска (vision)
        2. Хоткей /
        3. Геймпад навигация к иконке
        
        Returns:
            True если поиск открыт
        """
        self._emit("Открываю поиск островов")
        
        # Метод 1: Vision - найти и кликнуть иконку поиска
        snapshot = self.detect_screen_state()
        search_icon = snapshot.get_element('search_icon')
        
        if search_icon:
            self._emit("Найдена иконка поиска")
            self.click_element(search_icon)
            self.page.wait_for_timeout(500)
            
            # Проверяем что поиск открылся
            new_snapshot = self.detect_screen_state()
            if new_snapshot.state == ScreenState.SEARCH_PANEL:
                return True
        
        # Метод 2: Хоткей /
        self._emit("Пробую хоткей /")
        try:
            self.page.keyboard.press('/')
            self.page.wait_for_timeout(500)
            
            new_snapshot = self.detect_screen_state()
            if new_snapshot.has_element('search_input'):
                return True
        except Exception:
            pass
        
        # Метод 3: Tab для переключения фокуса + навигация
        self._emit("Пробую Tab навигацию")
        try:
            self.page.keyboard.press('Tab')
            self.page.wait_for_timeout(200)
            self.confirm()
            
            new_snapshot = self.detect_screen_state()
            if new_snapshot.state == ScreenState.SEARCH_PANEL:
                return True
        except Exception:
            pass
        
        self._emit("Не удалось открыть поиск")
        return False
    
    def type_island_code(self, code: str) -> bool:
        """
        Ввести код острова.
        
        Args:
            code: Код острова (например "1234-5678-9012")
            
        Returns:
            True если код введён
        """
        self._emit(f"Ввожу код острова: {code}")
        
        # Убедимся что мы в панели поиска
        snapshot = self.detect_screen_state()
        
        # Найти и кликнуть поле ввода
        input_field = snapshot.get_element('search_input')
        if input_field:
            self.click_element(input_field)
            self.page.wait_for_timeout(200)
        
        # Очистить поле
        try:
            self.page.keyboard.press('Control+a')
            self.page.wait_for_timeout(50)
            self.page.keyboard.press('Delete')
            self.page.wait_for_timeout(50)
        except Exception:
            pass
        
        # Ввести код
        self.type_text(code, delay=30)
        self.page.wait_for_timeout(200)
        
        return True
    
    def submit_search(self) -> bool:
        """
        Отправить поисковый запрос.
        
        Returns:
            True если поиск выполнен
        """
        self._emit("Отправляю поиск")
        
        try:
            self.page.keyboard.press('Enter')
            self.page.wait_for_timeout(1500)
            
            # Ждём появления результатов
            return self.wait_for_screen_change(timeout=5000)
        except Exception as e:
            logger.error(f"Submit failed: {e}")
            return False
    
    def select_island(self) -> bool:
        """
        Выбрать найденный остров.
        
        Returns:
            True если остров выбран
        """
        self._emit("Выбираю остров")
        
        snapshot = self.detect_screen_state()
        
        # Ищем кнопку SELECT
        select_btn = snapshot.get_element('select_button')
        if select_btn:
            return self.click_element(select_btn)
        
        # Fallback: Enter
        self.press_button(GamepadButton.A)
        return self.wait_for_screen_change(timeout=3000)
    
    def click_play(self) -> bool:
        """
        Нажать кнопку PLAY.
        
        Returns:
            True если Play нажата
        """
        self._emit("Нажимаю PLAY")
        
        snapshot = self.detect_screen_state()
        
        # Ищем кнопку PLAY
        play_btn = snapshot.get_element('play_button') or snapshot.get_element('play_button_alt')
        if play_btn:
            return self.click_element(play_btn)
        
        # Fallback: Enter
        self.press_button(GamepadButton.A)
        return self.wait_for_screen_change(timeout=5000)
    
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
    
    # ========================================================================
    # MOVEMENT & CAMERA
    # ========================================================================
    
    def move(
        self,
        direction: str,
        duration_ms: int = 500
    ) -> None:
        """
        Движение персонажа.
        
        Args:
            direction: 'forward', 'back', 'left', 'right'
            duration_ms: Время удержания
        """
        key_map = {
            'forward': 'w', 'up': 'w',
            'back': 's', 'down': 's',
            'left': 'a',
            'right': 'd',
        }
        
        key = key_map.get(direction.lower(), 'w')
        
        try:
            self.page.keyboard.down(key)
            self.page.wait_for_timeout(duration_ms)
            self.page.keyboard.up(key)
            self._log_action(f"move({direction}, {duration_ms})")
        except Exception as e:
            logger.error(f"Move failed: {e}")
    
    def look(self, dx: int = 0, dy: int = 0) -> None:
        """
        Поворот камеры мышью.
        
        Args:
            dx: Смещение по X
            dy: Смещение по Y
        """
        bounds = self.get_canvas_bounds()
        if not bounds:
            return
        
        cx = bounds[0] + bounds[2] // 2
        cy = bounds[1] + bounds[3] // 2
        
        try:
            self.page.mouse.move(cx, cy)
            self.page.mouse.down()
            self.page.mouse.move(cx + dx, cy + dy, steps=5)
            self.page.mouse.up()
            self._log_action(f"look({dx}, {dy})")
        except Exception as e:
            logger.error(f"Look failed: {e}")
    
    def jump(self) -> None:
        """Прыжок."""
        self.page.keyboard.press('Space')
        self._log_action("jump")
    
    def interact(self) -> None:
        """Взаимодействие (E)."""
        self.page.keyboard.press('e')
        self._log_action("interact")
    
    # ========================================================================
    # AFK PREVENTION
    # ========================================================================
    
    def do_random_action(self) -> None:
        """Выполнить случайное действие для AFK-защиты."""
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
