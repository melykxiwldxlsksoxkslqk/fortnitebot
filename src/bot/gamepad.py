"""
Virtual Gamepad Controller для Xbox Cloud Gaming.

Использует vgamepad для эмуляции Xbox 360 контроллера.
Это более надёжный способ управления чем клики мышкой,
так как Xbox Cloud Gaming изначально заточен под геймпад.

Особенности:
- D-pad навигация по меню
- Кнопки A/B/X/Y для действий
- Поддержка стиков для движения
- Автоматический fallback на клавиатуру если vgamepad недоступен
"""

import time
import threading
from enum import Enum, auto
from typing import Optional, Callable, Tuple
from dataclasses import dataclass

from ..core.logger import get_logger

logger = get_logger(__name__)

# Пробуем импортировать vgamepad
VGAMEPAD_AVAILABLE = False
vg = None
VGAMEPAD_ERROR = None

try:
    import vgamepad as vg
    # Проверяем что драйвер ViGEmBus установлен
    test_gamepad = vg.VX360Gamepad()
    test_gamepad.reset()
    del test_gamepad
    VGAMEPAD_AVAILABLE = True
    logger.info("vgamepad loaded successfully - virtual gamepad available")
except ImportError as e:
    VGAMEPAD_ERROR = f"vgamepad not installed: {e}"
    logger.warning(VGAMEPAD_ERROR)
except Exception as e:
    VGAMEPAD_ERROR = f"ViGEmBus driver not installed or vgamepad error: {e}"
    logger.warning(f"vgamepad initialization failed: {e}")
    logger.warning("Install ViGEmBus driver from: https://github.com/ViGEm/ViGEmBus/releases")
    logger.warning("Will use keyboard fallback instead")


class XboxButton(Enum):
    """Кнопки Xbox контроллера."""
    A = auto()           # Confirm/Select (зелёная)
    B = auto()           # Back/Cancel (красная)
    X = auto()           # Action 1 (синяя)
    Y = auto()           # Action 2 (жёлтая)
    LB = auto()          # Left Bumper
    RB = auto()          # Right Bumper
    LT = auto()          # Left Trigger
    RT = auto()          # Right Trigger
    START = auto()       # Menu
    BACK = auto()        # View/Select
    DPAD_UP = auto()
    DPAD_DOWN = auto()
    DPAD_LEFT = auto()
    DPAD_RIGHT = auto()
    LEFT_STICK = auto()  # L3 - нажатие левого стика
    RIGHT_STICK = auto() # R3 - нажатие правого стика


# Маппинг Xbox кнопок на vgamepad константы
if VGAMEPAD_AVAILABLE:
    VGAMEPAD_BUTTON_MAP = {
        XboxButton.A: vg.XUSB_BUTTON.XUSB_GAMEPAD_A,
        XboxButton.B: vg.XUSB_BUTTON.XUSB_GAMEPAD_B,
        XboxButton.X: vg.XUSB_BUTTON.XUSB_GAMEPAD_X,
        XboxButton.Y: vg.XUSB_BUTTON.XUSB_GAMEPAD_Y,
        XboxButton.LB: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER,
        XboxButton.RB: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER,
        XboxButton.START: vg.XUSB_BUTTON.XUSB_GAMEPAD_START,
        XboxButton.BACK: vg.XUSB_BUTTON.XUSB_GAMEPAD_BACK,
        XboxButton.DPAD_UP: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
        XboxButton.DPAD_DOWN: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
        XboxButton.DPAD_LEFT: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
        XboxButton.DPAD_RIGHT: vg.XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
        XboxButton.LEFT_STICK: vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB,
        XboxButton.RIGHT_STICK: vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_THUMB,
    }
else:
    VGAMEPAD_BUTTON_MAP = {}


# Маппинг Xbox кнопок на клавиатуру (fallback)
KEYBOARD_BUTTON_MAP = {
    XboxButton.A: 'Enter',
    XboxButton.B: 'Escape',
    XboxButton.X: 'x',
    XboxButton.Y: 'y',
    XboxButton.LB: 'q',
    XboxButton.RB: 'e',
    XboxButton.LT: 'z',
    XboxButton.RT: 'c',
    XboxButton.START: 'Escape',
    XboxButton.BACK: 'Tab',
    XboxButton.DPAD_UP: 'ArrowUp',
    XboxButton.DPAD_DOWN: 'ArrowDown',
    XboxButton.DPAD_LEFT: 'ArrowLeft',
    XboxButton.DPAD_RIGHT: 'ArrowRight',
    XboxButton.LEFT_STICK: 'l',
    XboxButton.RIGHT_STICK: 'r',
}


@dataclass
class StickPosition:
    """Позиция стика (-1.0 до 1.0)."""
    x: float = 0.0
    y: float = 0.0


class VirtualGamepad:
    """
    Виртуальный геймпад для управления Xbox Cloud Gaming.
    
    Использует vgamepad для эмуляции Xbox 360 контроллера.
    Автоматически переключается на клавиатуру если vgamepad недоступен.
    """
    
    def __init__(self, page=None, status_callback: Optional[Callable[[str], None]] = None):
        """
        Инициализация виртуального геймпада.
        
        Args:
            page: Playwright page для keyboard fallback
            status_callback: Callback для логирования статуса
        """
        self.page = page
        self.status_callback = status_callback
        self._gamepad = None
        self._use_virtual = False
        self._left_stick = StickPosition()
        self._right_stick = StickPosition()
        
        # Пробуем создать виртуальный геймпад
        if VGAMEPAD_AVAILABLE:
            try:
                self._gamepad = vg.VX360Gamepad()
                self._use_virtual = True
                self._emit("Виртуальный Xbox геймпад создан")
                logger.info("Virtual Xbox 360 gamepad created successfully")
            except Exception as e:
                logger.warning(f"Failed to create virtual gamepad: {e}")
                self._emit(f"Не удалось создать виртуальный геймпад: {e}")
        
        if not self._use_virtual:
            self._emit("Используется клавиатура вместо геймпада")
    
    def _emit(self, message: str) -> None:
        """Отправить сообщение о статусе."""
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass
    
    @property
    def is_virtual(self) -> bool:
        """Возвращает True если используется виртуальный геймпад."""
        return self._use_virtual
    
    def press_button(self, button: XboxButton, duration_ms: int = 100) -> None:
        """
        Нажать кнопку геймпада.
        
        Args:
            button: Кнопка для нажатия
            duration_ms: Длительность нажатия в мс
        """
        if self._use_virtual and button in VGAMEPAD_BUTTON_MAP:
            try:
                vg_button = VGAMEPAD_BUTTON_MAP[button]
                self._gamepad.press_button(button=vg_button)
                self._gamepad.update()
                time.sleep(duration_ms / 1000.0)
                self._gamepad.release_button(button=vg_button)
                self._gamepad.update()
                logger.debug(f"Virtual gamepad: pressed {button.name}")
                return
            except Exception as e:
                logger.warning(f"Virtual button press failed: {e}, falling back to keyboard")
        
        # Keyboard fallback
        if self.page and button in KEYBOARD_BUTTON_MAP:
            key = KEYBOARD_BUTTON_MAP[button]
            try:
                self.page.keyboard.press(key)
                logger.debug(f"Keyboard fallback: pressed {key} for {button.name}")
            except Exception as e:
                logger.error(f"Keyboard press failed: {e}")
    
    def hold_button(self, button: XboxButton) -> None:
        """Зажать кнопку."""
        if self._use_virtual and button in VGAMEPAD_BUTTON_MAP:
            try:
                vg_button = VGAMEPAD_BUTTON_MAP[button]
                self._gamepad.press_button(button=vg_button)
                self._gamepad.update()
                return
            except Exception:
                pass
        
        if self.page and button in KEYBOARD_BUTTON_MAP:
            key = KEYBOARD_BUTTON_MAP[button]
            try:
                self.page.keyboard.down(key)
            except Exception:
                pass
    
    def release_button(self, button: XboxButton) -> None:
        """Отпустить кнопку."""
        if self._use_virtual and button in VGAMEPAD_BUTTON_MAP:
            try:
                vg_button = VGAMEPAD_BUTTON_MAP[button]
                self._gamepad.release_button(button=vg_button)
                self._gamepad.update()
                return
            except Exception:
                pass
        
        if self.page and button in KEYBOARD_BUTTON_MAP:
            key = KEYBOARD_BUTTON_MAP[button]
            try:
                self.page.keyboard.up(key)
            except Exception:
                pass
    
    def set_left_stick(self, x: float, y: float) -> None:
        """
        Установить позицию левого стика.
        
        Args:
            x: Горизонтальная позиция (-1.0 до 1.0)
            y: Вертикальная позиция (-1.0 до 1.0)
        """
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        self._left_stick = StickPosition(x, y)
        
        if self._use_virtual:
            try:
                self._gamepad.left_joystick_float(x_value_float=x, y_value_float=y)
                self._gamepad.update()
                return
            except Exception:
                pass
        
        # Keyboard fallback для стика
        if self.page:
            self._emulate_stick_with_keyboard(x, y, 'left')
    
    def set_right_stick(self, x: float, y: float) -> None:
        """
        Установить позицию правого стика (камера).
        
        Args:
            x: Горизонтальная позиция (-1.0 до 1.0)
            y: Вертикальная позиция (-1.0 до 1.0)
        """
        x = max(-1.0, min(1.0, x))
        y = max(-1.0, min(1.0, y))
        self._right_stick = StickPosition(x, y)
        
        if self._use_virtual:
            try:
                self._gamepad.right_joystick_float(x_value_float=x, y_value_float=y)
                self._gamepad.update()
                return
            except Exception:
                pass
    
    def _emulate_stick_with_keyboard(self, x: float, y: float, stick: str) -> None:
        """Эмулирует стик через клавиатуру WASD."""
        if stick != 'left' or not self.page:
            return
        
        # Определяем какие клавиши нажать
        try:
            # Отпускаем все клавиши движения
            for key in ['w', 'a', 's', 'd']:
                self.page.keyboard.up(key)
            
            # Нажимаем нужные
            if y > 0.3:
                self.page.keyboard.down('w')
            elif y < -0.3:
                self.page.keyboard.down('s')
            
            if x > 0.3:
                self.page.keyboard.down('d')
            elif x < -0.3:
                self.page.keyboard.down('a')
        except Exception:
            pass
    
    def reset_sticks(self) -> None:
        """Сбросить оба стика в центральную позицию."""
        self.set_left_stick(0.0, 0.0)
        self.set_right_stick(0.0, 0.0)
    
    def set_left_trigger(self, value: float) -> None:
        """Установить значение левого триггера (0.0 до 1.0)."""
        value = max(0.0, min(1.0, value))
        if self._use_virtual:
            try:
                # vgamepad использует 0-255 для триггеров
                self._gamepad.left_trigger_float(value_float=value)
                self._gamepad.update()
            except Exception:
                pass
    
    def set_right_trigger(self, value: float) -> None:
        """Установить значение правого триггера (0.0 до 1.0)."""
        value = max(0.0, min(1.0, value))
        if self._use_virtual:
            try:
                self._gamepad.right_trigger_float(value_float=value)
                self._gamepad.update()
            except Exception:
                pass
    
    # ========================================================================
    # HIGH-LEVEL NAVIGATION
    # ========================================================================
    
    def navigate_up(self, times: int = 1, delay_ms: int = 200) -> None:
        """Навигация вверх через D-pad."""
        for i in range(times):
            logger.info(f"[Gamepad:Input] 🎮 D-PAD UP ({i+1}/{times})")
            self.press_button(XboxButton.DPAD_UP)
            time.sleep(delay_ms / 1000.0)
    
    def navigate_down(self, times: int = 1, delay_ms: int = 200) -> None:
        """Навигация вниз через D-pad."""
        for i in range(times):
            logger.info(f"[Gamepad:Input] 🎮 D-PAD DOWN ({i+1}/{times})")
            self.press_button(XboxButton.DPAD_DOWN)
            time.sleep(delay_ms / 1000.0)
    
    def navigate_left(self, times: int = 1, delay_ms: int = 200) -> None:
        """Навигация влево через D-pad."""
        for i in range(times):
            logger.info(f"[Gamepad:Input] 🎮 D-PAD LEFT ({i+1}/{times})")
            self.press_button(XboxButton.DPAD_LEFT)
            time.sleep(delay_ms / 1000.0)
    
    def navigate_right(self, times: int = 1, delay_ms: int = 200) -> None:
        """Навигация вправо через D-pad."""
        for i in range(times):
            logger.info(f"[Gamepad:Input] 🎮 D-PAD RIGHT ({i+1}/{times})")
            self.press_button(XboxButton.DPAD_RIGHT)
            time.sleep(delay_ms / 1000.0)
    
    def confirm(self) -> None:
        """Подтвердить выбор (кнопка A)."""
        logger.info("[Gamepad:Input] 🎮 Button A (Confirm)")
        self.press_button(XboxButton.A)
    
    def cancel(self) -> None:
        """Отменить/назад (кнопка B)."""
        logger.info("[Gamepad:Input] 🎮 Button B (Cancel)")
        self.press_button(XboxButton.B)
    
    def menu(self) -> None:
        """Открыть меню (Start)."""
        logger.info("[Gamepad:Input] 🎮 Button START (Menu)")
        self.press_button(XboxButton.START)
    
    def press_a(self) -> None:
        """Нажать кнопку A."""
        logger.info("[Gamepad:Input] 🎮 Button A")
        self.press_button(XboxButton.A)
    
    def press_b(self) -> None:
        """Нажать кнопку B."""
        logger.info("[Gamepad:Input] 🎮 Button B")
        self.press_button(XboxButton.B)
    
    def press_x(self) -> None:
        """Нажать кнопку X."""
        logger.info("[Gamepad:Input] 🎮 Button X")
        self.press_button(XboxButton.X)
    
    def press_y(self) -> None:
        """Нажать кнопку Y."""
        logger.info("[Gamepad:Input] 🎮 Button Y")
        self.press_button(XboxButton.Y)
    
    def release_all(self) -> None:
        """Отпустить все кнопки."""
        logger.info("[Gamepad:Input] 🎮 Release all buttons")
        if self._use_virtual:
            try:
                self._gamepad.reset()
                self._gamepad.update()
            except Exception:
                pass
    
    # ========================================================================
    # FORTNITE-SPECIFIC NAVIGATION
    # ========================================================================
    
    def open_discover_search(self) -> None:
        """
        Открыть поиск в Discover/Creative в Fortnite.
        
        В лобби Fortnite:
        1. Нажимаем LB/RB для переключения вкладок
        2. Или используем D-pad для навигации к поиску
        """
        self._emit("Открываю поиск через навигацию...")
        
        # В Fortnite лобби поиск обычно доступен через:
        # 1. Нажатие на иконку поиска (верхняя часть экрана)
        # 2. Или комбинация кнопок
        
        # Пробуем D-pad вверх несколько раз чтобы попасть в верхнее меню
        self.navigate_up(3, delay_ms=150)
        time.sleep(0.3)
        
        # Потом влево к иконке поиска
        self.navigate_left(2, delay_ms=150)
        time.sleep(0.3)
        
        # Подтверждаем
        self.confirm()
    
    def type_with_virtual_keyboard(self, text: str, delay_ms: int = 100) -> None:
        """
        Ввести текст через виртуальную клавиатуру консоли.
        
        Использует keyboard.type() так как на Xbox Cloud Gaming
        текстовый ввод обычно идёт через виртуальную клавиатуру.
        
        Args:
            text: Текст для ввода
            delay_ms: Задержка между символами
        """
        if self.page:
            try:
                self.page.keyboard.type(text, delay=delay_ms)
            except Exception as e:
                logger.error(f"Type failed: {e}")
    
    def close(self) -> None:
        """Закрыть виртуальный геймпад."""
        if self._use_virtual:
            try:
                self.reset_sticks()
                # Отпускаем все кнопки
                self._gamepad.reset()
                self._gamepad.update()
            except Exception:
                pass
        self._emit("Геймпад отключён")


def is_gamepad_available() -> bool:
    """Проверяет доступность виртуального геймпада."""
    return VGAMEPAD_AVAILABLE
