"""
Система макросів для автоматизації дій в емуляторі LDPlayer.

Класи:
- MacroAction: Тип дії (tap, swipe, key, wait, text)
- MacroStep: Одна дія з координатами/параметрами та затримкою
- MacroSequence: Послідовність кроків (один макрос)
- MacroComposer: Композиція макросів у складний сценарій
- MacroPlayer: Відтворення макросів з рандомізацією

Структура макросів (за описом):
1. launch_fortnite   — Запуск Fortnite через пошук Xbox
2. enter_island_code — Пошук та ввід коду мапи, приватна гра
3. gameplay          — AFK: біг/присідання/стрибки (1 хв × 45 раз)
4. exit_game         — Вихід + рекомендація
5. toggle_vpn        — Згорнути Chrome, перезапустити VPN, відкрити Chrome

Головний макрос:
  [enter_island_code + gameplay + exit_game] × 2 + toggle_vpn → loop forever
"""

import os
import json
import time
import random
import copy
from enum import Enum
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass, field

from ..core.logger import get_logger
from .config import MacroConfig, MACROS_DIR
from .ldplayer import LDPlayerManager, EmulatorInstance
from .exceptions import MacroError, MacroPlaybackError, MacroNotFoundError

logger = get_logger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class MacroAction(str, Enum):
    """Тип дії в макросі."""
    TAP = "tap"                     # Натискання на координати
    LONG_TAP = "long_tap"           # Довге натискання
    SWIPE = "swipe"                 # Свайп
    KEY_EVENT = "key_event"         # Кнопка Android (Back, Home тощо)
    TEXT_INPUT = "text_input"       # Ввід тексту
    WAIT = "wait"                   # Очікування (мс)
    LAUNCH_APP = "launch_app"       # Запуск додатку
    STOP_APP = "stop_app"           # Зупинка додатку
    SCREENSHOT = "screenshot"       # Скріншот (для дебагу)


# ============================================================================
# MACRO STEP
# ============================================================================


@dataclass
class MacroStep:
    """
    Одна дія макросу.
    
    Приклади:
        MacroStep(action=MacroAction.TAP, x=480, y=300, delay_ms=500)
        MacroStep(action=MacroAction.TEXT_INPUT, text="1234-5678-9012", delay_ms=300)
        MacroStep(action=MacroAction.KEY_EVENT, key_code=4, delay_ms=1000)  # Back
        MacroStep(action=MacroAction.WAIT, delay_ms=5000)
        MacroStep(action=MacroAction.SWIPE, x=240, y=400, x2=240, y2=200, delay_ms=300)
    """
    action: MacroAction
    # Координати (для tap, swipe)
    x: int = 0
    y: int = 0
    x2: int = 0    # Кінцева точка swipe
    y2: int = 0
    # Тривалість (для long_tap, swipe)
    duration_ms: int = 0
    # Key event
    key_code: int = 0
    # Текст
    text: str = ""
    # Пакет додатку
    package: str = ""
    activity: str = ""
    # Затримка ПІСЛЯ дії (мс)
    delay_ms: int = 100
    # Опис (для читабельності)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Серіалізація."""
        d = {
            'action': self.action.value,
            'delay_ms': self.delay_ms,
        }
        if self.x or self.y:
            d['x'] = self.x
            d['y'] = self.y
        if self.x2 or self.y2:
            d['x2'] = self.x2
            d['y2'] = self.y2
        if self.duration_ms:
            d['duration_ms'] = self.duration_ms
        if self.key_code:
            d['key_code'] = self.key_code
        if self.text:
            d['text'] = self.text
        if self.package:
            d['package'] = self.package
        if self.activity:
            d['activity'] = self.activity
        if self.description:
            d['description'] = self.description
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MacroStep':
        """Десеріалізація."""
        return cls(
            action=MacroAction(data['action']),
            x=data.get('x', 0),
            y=data.get('y', 0),
            x2=data.get('x2', 0),
            y2=data.get('y2', 0),
            duration_ms=data.get('duration_ms', 0),
            key_code=data.get('key_code', 0),
            text=data.get('text', ''),
            package=data.get('package', ''),
            activity=data.get('activity', ''),
            delay_ms=data.get('delay_ms', 100),
            description=data.get('description', ''),
        )


# ============================================================================
# MACRO SEQUENCE
# ============================================================================


@dataclass
class MacroSequence:
    """
    Послідовність кроків — один макрос.
    
    Приклади:
        gameplay = MacroSequence("gameplay", [
            MacroStep(MacroAction.TAP, x=100, y=300, description="Вперед"),
            MacroStep(MacroAction.WAIT, delay_ms=2000, description="Біг"),
            MacroStep(MacroAction.TAP, x=200, y=400, description="Присідання"),
            ...
        ], repeat_count=45, randomize=True)
    """
    name: str
    steps: List[MacroStep] = field(default_factory=list)
    
    # Повторення
    repeat_count: int = 1           # Скільки разів повторити
    
    # Рандомізація
    randomize_timing: bool = False
    timing_variance_ms: int = 200   # ±200 мс
    randomize_position: bool = False
    position_variance_px: int = 5   # ±5 пікселів
    
    # Метадані
    description: str = ""
    created_at: float = field(default_factory=time.time)

    @property
    def total_steps(self) -> int:
        """Загальна кількість кроків з повтореннями."""
        return len(self.steps) * self.repeat_count

    @property
    def estimated_duration_seconds(self) -> float:
        """Приблизна тривалість в секундах."""
        total_ms = sum(step.delay_ms + step.duration_ms for step in self.steps)
        return (total_ms * self.repeat_count) / 1000.0

    def add_step(self, step: MacroStep) -> 'MacroSequence':
        """Додає крок (fluent API)."""
        self.steps.append(step)
        return self

    def add_tap(self, x: int, y: int, delay_ms: int = 500, description: str = "") -> 'MacroSequence':
        """Додає натискання."""
        self.steps.append(MacroStep(
            action=MacroAction.TAP, x=x, y=y,
            delay_ms=delay_ms, description=description,
        ))
        return self

    def add_wait(self, delay_ms: int, description: str = "") -> 'MacroSequence':
        """Додає очікування."""
        self.steps.append(MacroStep(
            action=MacroAction.WAIT,
            delay_ms=delay_ms, description=description,
        ))
        return self

    def add_text(self, text: str, delay_ms: int = 300, description: str = "") -> 'MacroSequence':
        """Додає ввід тексту."""
        self.steps.append(MacroStep(
            action=MacroAction.TEXT_INPUT,
            text=text, delay_ms=delay_ms, description=description,
        ))
        return self

    def add_key(self, key_code: int, delay_ms: int = 300, description: str = "") -> 'MacroSequence':
        """Додає натискання кнопки."""
        self.steps.append(MacroStep(
            action=MacroAction.KEY_EVENT,
            key_code=key_code, delay_ms=delay_ms, description=description,
        ))
        return self

    def add_swipe(
        self, x1: int, y1: int, x2: int, y2: int,
        duration_ms: int = 300, delay_ms: int = 300,
        description: str = "",
    ) -> 'MacroSequence':
        """Додає свайп."""
        self.steps.append(MacroStep(
            action=MacroAction.SWIPE,
            x=x1, y=y1, x2=x2, y2=y2,
            duration_ms=duration_ms,
            delay_ms=delay_ms, description=description,
        ))
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Серіалізація."""
        return {
            'name': self.name,
            'description': self.description,
            'repeat_count': self.repeat_count,
            'randomize_timing': self.randomize_timing,
            'timing_variance_ms': self.timing_variance_ms,
            'randomize_position': self.randomize_position,
            'position_variance_px': self.position_variance_px,
            'created_at': self.created_at,
            'steps': [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MacroSequence':
        """Десеріалізація."""
        seq = cls(
            name=data['name'],
            description=data.get('description', ''),
            repeat_count=data.get('repeat_count', 1),
            randomize_timing=data.get('randomize_timing', False),
            timing_variance_ms=data.get('timing_variance_ms', 200),
            randomize_position=data.get('randomize_position', False),
            position_variance_px=data.get('position_variance_px', 5),
            created_at=data.get('created_at', 0),
        )
        for step_data in data.get('steps', []):
            seq.steps.append(MacroStep.from_dict(step_data))
        return seq

    def save(self, directory: Optional[str] = None) -> str:
        """Зберігає макрос у файл. Повертає шлях."""
        directory = directory or MACROS_DIR
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.debug(f"Макрос збережено: {path}")
        return path

    @classmethod
    def load(cls, name: str, directory: Optional[str] = None) -> 'MacroSequence':
        """Завантажує макрос з файлу."""
        directory = directory or MACROS_DIR
        path = os.path.join(directory, f"{name}.json")
        if not os.path.exists(path):
            raise MacroNotFoundError(f"Макрос не знайдено: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# MACRO COMPOSER
# ============================================================================


class MacroComposer:
    """
    Компонує складні макроси з простих.
    
    Дозволяє створити головний макрос зі списку підмакросів:
    
    main_macro = MacroComposer("main") \\
        .add(enter_island_code) \\
        .add(gameplay) \\
        .add(exit_game) \\
        .add(enter_island_code) \\
        .add(gameplay) \\
        .add(exit_game) \\
        .add(toggle_vpn) \\
        .set_loop_forever(True) \\
        .build()
    """

    def __init__(self, name: str, description: str = ""):
        self._name = name
        self._description = description
        self._sequences: List[MacroSequence] = []
        self._loop_forever: bool = False
        self._loop_count: int = 1

    def add(self, sequence: MacroSequence) -> 'MacroComposer':
        """Додає макрос до композиції."""
        self._sequences.append(sequence)
        return self

    def add_multiple(self, sequence: MacroSequence, count: int) -> 'MacroComposer':
        """Додає макрос кілька разів."""
        for _ in range(count):
            self._sequences.append(copy.deepcopy(sequence))
        return self

    def set_loop_forever(self, enabled: bool = True) -> 'MacroComposer':
        """Встановлює безкінечне повторення."""
        self._loop_forever = enabled
        return self

    def set_loop_count(self, count: int) -> 'MacroComposer':
        """Встановлює кількість повторень."""
        self._loop_count = count
        self._loop_forever = False
        return self

    @property
    def sequences(self) -> List[MacroSequence]:
        """Список макросів у композиції."""
        return self._sequences

    @property
    def loop_forever(self) -> bool:
        """Чи безкінечне повторення."""
        return self._loop_forever

    @property
    def loop_count(self) -> int:
        """Кількість повторень."""
        return self._loop_count

    def build(self) -> Dict[str, Any]:
        """
        Будує фінальну конфігурацію макросу.
        
        Returns:
            Словник з повною конфігурацією для MacroPlayer
        """
        return {
            'name': self._name,
            'description': self._description,
            'loop_forever': self._loop_forever,
            'loop_count': self._loop_count,
            'sequences': [seq.to_dict() for seq in self._sequences],
            'total_sequences': len(self._sequences),
            'estimated_single_pass_minutes': sum(
                s.estimated_duration_seconds for s in self._sequences
            ) / 60.0,
        }

    def save(self, directory: Optional[str] = None) -> str:
        """Зберігає скомпонований макрос."""
        directory = directory or MACROS_DIR
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"composed_{self._name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.build(), f, indent=2, ensure_ascii=False)
        logger.info(f"Скомпонований макрос збережено: {path}")
        return path

    @classmethod
    def load(cls, name: str, directory: Optional[str] = None) -> 'MacroComposer':
        """Завантажує скомпонований макрос."""
        directory = directory or MACROS_DIR
        path = os.path.join(directory, f"composed_{name}.json")
        if not os.path.exists(path):
            raise MacroNotFoundError(f"Скомпонований макрос не знайдено: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        composer = cls(data['name'], data.get('description', ''))
        composer._loop_forever = data.get('loop_forever', False)
        composer._loop_count = data.get('loop_count', 1)
        for seq_data in data.get('sequences', []):
            composer._sequences.append(MacroSequence.from_dict(seq_data))
        return composer


# ============================================================================
# MACRO PLAYER
# ============================================================================


class MacroPlayer:
    """
    Відтворює макроси в емуляторі LDPlayer.
    
    Підтримує:
    - Відтворення одного MacroSequence
    - Відтворення скомпонованого MacroComposer
    - Рандомізацію натискань (час + позиція)
    - Зупинку/паузу
    - Callback для моніторингу прогресу
    
    Використання:
        player = MacroPlayer(ldplayer, instance, config)
        player.play(gameplay_macro)
        player.play_composed(main_macro)
    """

    def __init__(
        self,
        ldplayer: LDPlayerManager,
        instance: EmulatorInstance,
        config: Optional[MacroConfig] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        self._ldplayer = ldplayer
        self._instance = instance
        self._config = config or MacroConfig()
        self._status_callback = status_callback
        self._stop_check = stop_check or (lambda: False)

        # Стан відтворення
        self._is_playing: bool = False
        self._is_paused: bool = False
        self._current_sequence: Optional[str] = None
        self._current_step: int = 0
        self._current_repeat: int = 0

        logger.info(f"MacroPlayer створено для '{instance.name}'")

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def is_playing(self) -> bool:
        """Чи йде відтворення."""
        return self._is_playing

    @property
    def is_paused(self) -> bool:
        """Чи на паузі."""
        return self._is_paused

    # ========================================================================
    # ВНУТРІШНІ
    # ========================================================================

    def _emit(self, message: str) -> None:
        """Відправляє статус."""
        logger.info(f"[Macro:{self._instance.name}] {message}")
        if self._status_callback:
            try:
                self._status_callback(message)
            except Exception:
                pass

    def _should_stop(self) -> bool:
        """Перевіряє зупинку."""
        try:
            return self._stop_check()
        except Exception:
            return False

    def _randomize_delay(self, delay_ms: int) -> float:
        """Додає рандомізацію до затримки."""
        if not self._config.randomize_timing:
            return delay_ms / 1000.0

        variance = self._config.timing_variance_ms
        randomized = delay_ms + random.randint(-variance, variance)
        return max(50, randomized) / 1000.0  # Мінімум 50 мс

    def _randomize_position(self, x: int, y: int) -> tuple:
        """Додає рандомізацію до координат."""
        if not self._config.randomize_position:
            return x, y

        variance = self._config.position_variance_px
        rx = x + random.randint(-variance, variance)
        ry = y + random.randint(-variance, variance)
        return max(0, rx), max(0, ry)

    # ========================================================================
    # ВИКОНАННЯ ОДНОГО КРОКУ
    # ========================================================================

    def _execute_step(self, step: MacroStep) -> None:
        """Виконує одну дію макросу."""
        action = step.action

        if action == MacroAction.TAP:
            x, y = self._randomize_position(step.x, step.y)
            self._ldplayer.adb_tap(self._instance, x, y)

        elif action == MacroAction.LONG_TAP:
            x, y = self._randomize_position(step.x, step.y)
            duration = step.duration_ms or 1000
            # Long tap через swipe на те ж місце
            self._ldplayer.adb_swipe(self._instance, x, y, x, y, duration)

        elif action == MacroAction.SWIPE:
            x1, y1 = self._randomize_position(step.x, step.y)
            x2, y2 = self._randomize_position(step.x2, step.y2)
            duration = step.duration_ms or 300
            self._ldplayer.adb_swipe(self._instance, x1, y1, x2, y2, duration)

        elif action == MacroAction.KEY_EVENT:
            self._ldplayer.adb_key_event(self._instance, step.key_code)

        elif action == MacroAction.TEXT_INPUT:
            self._ldplayer.adb_text(self._instance, step.text)

        elif action == MacroAction.LAUNCH_APP:
            self._ldplayer.launch_app(
                self._instance, step.package, step.activity,
            )

        elif action == MacroAction.STOP_APP:
            self._ldplayer.stop_app(self._instance, step.package)

        elif action == MacroAction.SCREENSHOT:
            # Для дебагу
            path = os.path.join(
                self._config.macros_dir,
                f"screenshot_{int(time.time())}.png",
            )
            self._ldplayer.take_screenshot(self._instance, path)

        elif action == MacroAction.WAIT:
            pass  # Затримка обробляється нижче

        # Затримка після дії
        delay = self._randomize_delay(step.delay_ms)
        if delay > 0:
            time.sleep(delay)

    # ========================================================================
    # ВІДТВОРЕННЯ ПОСЛІДОВНОСТІ
    # ========================================================================

    def play(self, sequence: MacroSequence) -> bool:
        """
        Відтворює один макрос (з повтореннями).
        
        Args:
            sequence: Макрос для відтворення
            
        Returns:
            True якщо завершено успішно (False якщо зупинено)
        """
        self._is_playing = True
        self._current_sequence = sequence.name

        total_repeats = sequence.repeat_count
        total_steps = len(sequence.steps)

        self._emit(f"▶ Макрос '{sequence.name}': {total_repeats}× по {total_steps} кроків")

        try:
            for repeat in range(total_repeats):
                self._current_repeat = repeat + 1

                if self._should_stop():
                    self._emit(f"⏹ Макрос зупинено на повторі {repeat + 1}/{total_repeats}")
                    return False

                if total_repeats > 1:
                    self._emit(f"  Повтор {repeat + 1}/{total_repeats}")

                for step_idx, step in enumerate(sequence.steps):
                    self._current_step = step_idx + 1

                    if self._should_stop():
                        return False

                    # Пауза
                    while self._is_paused:
                        time.sleep(0.5)
                        if self._should_stop():
                            return False

                    # Виконуємо крок
                    if step.description:
                        logger.debug(f"  [{step_idx + 1}/{total_steps}] {step.description}")

                    self._execute_step(step)

            self._emit(f"✓ Макрос '{sequence.name}' завершено")
            return True

        except Exception as e:
            logger.error(f"Помилка макросу '{sequence.name}': {e}")
            raise MacroPlaybackError(f"Помилка відтворення: {e}") from e

        finally:
            self._is_playing = False
            self._current_sequence = None

    def play_composed(self, composer: MacroComposer) -> bool:
        """
        Відтворює скомпонований макрос.
        
        Підтримує:
        - Послідовне виконання підмакросів
        - Безкінечне повторення (loop_forever)
        - Затримки між макросами
        
        Args:
            composer: Скомпонований макрос
            
        Returns:
            True якщо завершено успішно
        """
        build = composer.build()
        self._emit(
            f"▶▶ Композиція '{build['name']}': "
            f"{build['total_sequences']} макросів, "
            f"{'∞ loop' if composer.loop_forever else f'{composer.loop_count}× loop'}"
        )

        loop_num = 0
        while True:
            loop_num += 1

            if not composer.loop_forever and loop_num > composer.loop_count:
                break

            if self._should_stop():
                self._emit("⏹ Композиція зупинена")
                return False

            self._emit(f"--- Прохід #{loop_num} ---")

            for seq_idx, sequence in enumerate(composer.sequences):
                if self._should_stop():
                    return False

                self._emit(f"  [{seq_idx + 1}/{len(composer.sequences)}] → {sequence.name}")

                success = self.play(sequence)
                if not success:
                    return False

                # Затримка між макросами
                transition_delay = self._config.macro_transition_delay_ms / 1000.0
                if transition_delay > 0:
                    time.sleep(transition_delay)

        self._emit(f"✓✓ Композиція '{build['name']}' завершена")
        return True

    # ========================================================================
    # УПРАВЛІННЯ
    # ========================================================================

    def pause(self) -> None:
        """Ставить на паузу."""
        self._is_paused = True
        self._emit("⏸ Макрос на паузі")

    def resume(self) -> None:
        """Продовжує відтворення."""
        self._is_paused = False
        self._emit("▶ Макрос продовжено")

    def stop(self) -> None:
        """Зупиняє відтворення."""
        self._is_playing = False
        self._emit("⏹ Макрос зупинено")

    def get_status(self) -> Dict[str, Any]:
        """Повертає статус відтворення."""
        return {
            'is_playing': self._is_playing,
            'is_paused': self._is_paused,
            'current_sequence': self._current_sequence,
            'current_step': self._current_step,
            'current_repeat': self._current_repeat,
        }


# ============================================================================
# ФАБРИКА СТАНДАРТНИХ МАКРОСІВ
# ============================================================================


class MacroFactory:
    """
    Фабрика стандартних макросів для Fortnite фарму.
    
    Створює готові макроси за описом:
    1. launch_fortnite
    2. enter_island_code
    3. gameplay (AFK)
    4. exit_game
    5. toggle_vpn
    
    Координати розраховані для розширення 960×540.
    Для інших розширень масштабуються автоматично.
    """

    def __init__(
        self,
        resolution_width: int = 960,
        resolution_height: int = 540,
        island_code: str = "7048-8422-2298",
    ):
        self._width = resolution_width
        self._height = resolution_height
        self._island_code = island_code

    def _scale_x(self, x: int) -> int:
        """Масштабує X координату."""
        return int(x * self._width / 960)

    def _scale_y(self, y: int) -> int:
        """Масштабує Y координату."""
        return int(y * self._height / 540)

    def create_launch_fortnite(self) -> MacroSequence:
        """
        Макрос 1: Запуск Fortnite через пошук на Xbox Cloud Gaming.
        
        Кроки:
        - Натиснути пошук у Chrome
        - Вписати "Fortnite"
        - Вибрати результат
        - Натиснути Play
        """
        macro = MacroSequence(
            name="launch_fortnite",
            description="Запуск Fortnite через Xbox Cloud Gaming в Chrome",
        )

        # Натиснути на рядок пошуку Xbox
        macro.add_tap(self._scale_x(480), self._scale_y(50), 1000, "Пошук на Xbox")
        # Ввести Fortnite
        macro.add_text("Fortnite", 1000, "Ввід 'Fortnite'")
        # Чекаємо результати
        macro.add_wait(3000, "Очікування результатів пошуку")
        # Натиснути на результат
        macro.add_tap(self._scale_x(480), self._scale_y(200), 2000, "Вибір Fortnite")
        # Натиснути Play
        macro.add_tap(self._scale_x(480), self._scale_y(350), 5000, "Натиснути Play")
        # Чекаємо завантаження гри
        macro.add_wait(30000, "Очікування завантаження Fortnite")

        return macro

    def create_enter_island_code(self) -> MacroSequence:
        """
        Макрос 2: Пошук та ввід коду мапи.
        
        Кроки:
        - Відкрити Discover / Creative
        - Натиснути "Island Code"
        - Ввести код
        - Вибрати острів
        - Увімкнути Private match
        - Натиснути Play
        """
        macro = MacroSequence(
            name="enter_island_code",
            description=f"Ввід коду острову: {self._island_code}",
        )

        # Відкрити меню Discover
        macro.add_tap(self._scale_x(480), self._scale_y(480), 2000, "Відкрити Discover")
        # Натиснути Island Code
        macro.add_tap(self._scale_x(200), self._scale_y(100), 1000, "Island Code кнопка")
        # Очистити поле та ввести код
        macro.add_tap(self._scale_x(480), self._scale_y(200), 500, "Фокус на поле вводу")
        macro.add_text(self._island_code, 1000, f"Ввід коду: {self._island_code}")
        # Пошук
        macro.add_key(66, 2000, "Enter для пошуку")  # KEYCODE_ENTER
        # Вибрати острів зі списку
        macro.add_tap(self._scale_x(480), self._scale_y(300), 2000, "Вибір острову")
        # Private match toggle
        macro.add_tap(self._scale_x(700), self._scale_y(400), 1000, "Private match ON")
        # Play
        macro.add_tap(self._scale_x(480), self._scale_y(470), 5000, "Запуск гри")
        # Чекаємо завантаження
        macro.add_wait(15000, "Очікування завантаження карти")

        return macro

    def create_gameplay(self, config: Optional[MacroConfig] = None) -> MacroSequence:
        """
        Макрос 3: AFK геймплей (~1 хвилина дій × N повторів).
        
        Дії: біг вперед, назад, присідання, стрибки.
        Тільки рух — кнопки не натискаються.
        Рандомізація увімкнена.
        """
        cfg = config or MacroConfig()
        macro = MacroSequence(
            name="gameplay",
            description="AFK геймплей: біг, присідання, стрибки",
            repeat_count=cfg.gameplay_repeat_count,  # 45 разів
            randomize_timing=True,
            timing_variance_ms=cfg.timing_variance_ms,
            randomize_position=True,
            position_variance_px=cfg.position_variance_px,
        )

        cx = self._scale_x(480)
        cy = self._scale_y(270)

        # Біг вперед (свайп вверх на лівому стіку)
        macro.add_swipe(
            self._scale_x(120), self._scale_y(400),
            self._scale_x(120), self._scale_y(250),
            duration_ms=3000, delay_ms=500,
            description="Біг вперед",
        )

        # Присідання (натискання кнопки присідання)
        macro.add_tap(self._scale_x(850), self._scale_y(450), 1000, "Присідання")

        # Стрибок
        macro.add_tap(self._scale_x(880), self._scale_y(350), 500, "Стрибок")

        # Біг назад (свайп вниз)
        macro.add_swipe(
            self._scale_x(120), self._scale_y(300),
            self._scale_x(120), self._scale_y(450),
            duration_ms=2000, delay_ms=500,
            description="Біг назад",
        )

        # Поворот камери (свайп вправо)
        macro.add_swipe(
            self._scale_x(700), self._scale_y(270),
            self._scale_x(800), self._scale_y(270),
            duration_ms=500, delay_ms=300,
            description="Поворот камери",
        )

        # Чекаємо (загалом ~1 хвилина з затримками)
        remaining_ms = max(0, cfg.gameplay_duration_seconds * 1000 - 8000)
        if remaining_ms > 0:
            macro.add_wait(remaining_ms, "Очікування (AFK)")

        return macro

    def create_exit_game(self) -> MacroSequence:
        """
        Макрос 4: Вихід з Fortnite + рекомендація.
        """
        macro = MacroSequence(
            name="exit_game",
            description="Вихід з Fortnite та оцінка",
        )

        # Відкрити меню (ESC / Menu)
        macro.add_key(111, 2000, "Відкрити меню (ESC)")  # KEYCODE_ESCAPE
        # Натиснути "Leave match" / "Exit"
        macro.add_tap(self._scale_x(480), self._scale_y(350), 2000, "Leave match")
        # Підтвердити
        macro.add_tap(self._scale_x(480), self._scale_y(300), 3000, "Підтвердити вихід")
        # Рекомендація (thumbs up)
        macro.add_tap(self._scale_x(350), self._scale_y(400), 2000, "Рекомендація 👍")
        # Чекаємо повернення в лобі
        macro.add_wait(5000, "Повернення в лобі")

        return macro

    def create_toggle_vpn(self, vpn_package: str = "com.jumpjump.vpn") -> MacroSequence:
        """
        Макрос 5: Перезапуск VPN.
        
        Кроки:
        - Згорнути Chrome
        - Відкрити VPN
        - Вимкнути VPN
        - Увімкнути VPN
        - Повернутись в Chrome
        """
        macro = MacroSequence(
            name="toggle_vpn",
            description="Перезапуск JumpJumpVPN",
        )

        # Home (згорнути Chrome)
        macro.add_key(3, 1000, "Home (згорнути Chrome)")  # KEYCODE_HOME

        # Запуск VPN
        macro.add_step(MacroStep(
            action=MacroAction.LAUNCH_APP,
            package=vpn_package,
            delay_ms=3000,
            description="Відкрити VPN",
        ))

        # Вимкнути VPN (натиснути кнопку Disconnect)
        macro.add_tap(self._scale_x(480), self._scale_y(400), 3000, "Disconnect VPN")

        # Увімкнути VPN (натиснути кнопку Connect)
        macro.add_tap(self._scale_x(480), self._scale_y(400), 5000, "Connect VPN")

        # Чекаємо підключення
        macro.add_wait(10000, "Очікування підключення VPN")

        # Повернутись в Chrome (Recent apps → Chrome)
        macro.add_key(3, 1000, "Home")
        macro.add_step(MacroStep(
            action=MacroAction.LAUNCH_APP,
            package="com.android.chrome",
            delay_ms=3000,
            description="Відкрити Chrome",
        ))

        return macro

    def create_full_session_macro(self, config: Optional[MacroConfig] = None) -> MacroComposer:
        """
        Створює повний макрос сесії:
        
        [enter_island_code + gameplay + exit_game] × 2 + toggle_vpn
        Безкінечне повторення.
        """
        cfg = config or MacroConfig()

        enter = self.create_enter_island_code()
        gameplay = self.create_gameplay(cfg)
        exit_game = self.create_exit_game()
        toggle_vpn = self.create_toggle_vpn()

        composer = MacroComposer(
            name="full_session",
            description="Повна сесія фарму: 2 гри + перезапуск VPN, loop ∞",
        )

        # Додаємо [ввід коду + геймплей + вихід] × 2
        for _ in range(cfg.games_per_vpn_session):
            composer.add(enter)
            composer.add(gameplay)
            composer.add(exit_game)

        # Додаємо перезапуск VPN
        composer.add(toggle_vpn)

        # Безкінечне повторення
        composer.set_loop_forever(True)

        return composer
