"""
Модуль бизнес-логики бота и RL-среды.

Содержит:
- BotLogic: класс для управления ботом из GUI
- FortniteEnv: среда Gymnasium для обучения RL-агента
"""

import time
import os
import threading
import asyncio
from typing import Optional, Callable, Dict, Any, Tuple

import numpy as np
import cv2

# Опциональные импорты для RL
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    gym = None
    spaces = None
    GYM_AVAILABLE = False

# Импорт ввода (перенесён на уровень модуля для оптимизации)
try:
    import pydirectinput
    DIRECT_INPUT_AVAILABLE = True
except ImportError:
    pydirectinput = None
    DIRECT_INPUT_AVAILABLE = False

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    pyautogui = None
    PYAUTOGUI_AVAILABLE = False

from . import vision

try:
    from .logger import get_logger
    from .config import RL, VISION, TIMEOUTS
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from src.logger import get_logger
    from src.config import RL, VISION, TIMEOUTS

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
        self.stop_requested = False
        self.manual_lobby_event = threading.Event()
        
        self._login = account.get('login', 'unknown')
        logger.info(f"Создан бот для аккаунта: {self._login}")

    def _log(self, message: str) -> None:
        """Логирование с префиксом аккаунта + отправка статуса в UI."""
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
            # После подтверждения лобби — всегда включаем глобальный debug снимков
            try:
                vision.set_global_debug(True)
            except Exception:
                pass
            self._log("Получен сигнал: лобби готово (ручной)")
        except Exception:
            pass

    def request_stop(self):
        self.stop_requested = True
        self._log("Получен запрос на остановку...")

    async def run(self):
        self._log("Бот запущен...")
        try:
            # Импортируем локально, чтобы избежать циклических импортов на этапе загрузки модулей
            from .main import run_bot, load_island_code, BadCredentialsError, BrowserClosedError

            island_code = self.config.get('island_code') or load_island_code()
            headless = bool(self.config.get('headless', True))

            # Запуск синхронной функции в отдельном потоке, чтобы не блокировать event loop
            try:
                def _forward(msg: str):
                    if self.update_status:
                        self.update_status(self.account.get('login', 'unknown'), msg)
                success = await asyncio.to_thread(run_bot, self.account, island_code, headless, self.proxy, self.manual_lobby_event, _forward)
            except BadCredentialsError:
                if self.update_status:
                    self.update_status(self.account.get('login', 'unknown'), "Неверный логин/пароль")
                return
            except BrowserClosedError:
                if self.update_status:
                    self.update_status(self.account.get('login', 'unknown'), "Браузер закрыт пользователем")
                return

            if self.stop_requested:
                self._log("Бот остановлен по запросу.")
                return

            self._log("Бот завершил работу.")
            if self.update_status:
                if success:
                    self.update_status(self.account.get('login', 'unknown'), "Успех")
                else:
                    self.update_status(self.account.get('login', 'unknown'), "Не удалось загрузить карту")
        except Exception as e:
            self._log(f"Произошла ошибка: {e}")
            if self.update_status:
                self.update_status(self.account.get('login', 'unknown'), f"Ошибка: {e.__class__.__name__}")
        finally:
             self._log("Бот выключается.")


# --- Секция для ИИ-агента ---
# Используем модуль vision напрямую
v = vision


def _create_fortnite_env_class():
    """
    Фабрика для создания класса FortniteEnv.
    Позволяет отложить проверку gymnasium до момента использования.
    """
    if not GYM_AVAILABLE:
        raise ImportError(
            "Для использования FortniteEnv необходим gymnasium. "
            "Установите: pip install -r requirements-ml.txt"
        )
    
    class FortniteEnv(gym.Env):
        """
        Среда Gymnasium для обучения ИИ-агента в Fortnite.
        
        Действия:
            0: W (вперёд)
            1: A (влево)
            2: D (вправо)
            3: Jump (прыжок)
            4: LMB (атака)
            5: Turn left (поворот влево)
            6: Turn right (поворот вправо)
            7: Tab (поиск цели)
            8: Ability 1
            9: Ability 2
            10: RMB + turn left
            11: RMB + turn right
        """
        metadata = {'render.modes': ['human']}

        def __init__(self, island_code: str):
            super(FortniteEnv, self).__init__()
            
            self.island_code = island_code
            self.last_action_time = time.time()
            self.episode_start_time = time.time()
            self.steps_since_last_kill = 0
            self.total_kills_in_episode = 0
            self.last_tab_time = 0.0
            self.last_action_change_time = time.time()
            
            # Используем конфигурацию
            self.action_space = spaces.Discrete(RL.num_actions)
            
            # Пространство наблюдений: grayscale изображение
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(VISION.observation_height, VISION.observation_width, 1),
                dtype=np.uint8
            )
            
            logger.info(f"FortniteEnv инициализирована для острова: {island_code}")

        def _get_obs(self) -> np.ndarray:
            """Захватывает экран и преобразует в наблюдение."""
            try:
                img_bgr = v.capture_screen()
                img_resized = cv2.resize(
                    img_bgr, 
                    (VISION.observation_width, VISION.observation_height)
                )
                img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                return np.expand_dims(img_gray, axis=-1)
            except Exception as e:
                logger.error(f"Ошибка захвата экрана: {e}")
                return np.zeros(self.observation_space.shape, dtype=np.uint8)

        def _check_for_target(self, obs: np.ndarray) -> bool:
            """Проверяет наличие цели на экране."""
            try:
                frame_bgr = v.capture_obs_frame()
                return v.detect_enemy_health_bar(frame_bgr)
            except Exception:
                return True

        def _check_orientation(self, obs: np.ndarray) -> bool:
            """Проверяет корректность ориентации (не уткнулся в стену/небо)."""
            mean_color = np.mean(obs)
            return 40 <= mean_color <= 210

        def _execute_action(self, action: int, has_target: bool) -> float:
            """
            Выполняет действие и возвращает награду.
            Использует pydirectinput если доступен, иначе pyautogui.
            """
            reward = RL.base_step_penalty
            
            # Выбираем модуль ввода
            if DIRECT_INPUT_AVAILABLE:
                input_module = pydirectinput
            elif PYAUTOGUI_AVAILABLE:
                input_module = pyautogui
            else:
                logger.error("Нет доступного модуля ввода!")
                return reward
            
            try:
                if action == 0:  # W (вперёд)
                    input_module.keyDown('w')
                    time.sleep(TIMEOUTS.key_press_duration * 2.5)
                    input_module.keyUp('w')
                    if has_target:
                        reward += RL.movement_with_target_reward
                        
                elif action == 1:  # A (влево)
                    input_module.keyDown('a')
                    time.sleep(TIMEOUTS.key_press_duration)
                    input_module.keyUp('a')
                    
                elif action == 2:  # D (вправо)
                    input_module.keyDown('d')
                    time.sleep(TIMEOUTS.key_press_duration)
                    input_module.keyUp('d')
                    
                elif action == 3:  # Jump
                    input_module.press('space')
                    
                elif action == 4:  # LMB attack
                    if PYAUTOGUI_AVAILABLE:
                        pyautogui.click()
                    reward += RL.attack_with_target_reward if has_target else RL.attack_without_target_penalty
                    
                elif action == 5:  # Turn left
                    if PYAUTOGUI_AVAILABLE:
                        pyautogui.move(-150, 0, duration=TIMEOUTS.action_delay)
                    if has_target:
                        reward += RL.turn_with_target_reward
                        
                elif action == 6:  # Turn right
                    if PYAUTOGUI_AVAILABLE:
                        pyautogui.move(150, 0, duration=TIMEOUTS.action_delay)
                    if has_target:
                        reward += RL.turn_with_target_reward
                        
                elif action == 7:  # Tab search
                    input_module.press('tab')
                    now = time.time()
                    if not has_target:
                        reward += RL.search_reward
                    if now - self.last_tab_time < RL.search_cooldown:
                        reward += RL.frequent_search_penalty
                    self.last_tab_time = now
                    
                elif action == 8:  # Ability 1
                    input_module.press('1')
                    reward += RL.ability_with_target_reward if has_target else RL.ability_without_target_penalty
                    
                elif action == 9:  # Ability 2
                    input_module.press('2')
                    reward += RL.ability_with_target_reward if has_target else RL.ability_without_target_penalty
                    
                elif action == 10:  # RMB + turn left
                    if PYAUTOGUI_AVAILABLE:
                        pyautogui.mouseDown(button='right')
                        pyautogui.move(-200, 0, duration=TIMEOUTS.key_press_duration)
                        pyautogui.mouseUp(button='right')
                    if has_target:
                        reward += RL.movement_with_target_reward
                        
                elif action == 11:  # RMB + turn right
                    if PYAUTOGUI_AVAILABLE:
                        pyautogui.mouseDown(button='right')
                        pyautogui.move(200, 0, duration=TIMEOUTS.key_press_duration)
                        pyautogui.mouseUp(button='right')
                    if has_target:
                        reward += RL.movement_with_target_reward
                        
            except Exception as e:
                logger.error(f"Ошибка выполнения действия {action}: {e}")
            
            return reward

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            logger.info("--- EPISODE RESET ---")
            
            self.episode_start_time = time.time()
            self.steps_since_last_kill = 0
            self.total_kills_in_episode = 0

            try:
                self._navigate_to_island()
            except Exception as e:
                logger.error(f"FATAL: Не удалось перейти на остров: {e}")
                return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

            logger.info("Среда сброшена. Начинаем новый эпизод.")
            return self._get_obs(), {}

        def _navigate_to_island(self):
            """Навигация по меню для запуска острова."""
            logger.info("Навигация по меню для запуска острова...")
            
            try:
                v.focus_any_window([
                    "xbox", "cloud gaming", "fortnite", "edge", "chrome", "microsoft xbox"
                ])
            except Exception:
                pass
            
            v.set_disable_os_input(False)
            time.sleep(5)

            # Шаг 1: Creative Mode
            if not v.navigate_and_select_image(
                page=None,
                target_template_path='assets/creative_mode_button.png',
                focused_template_path='assets/button_focused.png',
                navigation_keys=['right', 'down'],
                confidence=VISION.default_confidence,
                timeout=TIMEOUTS.game_load,
                use_gamepad=True
            ):
                raise Exception("Не удалось выбрать 'Creative Mode'")
            time.sleep(1.5)

            # Шаг 2: Island Code
            if not v.navigate_and_select_image(
                page=None,
                target_template_path='assets/island_code_button.png',
                focused_template_path='assets/button_focused.png',
                navigation_keys=['right', 'down'],
                confidence=VISION.default_confidence,
                timeout=TIMEOUTS.element_wait * 2.5,
                use_gamepad=True
            ):
                raise Exception("Не удалось выбрать 'Island Code'")
            time.sleep(1.5)

            # Шаг 3: Поле ввода кода
            if not v.click_on_image(
                'assets/island_code_input_field.png',
                confidence=VISION.default_confidence,
                timeout=TIMEOUTS.element_wait * 2
            ):
                raise Exception("Не найдено поле ввода кода острова")
            time.sleep(1)
            
            v.type_text(self.island_code, interval=TIMEOUTS.action_delay)
            v.press_key('enter')
            time.sleep(2)
            
            # Шаг 4: Launch Island
            if not v.navigate_and_select_image(
                page=None,
                target_template_path='assets/launch_island_button.png',
                focused_template_path='assets/button_focused.png',
                navigation_keys=['right', 'up'],
                confidence=VISION.default_confidence,
                timeout=TIMEOUTS.element_wait * 2,
                use_gamepad=True
            ):
                raise Exception("Не удалось нажать 'Launch Island'")

            logger.info("Остров запускается. Ожидание загрузки...")
            
            # Ожидание загрузки
            hud_candidates = ['assets/play_button.png']
            loaded = False
            
            for path in hud_candidates:
                if os.path.exists(path):
                    if v.wait_for_image_state(
                        path,
                        should_appear=True,
                        confidence=0.7,
                        timeout=TIMEOUTS.game_load
                    ):
                        loaded = True
                        break
            
            if not loaded:
                loaded = v.wait_for_scene_change(
                    timeout=TIMEOUTS.game_load,
                    sample_interval=1.5,
                    diff_threshold=10.0
                )
            
            if not loaded:
                raise Exception("Карта не загрузилась вовремя")
            
            logger.info("Карта загружена успешно!")

        def step(self, action: int):
            observation = self._get_obs()
            has_target = self._check_for_target(observation)
            self.steps_since_last_kill += 1

            # Выполняем действие и получаем награду
            reward = self._execute_action(action, has_target)
            
            terminated = False
            truncated = False
            info = {'kills': self.total_kills_in_episode}

            # Штраф за бездействие при наличии цели
            if time.time() - self.last_action_time > TIMEOUTS.idle_penalty_threshold and has_target:
                reward += RL.idle_penalty

            # Штраф за отсутствие цели
            if not has_target:
                reward += RL.no_target_penalty

            # Терминация за долгое отсутствие убийств
            if self.steps_since_last_kill > RL.steps_without_kill_limit and has_target:
                reward += RL.no_kill_timeout_penalty
                terminated = True
                logger.info("Терминация: слишком долго без убийств")

            # Штраф за плохую ориентацию
            if not self._check_orientation(observation):
                reward += RL.bad_orientation_penalty
                logger.debug("Штраф: плохая ориентация")

            self.last_action_time = time.time()

            # Ограничение по времени эпизода
            if time.time() - self.episode_start_time > TIMEOUTS.episode_max_duration:
                truncated = True
                logger.info("Усечение: достигнут лимит времени")

            return self._get_obs(), reward, terminated, truncated, info

        def render(self, mode: str = 'human') -> None:
            """Рендеринг среды (не реализован)."""
            pass

        def close(self) -> None:
            """Закрытие среды."""
            logger.info("Закрытие FortniteEnv")
    
    return FortniteEnv


# Создаём класс только при импорте, если gymnasium доступен
if GYM_AVAILABLE:
    FortniteEnv = _create_fortnite_env_class()
else:
    FortniteEnv = None  # type: ignore