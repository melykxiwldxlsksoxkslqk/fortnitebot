import gymnasium as gym
from gymnasium import spaces
import numpy as np
# pyautogui больше не используется напрямую в этом файле для навигации
# import pyautogui
import time
from . import vision
import cv2
import asyncio
from playwright.async_api import async_playwright, Page, TimeoutError as PlaywrightTimeoutError
import os
import threading

# Этот класс используется графическим интерфейсом app.py
class BotLogic:
    def __init__(self, account, proxy, config, update_status_callback=None):
        self.account = account
        self.proxy = proxy
        self.config = config
        self.update_status = update_status_callback
        self.playwright = None
        self.browser = None
        self.stop_requested = False
        self.manual_lobby_event = threading.Event()
        # Для простоты, создадим экземпляр Vision прямо здесь
        # self.vision = vision() # так как vision теперь модуль
        # В идеале, vision тоже должен быть классом

    def _log(self, message):
        """Логирование с префиксом аккаунта + отправка статуса в UI."""
        login = self.account.get('login', 'unknown')
        try:
            if self.update_status:
                # Отправляем событие в UI
                self.update_status(login, str(message))
        except Exception:
            pass
        # Дублируем в stdout для обычных логов
        print(f"[{login}] {message}")

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
                success = await asyncio.to_thread(run_bot, self.account, island_code, headless, self.proxy, self.manual_lobby_event)
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
# Этот класс используется скриптом main.py для обучения
v = vision # Используем модуль vision напрямую

class FortniteEnv(gym.Env):
    """
    Среда Gymnasium для обучения ИИ-агента в Fortnite.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, island_code):
        super(FortniteEnv, self).__init__()

        self.island_code = island_code
        self.last_action_time = time.time()
        self.episode_start_time = time.time()
        self.steps_since_last_kill = 0
        self.total_kills_in_episode = 0

        # --- Пространство Действий ---
        # 0: W, 1: A, 2: D, 3: Jump, 4: LMB Attack, 5: turn left, 6: turn right,
        # 7: Tab (target search), 8: Ability 1, 9: Ability 2, 10: RMB+turn left, 11: RMB+turn right
        self.action_space = spaces.Discrete(12)

        # --- Пространство Наблюдений ---
        # Формат (H, W, C) как ранее, чтобы не нарушать совместимость существующего кода
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(360, 640, 1),
            dtype=np.uint8
        )
        self.last_tab_time = 0.0
        self.last_action_change_time = time.time()

    def _get_obs(self):
        """
        Захватывает экран, изменяет размер и переводит в градации серого.
        """
        img_bgr = v.capture_screen()
        img_resized = cv2.resize(img_bgr, (640, 360))
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        # Возвращаем как (H, W, C)
        return np.expand_dims(img_gray, axis=-1)

    def _check_for_target(self, obs):
        # Используем OBS-кадр для распознавания здоровья цели, если доступен; иначе простая эвристика
        try:
            frame_bgr = v.capture_obs_frame()
            return v.detect_enemy_health_bar(frame_bgr)
        except Exception:
            return True

    def _check_orientation(self, obs):
        mean_color = np.mean(obs)
        if mean_color > 210 or mean_color < 40:
            return False
        return True

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("--- EPISODE RESET ---")
        self.episode_start_time = time.time()
        self.steps_since_last_kill = 0
        self.total_kills_in_episode = 0

        try:
            self._navigate_to_island()
        except Exception as e:
            print(f"FATAL: Failed to navigate to island in reset(): {e}")
            return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

        print("Environment reset successful. Starting new episode (login/map load success).")
        return self._get_obs(), {}

    def _navigate_to_island(self):
        """
        Выполняет последовательность навигации и кликов для запуска острова,
        используя новый надежный метод навигации в стиле геймпада.
        """
        print("Navigating menus to launch island using keyboard...")
        # Сначала пытаемся сфокусировать окно облачного стрима/браузера
        try:
            v.focus_any_window([
                "xbox", "cloud gaming", "fortnite", "edge", "chrome", "microsoft xbox"
            ])
        except Exception:
            pass
        # Разрешаем OS-ввод для pydirectinput/vgamepad
        v.set_disable_os_input(False)
        time.sleep(5)
        # ВАЖНО: Для работы этой функции вам нужно создать новые ассеты:
        # 'assets/button_focused.png' - общий вид подсвеченной/выделенной кнопки в меню.
        # Вы можете создать более специфичные ассеты для каждой кнопки, если они сильно отличаются.

        # Шаг 1: Выбрать "Creative Mode"
        if not v.navigate_and_select_image(
            page=None, # page=None указывает, что нужно работать с захватом всего экрана
            target_template_path='assets/creative_mode_button.png',
            focused_template_path='assets/button_focused.png',
            navigation_keys=['right', 'down'],
            confidence=0.8,
            timeout=90,
            use_gamepad=True
        ):
            raise Exception("Failed to navigate to and select 'Creative Mode'.")
        time.sleep(1.5)

        # Шаг 2: Выбрать "Island Code"
        if not v.navigate_and_select_image(
            page=None,
            target_template_path='assets/island_code_button.png',
            focused_template_path='assets/button_focused.png',
            navigation_keys=['right', 'down'],
            confidence=0.8,
            timeout=25,
            use_gamepad=True
        ):
            raise Exception("Failed to navigate to and select 'Island Code'.")
        time.sleep(1.5)

        # Шаг 3: Выбрать поле ввода кода (здесь клик может быть единственным вариантом)
        # Оставляем click_on_image, так как поля ввода часто не "выделяются"
        if not v.click_on_image('assets/island_code_input_field.png', confidence=0.8, timeout=20):
            raise Exception("Input field for island code not found.")
        time.sleep(1)
        
        # Используем новую надежную функцию для ввода текста
        v.type_text(self.island_code, interval=0.15)
        v.press_key('enter')
        time.sleep(2)
        
        # Шаг 4: Запустить остров
        if not v.navigate_and_select_image(
            page=None,
            target_template_path='assets/launch_island_button.png',
            focused_template_path='assets/button_focused.png',
            navigation_keys=['right', 'up'], # Может понадобиться идти вверх
            confidence=0.8,
            timeout=20,
            use_gamepad=True
        ):
            raise Exception("Failed to navigate to and select 'Launch Island'.")

        print("Island is launching. Waiting for the match to load...")
        # Ждем исчезновения экрана загрузки / заметной смены сцены
        # Если есть специфичный индикатор HUD — можно ждать его появления.
        hud_candidates = [
            'assets/play_button.png', # пример: индикатор HUD или мини-карта (замените при наличии)
        ]
        loaded = False
        # Сначала пробуем явные образы HUD
        for path in hud_candidates:
            if os.path.exists(path):
                if v.wait_for_image_state(path, should_appear=True, confidence=0.7, timeout=90):
                    loaded = True
                    break
        # Если HUD-ассетов нет, используем смену сцены как эвристику загрузки
        if not loaded:
            loaded = v.wait_for_scene_change(timeout=90, sample_interval=1.5, diff_threshold=10.0)
        if not loaded:
            raise Exception("Map did not appear in time. Loading seems stuck.")
        print("Match loaded. Map is visible. Launch success.")

    def step(self, action):
        reward = -0.01  # базовый шаговый штраф
        terminated = False
        truncated = False
        info = {}

        # Для действий внутри игры оставляем pyautogui, так как pydirectinput
        # может требовать прав администратора и более сложной настройки.
        # Если pyautogui работает в игре, его можно оставить.
        # Если нет - нужно будет заменить все вызовы ниже на v.press_key и т.д.
        # и запускать скрипт с повышенными правами.
        import pyautogui

        observation = self._get_obs()
        has_target = self._check_for_target(observation)
        self.steps_since_last_kill += 1

        if action == 0:  # W
            pyautogui.keyDown('w'); time.sleep(0.5); pyautogui.keyUp('w')
            if has_target: reward += 0.05
        elif action == 1:  # A
            pyautogui.keyDown('a'); time.sleep(0.2); pyautogui.keyUp('a')
        elif action == 2:  # D
            pyautogui.keyDown('d'); time.sleep(0.2); pyautogui.keyUp('d')
        elif action == 3:  # Jump
            pyautogui.press('space')
        elif action == 4:  # LMB attack
            if has_target:
                pyautogui.click()
                reward += 0.5
            else:
                reward -= 0.1
        elif action == 5:  # turn left
            pyautogui.move(-150, 0, duration=0.15)
            if has_target: reward += 0.03
        elif action == 6:  # turn right
            pyautogui.move(150, 0, duration=0.15)
            if has_target: reward += 0.03
        elif action == 7:  # Tab search
            pyautogui.press('tab')
            now = time.time()
            if not has_target:
                reward += 0.02  # лёгкая награда за поиск
            if now - self.last_tab_time < 2.0:
                reward -= 0.05  # штраф за слишком частый Tab
            self.last_tab_time = now
        elif action == 8:  # Ability 1
            if has_target:
                pyautogui.press('1'); reward += 0.3
            else:
                pyautogui.press('1'); reward -= 0.05
        elif action == 9:  # Ability 2
            if has_target:
                pyautogui.press('2'); reward += 0.3
            else:
                pyautogui.press('2'); reward -= 0.05
        elif action == 10:  # RMB + turn left
            pyautogui.mouseDown(button='right')
            pyautogui.move(-200, 0, duration=0.2)
            pyautogui.mouseUp(button='right')
            if has_target: reward += 0.05
        elif action == 11:  # RMB + turn right
            pyautogui.mouseDown(button='right')
            pyautogui.move(200, 0, duration=0.2)
            pyautogui.mouseUp(button='right')
            if has_target: reward += 0.05

        # Штраф за бездействие при наличии цели
        if time.time() - self.last_action_time > 5 and has_target:
            reward -= 0.2

        if not has_target:
            reward -= 0.1

        if self.steps_since_last_kill > 250 and has_target:
            reward -= 1.0
            terminated = True
            print("Termination: No kill for too long.")

        if not self._check_orientation(observation):
            reward -= 0.3
            print("Penalty: Bad orientation.")

        self.last_action_time = time.time()

        if time.time() - self.episode_start_time > 300:
            truncated = True
            print("Truncation: Time limit reached.")

        obs = self._get_obs()
        info['kills'] = self.total_kills_in_episode
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        print("Closing Fortnite Environment.") 