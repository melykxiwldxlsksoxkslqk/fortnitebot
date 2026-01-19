"""
Конфигурация и константы приложения EpicBot.

Централизованное хранение всех настроек, констант и значений по умолчанию.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass, field

# ============================================================================
# ПУТИ
# ============================================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(ROOT_DIR, 'config')
ASSETS_DIR = os.path.join(ROOT_DIR, 'assets')
DEBUG_DIR = os.path.join(ROOT_DIR, 'debug')
LOGS_DIR = os.path.join(ROOT_DIR, 'logs')

# База данных
DB_PATH = os.path.join(CONFIG_DIR, 'epicbot.db')

# ============================================================================
# ВРЕМЕННЫЕ КОНСТАНТЫ (секунды)
# ============================================================================

@dataclass
class Timeouts:
    """Таймауты для различных операций."""
    # Браузер и навигация
    page_load: int = 30
    element_wait: int = 10
    login_timeout: int = 60
    game_load: int = 120
    
    # Поиск изображений
    image_search: int = 10
    image_search_fast: int = 3
    
    # Игровой процесс
    action_delay: float = 0.15
    key_press_duration: float = 0.2
    
    # RL агент
    episode_max_duration: int = 300  # 5 минут
    steps_without_kill_limit: int = 250
    idle_penalty_threshold: int = 5


TIMEOUTS = Timeouts()

# ============================================================================
# НАСТРОЙКИ VISION (КОМПЬЮТЕРНОЕ ЗРЕНИЕ)
# ============================================================================

@dataclass
class VisionConfig:
    """Настройки для модуля компьютерного зрения."""
    # Template matching
    default_confidence: float = 0.8
    high_confidence: float = 0.92
    low_confidence: float = 0.6
    
    # Пирамида изображений
    pyramid_levels: int = 2
    pyramid_scales: List[float] = field(default_factory=lambda: [0.6, 0.8, 1.0, 1.2, 1.4])
    
    # Стабильность детекции
    peak_ratio_min: float = 1.03
    stable_frames_needed: int = 1
    stable_radius_px: int = 16
    
    # ORB детектор
    orb_features: int = 800
    orb_match_ratio: float = 0.75
    
    # Размеры для RL
    observation_width: int = 640
    observation_height: int = 360


VISION = VisionConfig()

# ============================================================================
# НАСТРОЙКИ RL АГЕНТА
# ============================================================================

@dataclass
class RLConfig:
    """Настройки для Reinforcement Learning агента."""
    # Пространство действий
    num_actions: int = 12
    
    # Награды
    base_step_penalty: float = -0.01
    movement_with_target_reward: float = 0.05
    attack_with_target_reward: float = 0.5
    attack_without_target_penalty: float = -0.1
    ability_with_target_reward: float = 0.3
    ability_without_target_penalty: float = -0.05
    turn_with_target_reward: float = 0.03
    search_reward: float = 0.02
    frequent_search_penalty: float = -0.05
    no_target_penalty: float = -0.1
    idle_penalty: float = -0.2
    bad_orientation_penalty: float = -0.3
    no_kill_timeout_penalty: float = -1.0
    
    # Управление
    search_cooldown: float = 2.0


RL = RLConfig()

# ============================================================================
# НАСТРОЙКИ БРАУЗЕРА
# ============================================================================

@dataclass  
class BrowserConfig:
    """Настройки браузера Playwright/Camoufox."""
    # Общие
    default_headless: bool = False
    start_maximized: bool = True
    user_data_dir: str = field(default_factory=lambda: os.path.join(ROOT_DIR, 'browser-profile'))
    
    # Выбор браузера: 'camoufox', 'chromium', 'firefox', 'webkit'
    preferred_browser: str = 'camoufox'
    
    # URLs
    xbox_cloud_url: str = "https://www.xbox.com/play"
    
    # Таймауты (мс)
    navigation_timeout: int = 30000
    action_timeout: int = 10000
    
    # Camoufox настройки
    camoufox_humanize: bool = True
    camoufox_geoip: bool = True
    camoufox_locale: str = 'en-US'
    camoufox_timezone: str = 'America/New_York'


BROWSER = BrowserConfig()

# ============================================================================
# МАППИНГ КЛАВИШ ГЕЙМПАДА
# ============================================================================

DEFAULT_INPUT_MAP: Dict[str, List[str]] = {
    "A": ["Enter", "KeyA"],
    "B": ["Escape", "Backspace", "KeyB"],
    "X": ["KeyX"],
    "Y": ["Slash", "KeyY"],
    "UP": ["ArrowUp"],
    "DOWN": ["ArrowDown"],
    "LEFT": ["ArrowLeft"],
    "RIGHT": ["ArrowRight"],
    "LB": ["BracketLeft"],
    "RB": ["BracketRight"],
    "LT": ["Minus"],
    "RT": ["Equal"],
    "MENU": ["KeyM", "Tab"],
    "VIEW": ["KeyV", "F1"],
    "NEXUS": ["KeyN"],
}

# ============================================================================
# АССЕТЫ (ШАБЛОНЫ ИЗОБРАЖЕНИЙ)
# ============================================================================

@dataclass
class Assets:
    """Пути к ассетам для распознавания."""
    creative_mode_button: str = 'assets/creative_mode_button.png'
    island_code_button: str = 'assets/island_code_button.png'
    island_code_input_field: str = 'assets/island_code_input_field.png'
    launch_island_button: str = 'assets/launch_island_button.png'
    button_focused: str = 'assets/button_focused.png'
    play_button: str = 'assets/play_button.png'
    
    @classmethod
    def get_required(cls) -> List[str]:
        """Возвращает список обязательных ассетов."""
        return [
            cls.creative_mode_button,
            cls.island_code_button,
            cls.island_code_input_field,
            cls.launch_island_button,
        ]


ASSETS = Assets()

# ============================================================================
# ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ
# ============================================================================

DEFAULT_SETTINGS: Dict[str, Any] = {
    "island_code": "1234-5678-9012",
    "time_on_island_min": 15,
    "headless": False,
    "appearance": "Dark",
    "theme": "dark-blue",
    "ingame_mode": "passive",
    "invert_bg": False,
    "log_level": "INFO",
}

# ============================================================================
# ВАЛИДАЦИЯ
# ============================================================================

def validate_island_code(code: str) -> bool:
    """Проверяет формат кода острова (XXXX-XXXX-XXXX)."""
    import re
    pattern = r'^\d{4}-\d{4}-\d{4}$'
    return bool(re.match(pattern, code))


def validate_email(email: str) -> bool:
    """Базовая проверка email."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
