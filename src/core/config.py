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


TIMEOUTS = Timeouts()

# ============================================================================
# НАСТРОЙКИ ЭМУЛЯТОРА (по умолчанию)
# ============================================================================

@dataclass
class EmulatorDefaults:
    """Настройки LDPlayer по умолчанию (общие, не per-instance)."""
    ldplayer_dir: str = field(
        default_factory=lambda: r"C:\LDPlayer\LDPlayer9"
    )
    xbox_cloud_url: str = "https://www.xbox.com/en-GB/play/games/fortnite/BT5P2X999VH2"
    epic_activate_url: str = "https://www.epicgames.com/activate"
    vpn_package: str = "com.jumpjumpvpn.jumpjump"
    chrome_package: str = "com.android.chrome"
    macro_dir: str = field(
        default_factory=lambda: os.path.join(ROOT_DIR, 'config', 'macros')
    )
    max_instances: int = 10
    adb_timeout: int = 30


EMULATOR = EmulatorDefaults()

# ============================================================================
# ЗНАЧЕНИЯ ПО УМОЛЧАНИЮ
# ============================================================================

DEFAULT_SETTINGS: Dict[str, Any] = {
    "island_code": "1234-5678-9012",
    "time_on_island_min": 15,
    "log_level": "INFO",
    "max_instances": 5,
    "vpn_region": "US",
    "macro_randomize_timing": True,
    "macro_randomize_position": True,
    "session_repeat_count": 0,   # 0 = бесконечно
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


# ============================================================================
# ФУНКЦИИ ЗАГРУЗКИ КОНФИГУРАЦИИ
# ============================================================================

def load_settings() -> Dict[str, Any]:
    """
    Загружает настройки из файла или базы данных.
    
    Returns:
        Dict с настройками или DEFAULT_SETTINGS
    """
    import json
    
    settings_file = os.path.join(CONFIG_DIR, 'settings.json')
    
    try:
        if os.path.exists(settings_file):
            with open(settings_file, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                # Merge with defaults
                result = DEFAULT_SETTINGS.copy()
                result.update(loaded)
                return result
    except Exception:
        pass
    
    return DEFAULT_SETTINGS.copy()


def load_accounts() -> List[Dict[str, str]]:
    """
    Загружает список аккаунтов из файла.
    
    Returns:
        Список словарей с email и password
    """
    accounts_file = os.path.join(CONFIG_DIR, 'accounts.txt')
    accounts = []
    
    try:
        if os.path.exists(accounts_file):
            with open(accounts_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Format: email:password or email|password
                    if ':' in line:
                        parts = line.split(':', 1)
                    elif '|' in line:
                        parts = line.split('|', 1)
                    else:
                        continue
                    
                    if len(parts) == 2:
                        accounts.append({
                            'email': parts[0].strip(),
                            'password': parts[1].strip()
                        })
    except Exception:
        pass
    
    return accounts


def load_island_code() -> str:
    """
    Загружает код острова из файла.
    
    Returns:
        Код острова или пустая строка
    """
    code_file = os.path.join(CONFIG_DIR, 'island_code.txt')
    
    try:
        if os.path.exists(code_file):
            with open(code_file, 'r', encoding='utf-8') as f:
                code = f.read().strip()
                if validate_island_code(code):
                    return code
    except Exception:
        pass
    
    return DEFAULT_SETTINGS.get('island_code', '')


def save_settings(settings: Dict[str, Any]) -> bool:
    """
    Сохраняет настройки в файл.
    
    Args:
        settings: Словарь с настройками
        
    Returns:
        True если успешно
    """
    import json
    
    settings_file = os.path.join(CONFIG_DIR, 'settings.json')
    
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False
