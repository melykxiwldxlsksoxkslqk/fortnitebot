"""
Конфігурація модуля емулятора.

Dataclass-и для всіх налаштувань: LDPlayer, VPN, макроси, сесії, APK.
Побудовано аналогічно core.config (TIMEOUTS, VISION, BROWSER тощо).
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..core.config import ROOT_DIR, CONFIG_DIR

# ============================================================================
# ШЛЯХИ
# ============================================================================

EMULATOR_CONFIG_DIR = os.path.join(CONFIG_DIR, 'emulator')
MACROS_DIR = os.path.join(EMULATOR_CONFIG_DIR, 'macros')
APK_DIR = os.path.join(EMULATOR_CONFIG_DIR, 'apks')
ACCOUNTS_DATA_DIR = os.path.join(EMULATOR_CONFIG_DIR, 'accounts')

# ============================================================================
# КОНФІГУРАЦІЯ ІНСТАНСУ LDPLAYER
# ============================================================================


@dataclass
class InstanceConfig:
    """
    Налаштування одного інстансу LDPlayer.

    Параметри оптимізовані для мінімального споживання ресурсів:
    - Планшет з мінімальним розширенням
    - 2 ядра / 2 ГБ RAM
    - 10 FPS
    - Різні моделі та IMEI для кожного інстансу
    """
    name: str = "FarmBot"
    # Продуктивність
    cpu_cores: int = 2
    ram_mb: int = 2048
    fps: int = 10
    # Дисплей (планшет, мінімальне розширення)
    resolution_width: int = 960
    resolution_height: int = 540
    dpi: int = 160
    # Пристрій
    device_model: str = "SM-T510"
    manufacturer: str = "samsung"
    imei: str = ""          # Генерується автоматично якщо порожньо
    android_id: str = ""    # Генерується автоматично якщо порожньо
    # Рендерінг
    render_mode: str = "speed"  # 'speed' | 'compatibility' | 'opengl'
    # Мережа
    use_proxy: bool = False
    proxy_host: str = ""
    proxy_port: str = ""
    proxy_username: str = ""
    proxy_password: str = ""

    def to_ldplayer_args(self) -> Dict[str, str]:
        """Конвертує конфігурацію в аргументи ldconsole."""
        args = {
            '--cpu': str(self.cpu_cores),
            '--memory': str(self.ram_mb),
            '--fps': str(self.fps),
            '--width': str(self.resolution_width),
            '--height': str(self.resolution_height),
            '--dpi': str(self.dpi),
            '--manufacturer': self.manufacturer,
            '--model': self.device_model,
        }
        if self.imei:
            args['--imei'] = self.imei
        if self.android_id:
            args['--androidid'] = self.android_id
        return args


# ============================================================================
# КОНФІГУРАЦІЯ LDPLAYER
# ============================================================================


@dataclass
class LDPlayerConfig:
    """Шлях до LDPlayer та загальні налаштування."""
    # Шляхи до LDPlayer (типові для Windows)
    install_dir: str = r"C:\LDPlayer\LDPlayer9"
    ldconsole_path: str = ""  # Автовизначення
    ld_path: str = ""         # Автовизначення

    # Таймаути (секунди)
    launch_timeout: int = 120
    shutdown_timeout: int = 30
    adb_timeout: int = 30

    # Обмеження
    max_instances: int = 10
    
    # Шаблон для нових інстансів
    default_instance: InstanceConfig = field(default_factory=InstanceConfig)

    def __post_init__(self):
        """Автовизначення шляхів."""
        if not self.ldconsole_path:
            self.ldconsole_path = os.path.join(self.install_dir, 'ldconsole.exe')
        if not self.ld_path:
            self.ld_path = os.path.join(self.install_dir, 'ld.exe')


# ============================================================================
# КОНФІГУРАЦІЯ VPN
# ============================================================================


@dataclass
class VPNConfig:
    """
    Налаштування JumpJumpVPN.
    
    VPN має бути увімкнений весь час.
    Безкоштовна версія дає ~2.5 години.
    """
    package_name: str = "com.jumpjump.vpn"
    activity_name: str = "com.jumpjump.vpn.MainActivity"
    # Регіон за замовчуванням
    default_region: str = "United States"
    # Таймінги
    connection_timeout: int = 30          # секунд на підключення
    session_duration_minutes: int = 150   # ~2.5 години безкоштовно
    reconnect_delay: int = 5              # секунд між перепідключеннями
    # Координати UI елементів (налаштовуються під розширення)
    connect_button_x: int = 480
    connect_button_y: int = 400
    region_button_x: int = 480
    region_button_y: int = 100
    # Lucky Patcher — модифікований APK без реклами
    use_patched_apk: bool = True
    patched_apk_path: str = ""


# ============================================================================
# КОНФІГУРАЦІЯ APK
# ============================================================================


@dataclass
class APKConfig:
    """Шляхи та налаштування APK файлів."""
    # APK файли для встановлення
    vpn_apk_path: str = ""
    chrome_apk_path: str = ""
    lucky_patcher_apk_path: str = ""
    # Пакети
    chrome_package: str = "com.android.chrome"
    lucky_patcher_package: str = "ru.mgames.luckypatcher"
    # Директорія з APK
    apk_directory: str = field(default_factory=lambda: APK_DIR)


# ============================================================================
# КОНФІГУРАЦІЯ МАКРОСІВ
# ============================================================================


@dataclass
class MacroConfig:
    """
    Налаштування системи макросів.
    
    Структура макросів:
    1. launch_fortnite   — Запуск Fortnite через пошук
    2. enter_island_code — Пошук та ввід коду мапи, приватна гра, запуск
    3. gameplay          — AFK дії 1 хв (біг, присідання, стрибки тощо) × 45 разів
    4. exit_game         — Вихід з Fortnite, рекомендація
    5. toggle_vpn        — Згортання Chrome, перезапуск VPN

    Головний макрос = [enter_island_code + gameplay + exit_game] × 2 + toggle_vpn
    Повторюється безкінечно.
    """
    # Директорія зберігання макросів
    macros_dir: str = field(default_factory=lambda: MACROS_DIR)

    # Геймплей
    gameplay_duration_seconds: int = 60       # 1 хвилина одного циклу
    gameplay_repeat_count: int = 45           # Повторів геймплей-макросу
    gameplay_total_minutes: int = 45          # Загалом ~45 хв

    # Рандомізація натискань
    randomize_timing: bool = True
    timing_variance_ms: int = 200             # ±200 мс
    randomize_position: bool = True
    position_variance_px: int = 5             # ±5 пікселів

    # Xbox обмеження
    xbox_session_minutes: int = 60            # Xbox дає 60 хв гри
    games_per_vpn_session: int = 2            # Скільки ігор між перезапуском VPN

    # Затримки між діями
    action_delay_ms: int = 100
    step_delay_ms: int = 50
    macro_transition_delay_ms: int = 2000     # Між різними макросами


# ============================================================================
# КОНФІГУРАЦІЯ СЕСІЇ
# ============================================================================


@dataclass
class SessionConfig:
    """
    Конфігурація повного ігрового сеансу.
    
    Потік:
    1. Запуск VPN → вибір серверу
    2. Відкрити Chrome → xbox.com/play → Fortnite
    3. Запуск макросу (гра)
    4. Вихід → перезапуск VPN → повтор
    """
    # URL для запуску Fortnite
    xbox_play_url: str = "https://www.xbox.com/en-GB/play/games/fortnite/BT5P2X999VH2"
    epic_activate_url: str = "http://www.epicgames.com/activate"

    # Режим роботи
    loop_forever: bool = True
    max_sessions: int = 0                     # 0 = безкінечно
    
    # Затримки
    session_cooldown_seconds: int = 30        # Пауза між сесіями
    vpn_restart_delay_seconds: int = 10       # Пауза після перезапуску VPN
    error_retry_delay_seconds: int = 60       # Пауза після помилки

    # Код острову Fortnite
    island_code: str = "7048-8422-2298"
    use_private_match: bool = True

    # Моніторинг
    watch_for_surveys: bool = True            # Слідкувати за опитуваннями
    auto_dismiss_surveys: bool = True         # Автоматично закривати опитування
    screenshot_on_error: bool = True          # Скріншот при помилці


# ============================================================================
# ГОЛОВНА КОНФІГУРАЦІЯ ЕМУЛЯТОРА
# ============================================================================


@dataclass
class EmulatorConfig:
    """
    Головна конфігурація, що об'єднує всі підконфігурації.
    
    Використання:
        config = EmulatorConfig()
        config.ldplayer.install_dir = r"D:\\LDPlayer9"
        config.vpn.default_region = "United States"
    """
    ldplayer: LDPlayerConfig = field(default_factory=LDPlayerConfig)
    vpn: VPNConfig = field(default_factory=VPNConfig)
    apk: APKConfig = field(default_factory=APKConfig)
    macros: MacroConfig = field(default_factory=MacroConfig)
    session: SessionConfig = field(default_factory=SessionConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Серіалізує конфігурацію в словник."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmulatorConfig':
        """Десеріалізує конфігурацію зі словника."""
        config = cls()
        if 'ldplayer' in data:
            for k, v in data['ldplayer'].items():
                if k == 'default_instance' and isinstance(v, dict):
                    for ik, iv in v.items():
                        if hasattr(config.ldplayer.default_instance, ik):
                            setattr(config.ldplayer.default_instance, ik, iv)
                elif hasattr(config.ldplayer, k):
                    setattr(config.ldplayer, k, v)
        for section_name in ('vpn', 'apk', 'macros', 'session'):
            if section_name in data:
                section = getattr(config, section_name)
                for k, v in data[section_name].items():
                    if hasattr(section, k):
                        setattr(section, k, v)
        return config

    def save(self, path: Optional[str] = None) -> bool:
        """Зберігає конфігурацію у JSON файл."""
        import json
        if path is None:
            path = os.path.join(EMULATOR_CONFIG_DIR, 'emulator_settings.json')
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    @classmethod
    def load(cls, path: Optional[str] = None) -> 'EmulatorConfig':
        """Завантажує конфігурацію з JSON файлу."""
        import json
        if path is None:
            path = os.path.join(EMULATOR_CONFIG_DIR, 'emulator_settings.json')
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return cls.from_dict(data)
        except Exception:
            pass
        return cls()


# ============================================================================
# ЗНАЧЕННЯ ЗА ЗАМОВЧУВАННЯМ
# ============================================================================

DEFAULT_EMULATOR_SETTINGS: Dict[str, Any] = {
    "ldplayer_path": r"C:\LDPlayer\LDPlayer9",
    "vpn_region": "United States",
    "island_code": "7048-8422-2298",
    "gameplay_repeat_count": 45,
    "games_per_vpn_session": 2,
    "loop_forever": True,
    "use_private_match": True,
    "auto_dismiss_surveys": True,
    "fps": 10,
    "cpu_cores": 2,
    "ram_mb": 2048,
}


# ============================================================================
# МОДЕЛІ ПРИСТРОЇВ ДЛЯ РАНДОМІЗАЦІЇ
# ============================================================================

DEVICE_MODELS: List[Dict[str, str]] = [
    {"model": "SM-T510", "manufacturer": "samsung", "name": "Galaxy Tab A 10.1"},
    {"model": "SM-T500", "manufacturer": "samsung", "name": "Galaxy Tab A7"},
    {"model": "SM-T220", "manufacturer": "samsung", "name": "Galaxy Tab A7 Lite"},
    {"model": "SM-T290", "manufacturer": "samsung", "name": "Galaxy Tab A 8.0"},
    {"model": "SM-P610", "manufacturer": "samsung", "name": "Galaxy Tab S6 Lite"},
    {"model": "Lenovo TB-X306F", "manufacturer": "lenovo", "name": "Tab M10 HD"},
    {"model": "Lenovo TB-X606F", "manufacturer": "lenovo", "name": "Tab M10 FHD Plus"},
    {"model": "Lenovo TB-8505F", "manufacturer": "lenovo", "name": "Tab M8"},
    {"model": "KFMAWI", "manufacturer": "amazon", "name": "Fire HD 10"},
    {"model": "KFONWI", "manufacturer": "amazon", "name": "Fire HD 8"},
    {"model": "Pixel C", "manufacturer": "google", "name": "Pixel C"},
    {"model": "Nexus 9", "manufacturer": "htc", "name": "Nexus 9"},
]

# Мінімальні розширення для планшетів
TABLET_RESOLUTIONS: List[Dict[str, int]] = [
    {"width": 960, "height": 540, "dpi": 160},
    {"width": 1024, "height": 600, "dpi": 160},
    {"width": 1024, "height": 768, "dpi": 160},
    {"width": 800, "height": 480, "dpi": 120},
    {"width": 960, "height": 600, "dpi": 160},
]
