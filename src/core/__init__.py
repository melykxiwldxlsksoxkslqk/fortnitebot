"""
Core модуль - базовые компоненты приложения.

Содержит:
- config: Конфигурация и константы
- logger: Централизованное логирование
- db: Работа с базой данных
- security: Шифрование и безопасность
- exceptions: Пользовательские исключения
"""

from .config import (
    ROOT_DIR,
    CONFIG_DIR,
    ASSETS_DIR,
    DEBUG_DIR,
    LOGS_DIR,
    DB_PATH,
    TIMEOUTS,
    VISION,
    RL,
    BROWSER,
    ASSETS,
    DEFAULT_SETTINGS,
    load_settings,
    load_accounts,
    load_island_code,
    save_settings,
    validate_island_code,
    validate_email,
)
from .logger import get_logger, setup_logging, LogContext
from .db import (
    init_db,
    get_connection,
    fetch_accounts,
    add_account,
    delete_account,
    fetch_proxies,
    add_proxy,
    delete_proxy,
    get_settings,
    get_setting,
    set_setting,
    set_settings,
)
from .security import encrypt_password, decrypt_password, is_encrypted
from .exceptions import (
    EpicBotError,
    BadCredentialsError,
    CodeRequiredError,
    BrowserClosedError,
    VisionError,
    TimeoutError,
    TemplateNotFoundError,
    NavigationError,
)

__all__ = [
    # Config
    'ROOT_DIR',
    'CONFIG_DIR', 
    'ASSETS_DIR',
    'DEBUG_DIR',
    'LOGS_DIR',
    'DB_PATH',
    'TIMEOUTS',
    'VISION',
    'RL',
    'BROWSER',
    'ASSETS',
    'DEFAULT_SETTINGS',
    'load_settings',
    'load_accounts',
    'load_island_code',
    'save_settings',
    'validate_island_code',
    'validate_email',
    # Logger
    'get_logger',
    'setup_logging',
    'LogContext',
    # DB
    'init_db',
    'get_connection',
    'fetch_accounts',
    'add_account',
    'delete_account',
    'fetch_proxies',
    'add_proxy',
    'delete_proxy',
    'get_settings',
    'get_setting',
    'set_setting',
    'set_settings',
    # Security
    'encrypt_password',
    'decrypt_password',
    'is_encrypted',
    # Exceptions
    'EpicBotError',
    'BadCredentialsError',
    'CodeRequiredError',
    'BrowserClosedError',
    'VisionError',
    'TimeoutError',
    'TemplateNotFoundError',
    'NavigationError',
]
