"""
EpicBot — Fortnite XP Farm через LDPlayer + Xbox Cloud Gaming.

Архітектура:
    - core: Базові компоненти (config, logger, db, security, exceptions)
    - emulator: Автоматизація LDPlayer (ldplayer, vpn, macros, accounts, session)
    - ipc: JSON-RPC сервер для Desktop GUI (Electron/React)
"""

__version__ = "4.0.0"
__author__ = "EpicBot Team"

# === Core модуль ===
from .core import (
    # Логирование
    get_logger,
    setup_logging,
    LogContext,
    # Конфигурация
    ROOT_DIR,
    TIMEOUTS,
    EMULATOR,
    DEFAULT_SETTINGS,
    # База данных
    init_db,
    fetch_accounts,
    fetch_proxies,
    get_settings,
    set_settings,
    # Безопасность
    encrypt_password,
    decrypt_password,
    # Исключения
    EpicBotError,
    BadCredentialsError,
    NavigationError,
)

# === Emulator модуль ===
from .emulator import (
    # Конфігурація
    EmulatorConfig,
    InstanceConfig,
    VPNConfig,
    MacroConfig,
    SessionConfig,
    # LDPlayer
    LDPlayerManager,
    EmulatorInstance,
    InstanceStatus,
    # VPN
    VPNManager,
    VPNStatus,
    VPNRegion,
    # APK
    APKManager,
    APKInfo,
    APKType,
    # Акаунти
    EmulatorAccountManager,
    AccountData,
    AccountType,
    # Макроси
    MacroPlayer,
    MacroComposer,
    MacroFactory,
    MacroSequence,
    MacroStep,
    MacroAction,
    # Сесія
    SessionOrchestrator,
    GameSession,
    SessionState,
)

# === IPC модуль ===
from .ipc import (
    IPCServer,
    handle_command,
)

__all__ = [
    # Core
    "get_logger",
    "setup_logging",
    "LogContext",
    "ROOT_DIR",
    "TIMEOUTS",
    "EMULATOR",
    "DEFAULT_SETTINGS",
    "init_db",
    "fetch_accounts",
    "fetch_proxies",
    "get_settings",
    "set_settings",
    "encrypt_password",
    "decrypt_password",
    "EpicBotError",
    "BadCredentialsError",
    "NavigationError",
    # Emulator
    "EmulatorConfig",
    "InstanceConfig",
    "VPNConfig",
    "MacroConfig",
    "SessionConfig",
    "LDPlayerManager",
    "EmulatorInstance",
    "InstanceStatus",
    "VPNManager",
    "VPNStatus",
    "VPNRegion",
    "APKManager",
    "APKInfo",
    "APKType",
    "EmulatorAccountManager",
    "AccountData",
    "AccountType",
    "MacroPlayer",
    "MacroComposer",
    "MacroFactory",
    "MacroSequence",
    "MacroStep",
    "MacroAction",
    "SessionOrchestrator",
    "GameSession",
    "SessionState",
    # IPC
    "IPCServer",
    "handle_command",
    # Версия
    "__version__",
]
