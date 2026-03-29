"""
Emulator модуль — автоматизація LDPlayer для фарму Fortnite через Xbox Cloud Gaming.

Містить:
- config: Конфігурація емулятора, VPN, макросів
- ldplayer: Управління інстансами LDPlayer (створення, клонування, налаштування)
- vpn: Управління JumpJumpVPN (увімкнення/вимкнення, регіон, таймер)
- apk: Встановлення та модифікація APK (VPN, Chrome, Lucky Patcher)
- accounts: Створення та зберігання акаунтів Microsoft/Epic
- macros: Запис, відтворення та композиція макросів
- session: Оркестрація повного ігрового сеансу
- exceptions: Виключення для модуля емулятора

Архітектура (OOP):
┌─────────────────────────────────────────────────────────────┐
│  SessionOrchestrator                                        │
│  (повний цикл: VPN→гра→макрос→вихід→повтор нескінченно)    │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌───────────┐ ┌────────────┐             │
│  │ LDPlayerMgr  │ │ VPNManager│ │MacroEngine │             │
│  │ (інстанси)   │ │ (JumpVPN) │ │(запис/плей)│             │
│  └──────────────┘ └───────────┘ └────────────┘             │
│  ┌──────────────┐ ┌───────────┐                             │
│  │ AccountMgr   │ │ APKManager│                             │
│  │ (MS/Epic)    │ │(патч APK) │                             │
│  └──────────────┘ └───────────┘                             │
├─────────────────────────────────────────────────────────────┤
│  EmulatorConfig (dataclass конфігурація)                     │
│  EmulatorDatabase (SQLite зберігання стану)                  │
└─────────────────────────────────────────────────────────────┘
"""

from .config import (
    EmulatorConfig,
    InstanceConfig,
    VPNConfig,
    MacroConfig,
    SessionConfig,
    APKConfig,
    DEFAULT_EMULATOR_SETTINGS,
)
from .ldplayer import (
    EmulatorInstance,
    LDPlayerManager,
    InstanceStatus,
)
from .vpn import (
    VPNManager,
    VPNStatus,
    VPNRegion,
)
from .apk import (
    APKInfo,
    APKType,
    APKManager,
)
from .accounts import (
    AccountData,
    AccountType,
    EmulatorAccountManager,
)
from .macros import (
    MacroStep,
    MacroAction,
    MacroSequence,
    MacroComposer,
    MacroPlayer,
    MacroFactory,
)
from .session import (
    GameSession,
    SessionState,
    SessionOrchestrator,
)
from .exceptions import (
    EmulatorError,
    LDPlayerError,
    VPNError,
    MacroError,
    APKError,
    AccountCreationError,
    SessionError,
)

__all__ = [
    # Config
    'EmulatorConfig',
    'InstanceConfig',
    'VPNConfig',
    'MacroConfig',
    'SessionConfig',
    'APKConfig',
    'DEFAULT_EMULATOR_SETTINGS',
    # LDPlayer
    'EmulatorInstance',
    'LDPlayerManager',
    'InstanceStatus',
    # VPN
    'VPNManager',
    'VPNStatus',
    'VPNRegion',
    # APK
    'APKInfo',
    'APKType',
    'APKManager',
    # Accounts
    'AccountData',
    'AccountType',
    'EmulatorAccountManager',
    # Macros
    'MacroStep',
    'MacroAction',
    'MacroSequence',
    'MacroComposer',
    'MacroPlayer',
    'MacroFactory',
    # Session
    'GameSession',
    'SessionState',
    'SessionOrchestrator',
    # Exceptions
    'EmulatorError',
    'LDPlayerError',
    'VPNError',
    'MacroError',
    'APKError',
    'AccountCreationError',
    'SessionError',
]
