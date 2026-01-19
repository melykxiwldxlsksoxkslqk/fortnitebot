"""
EpicBot - Cloud Gaming Automation Bot.

Модули:
    - bot_logic: Бизнес-логика бота и RL-среда
    - config: Конфигурация и константы
    - db: Работа с базой данных SQLite
    - logger: Централизованное логирование
    - main: Точка входа и основной цикл
    - security: Шифрование паролей
    - stream_input: Управление вводом для стрима
    - vision: Компьютерное зрение и распознавание
"""

__version__ = "2.0.0"
__author__ = "EpicBot Team"

# Экспортируем основные компоненты
from .logger import get_logger, setup_logging
from .config import (
    TIMEOUTS,
    VISION,
    RL,
    BROWSER,
    ASSETS,
    DEFAULT_SETTINGS,
    load_user_config,
)

__all__ = [
    # Логирование
    "get_logger",
    "setup_logging",
    # Конфигурация
    "TIMEOUTS",
    "VISION",
    "RL",
    "BROWSER",
    "ASSETS",
    "DEFAULT_SETTINGS",
    "load_user_config",
    # Версия
    "__version__",
]