"""
Bot модуль - бизнес-логика бота.

Содержит:
- logic: Основная логика работы бота
- runner: Запуск и управление ботами
- env: RL среда для обучения
"""

from .logic import BotLogic
from .runner import run_bot, BotRunner

__all__ = [
    'BotLogic',
    'run_bot',
    'BotRunner',
]
