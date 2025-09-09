"""EpicBot structured package (non-breaking).

- Не удаляет и не перемещает существующие файлы.
- Даёт аккуратные неймспейсы поверх текущих модулей.
"""
from .core.config import settings, load_island_code, load_accounts
from .automation.runner import BotRunner
from .cv.vision import vision
from .ui.ipc import ipc

__all__ = [
	"settings",
	"load_island_code",
	"load_accounts",
	"BotRunner",
	"vision",
	"ipc",
] 