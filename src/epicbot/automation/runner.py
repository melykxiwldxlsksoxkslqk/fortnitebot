from __future__ import annotations
from typing import Optional, Dict, Any

try:
	from src.bot_logic import BotLogic
	except_import = None
except Exception as e:  # pragma: no cover
	except_import = e
	BotLogic = None  # type: ignore


class BotRunner:
	"""Высокоуровневый запуск ботов через BotLogic.
	Ничего не ломает; просто аккуратный интерфейс.
	"""
	def __init__(self, account: Dict[str, Any], proxy: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None):
		if BotLogic is None:
			raise RuntimeError(f"BotLogic unavailable: {except_import}")
		self.logic = BotLogic(account, proxy or {}, config or {}, update_status_callback=None)

	async def run(self) -> None:
		await self.logic.run() 