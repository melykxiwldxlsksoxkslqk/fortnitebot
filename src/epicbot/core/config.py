"""Core configuration facades that call existing functions in src/main.py.
Не меняет исходные файлы, только предоставляет аккуратный импорт.
"""
from __future__ import annotations
from typing import Any, Dict, List

try:
	from src.main import load_settings as _load_settings
	from src.main import load_island_code as _load_island_code
	from src.main import load_accounts as _load_accounts
except Exception:  # pragma: no cover
	def _load_settings() -> Dict[str, Any]:
		return {}
	def _load_island_code() -> str:
		return ""
	def _load_accounts() -> List[Dict[str, Any]]:
		return []


def settings() -> Dict[str, Any]:
	return _load_settings()

def load_island_code() -> str:
	return _load_island_code()

def load_accounts() -> List[Dict[str, Any]]:
	return _load_accounts() 