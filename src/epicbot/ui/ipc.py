"""UI IPC facade (optional, thin helper)."""
from __future__ import annotations
import json, sys
from typing import Any, Dict

class ipc:
	@staticmethod
	def send(method: str, params: Dict[str, Any] | None = None) -> None:
		payload = {"id": 0, "method": method, "params": params or {}}
		try:
			sys.stdout.write(json.dumps(payload) + "\n")
			sys.stdout.flush()
		except Exception:
			pass 