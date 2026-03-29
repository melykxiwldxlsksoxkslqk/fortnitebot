"""
JSON-RPC 2.0 IPC сервер для Desktop GUI (Electron/React).

Протокол:
    - Зчитує JSON-RPC запити з stdin (по рядку)
    - Відправляє JSON-RPC відповіді у stdout
    - Підтримує notifications (events) через stdout

Команди:
    # Глобальні
    ping                     → pong
    get_version              → версія бота
    get_status               → загальний статус (інстанси, сесії, акаунти)

    # Акаунти
    get_accounts             → список акаунтів з БД
    add_account(login, pwd)  → додати акаунт
    delete_account(login)    → видалити акаунт
    import_accounts(text)    → масовий імпорт (email:password по рядках)

    # Інстанси LDPlayer
    list_instances           → список інстансів з їх станом
    setup_instance(name)     → створити + налаштувати новий інстанс
    clone_instance(src,dst)  → клонувати інстанс
    remove_instance(name)    → видалити інстанс

    # Фарм
    start_farm(name, email)  → запустити фарм на інстансі
    stop_farm(name)          → зупинити фарм на інстансі
    stop_all                 → зупинити все
    shutdown_all             → зупинити все + вимкнути емулятори

    # Налаштування
    get_settings             → поточні налаштування
    set_settings(data)       → зберегти налаштування
    get_emulator_config      → конфігурація емулятора
    set_emulator_config(d)   → зберегти конфігурацію емулятора

    # Логи
    get_recent_logs(n)       → останні n рядків логу
"""

import sys
import json
import threading
import time
import os
import io
from typing import Any, Dict, Optional, Callable, List

# IPC I/O потоки. При запуску через __main__.py вони будуть перезаписані
# на клоновані fd до того як setup_logging() зламає sys.stdout.
# Значення за замовчуванням — для тестів де stdout не зламаний.
_IPC_STDOUT: io.TextIOWrapper = sys.stdout  # type: ignore[assignment]
_IPC_STDIN: io.TextIOWrapper = sys.stdin    # type: ignore[assignment]

from ..core.logger import get_logger, setup_logging
from ..core.db import (
    init_db,
    fetch_accounts,
    add_account,
    delete_account,
    upsert_accounts,
    get_account_count,
    fetch_proxies,
    get_settings,
    set_settings,
    get_setting,
    set_setting,
)
from ..core.config import ROOT_DIR, LOGS_DIR
from ..emulator import (
    SessionOrchestrator,
    EmulatorConfig,
    AccountData,
    SessionState,
    EmulatorError,
)

logger = get_logger(__name__)


# ============================================================================
# JSON-RPC HELPERS
# ============================================================================

def _ok(result: Any, req_id: Any) -> Dict:
    """Успішна JSON-RPC відповідь."""
    return {"jsonrpc": "2.0", "result": result, "id": req_id}


def _error(code: int, message: str, req_id: Any = None, data: Any = None) -> Dict:
    """JSON-RPC помилка."""
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "error": err, "id": req_id}


def _notification(method: str, params: Any = None) -> Dict:
    """JSON-RPC notification (без id — не потребує відповіді)."""
    msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


# Error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
APP_ERROR = -32000


# ============================================================================
# IPC SERVER
# ============================================================================

class IPCServer:
    """
    JSON-RPC 2.0 сервер для зв'язку з Electron GUI.
    
    Lifecycle:
        server = IPCServer()
        server.run()  # блокуючий цикл (stdin → process → stdout)
    """

    def __init__(self) -> None:
        self._orchestrator: Optional[SessionOrchestrator] = None
        self._lock = threading.Lock()
        self._running = False
        self._event_listeners: List[Callable] = []

        # Реєструємо методи
        self._methods: Dict[str, Callable] = {
            # Global
            "ping": self._cmd_ping,
            "get_version": self._cmd_get_version,
            "get_status": self._cmd_get_status,
            # Accounts
            "get_accounts": self._cmd_get_accounts,
            "add_account": self._cmd_add_account,
            "delete_account": self._cmd_delete_account,
            "import_accounts": self._cmd_import_accounts,
            # Instances
            "list_instances": self._cmd_list_instances,
            "setup_instance": self._cmd_setup_instance,
            "clone_instance": self._cmd_clone_instance,
            "remove_instance": self._cmd_remove_instance,
            # Farm
            "start_farm": self._cmd_start_farm,
            "stop_farm": self._cmd_stop_farm,
            "stop_all": self._cmd_stop_all,
            "shutdown_all": self._cmd_shutdown_all,
            # Settings
            "get_settings": self._cmd_get_settings,
            "set_settings": self._cmd_set_settings,
            "get_emulator_config": self._cmd_get_emulator_config,
            "set_emulator_config": self._cmd_set_emulator_config,
            # Logs
            "get_recent_logs": self._cmd_get_recent_logs,
        }

    # ========================================================================
    # ORCHESTRATOR LIFECYCLE
    # ========================================================================

    def _ensure_orchestrator(self) -> SessionOrchestrator:
        """Lazy-ініціалізація оркестратора."""
        if self._orchestrator is None:
            config = EmulatorConfig.load()
            self._orchestrator = SessionOrchestrator(
                config=config,
                status_callback=self._on_status,
            )
        return self._orchestrator

    def _on_status(self, message: str) -> None:
        """Колбек статусу від оркестратора → event у GUI."""
        self._send_event("status", {"message": message, "ts": time.time()})

    def _send_event(self, event_type: str, data: Any = None) -> None:
        """Відправляє notification (event) у stdout."""
        msg = _notification(f"event.{event_type}", data)
        self._write(msg)

    # ========================================================================
    # I/O
    # ========================================================================

    def _write(self, msg: Dict) -> None:
        """Записує JSON у stdout (thread-safe). Використовує клонований fd."""
        try:
            line = json.dumps(msg, ensure_ascii=False, default=str)
            _IPC_STDOUT.write(line + "\n")
            _IPC_STDOUT.flush()
        except Exception as e:
            logger.error(f"IPC write error: {e}")

    def run(self) -> None:
        """Головний цикл: читаємо stdin, обробляємо, пишемо stdout."""
        self._running = True
        logger.info("IPC Server started (stdin/stdout JSON-RPC)")
        self._send_event("ready", {"version": "4.0.0"})

        for line in _IPC_STDIN:
            line = line.strip()
            if not line:
                continue

            try:
                request = json.loads(line)
            except json.JSONDecodeError:
                self._write(_error(PARSE_ERROR, "Invalid JSON"))
                continue

            response = self._handle(request)
            if response is not None:
                self._write(response)

        self._running = False
        logger.info("IPC Server stopped")

    def _handle(self, request: Dict) -> Optional[Dict]:
        """Обробляє один JSON-RPC запит."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if not method:
            return _error(INVALID_REQUEST, "Missing 'method'", req_id)

        handler = self._methods.get(method)
        if not handler:
            return _error(METHOD_NOT_FOUND, f"Unknown method: {method}", req_id)

        try:
            if isinstance(params, dict):
                result = handler(**params)
            elif isinstance(params, list):
                result = handler(*params)
            else:
                result = handler()
            return _ok(result, req_id)
        except TypeError as e:
            return _error(INVALID_PARAMS, str(e), req_id)
        except EmulatorError as e:
            return _error(APP_ERROR, str(e), req_id)
        except Exception as e:
            logger.exception(f"IPC handler error: {method}")
            return _error(INTERNAL_ERROR, str(e), req_id)

    # ========================================================================
    # COMMANDS — Global
    # ========================================================================

    def _cmd_ping(self) -> str:
        return "pong"

    def _cmd_get_version(self) -> Dict:
        from .. import __version__
        return {"version": __version__, "mode": "emulator"}

    def _cmd_get_status(self) -> Dict:
        orch = self._ensure_orchestrator()
        status = orch.get_status()
        status["accounts_in_db"] = get_account_count()
        return status

    # ========================================================================
    # COMMANDS — Accounts
    # ========================================================================

    def _cmd_get_accounts(self) -> List[Dict]:
        return fetch_accounts()

    def _cmd_add_account(self, login: str, password: str) -> Dict:
        ok = add_account(login, password)
        return {"success": ok, "login": login}

    def _cmd_delete_account(self, login: str) -> Dict:
        ok = delete_account(login)
        return {"success": ok, "login": login}

    def _cmd_import_accounts(self, text: str) -> Dict:
        """Масовий імпорт: кожен рядок — email:password або email|password."""
        accounts = []
        for line in text.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sep = ":" if ":" in line else "|"
            parts = line.split(sep, 1)
            if len(parts) == 2:
                accounts.append({
                    "login": parts[0].strip(),
                    "password": parts[1].strip(),
                })
        imported = upsert_accounts(accounts) if accounts else 0
        return {"imported": imported, "total_lines": len(text.strip().splitlines())}

    # ========================================================================
    # COMMANDS — Instances
    # ========================================================================

    def _cmd_list_instances(self) -> List[Dict]:
        orch = self._ensure_orchestrator()
        instances = orch.ldplayer.list_instances()
        result = []
        for inst in instances:
            d = inst.to_dict()
            # Додаємо стан фарму
            session = orch._active_sessions.get(inst.name)
            d["farm_state"] = session.state.value if session else "idle"
            result.append(d)
        return result

    def _cmd_setup_instance(self, name: str) -> Dict:
        orch = self._ensure_orchestrator()
        instance = orch.setup_instance(name)
        return instance.to_dict()

    def _cmd_clone_instance(self, source: str, new_name: str) -> Dict:
        orch = self._ensure_orchestrator()
        clone = orch.clone_and_setup(source, new_name)
        return clone.to_dict()

    def _cmd_remove_instance(self, name: str) -> Dict:
        orch = self._ensure_orchestrator()
        # Спочатку зупиняємо фарм якщо є
        orch.stop_instance(name)
        # Видаляємо інстанс
        instance = orch.ldplayer.get_instance(name)
        if instance:
            orch.ldplayer.remove_instance(instance)
            return {"success": True, "name": name}
        return {"success": False, "error": f"Instance not found: {name}"}

    # ========================================================================
    # COMMANDS — Farm
    # ========================================================================

    def _cmd_start_farm(self, instance_name: str, email: str) -> Dict:
        orch = self._ensure_orchestrator()
        # Знаходимо акаунт
        accounts = orch.account_storage.get_all_accounts()
        account = next((a for a in accounts if a.ms_email == email), None)
        if not account:
            # Створюємо мінімальний AccountData
            db_accounts = fetch_accounts()
            db_acct = next((a for a in db_accounts if a["login"] == email), None)
            if not db_acct:
                return {"success": False, "error": f"Account not found: {email}"}
            account = AccountData(
                ms_email=db_acct["login"],
                ms_password=db_acct.get("password", ""),
            )
            orch.account_storage.add_account(account)

        orch.start_farming(instance_name, account, in_background=True)
        return {"success": True, "instance": instance_name, "account": email}

    def _cmd_stop_farm(self, instance_name: str) -> Dict:
        orch = self._ensure_orchestrator()
        orch.stop_instance(instance_name)
        return {"success": True, "instance": instance_name}

    def _cmd_stop_all(self) -> Dict:
        orch = self._ensure_orchestrator()
        orch.stop_all()
        return {"success": True}

    def _cmd_shutdown_all(self) -> Dict:
        orch = self._ensure_orchestrator()
        orch.shutdown_everything()
        return {"success": True}

    # ========================================================================
    # COMMANDS — Settings
    # ========================================================================

    def _cmd_get_settings(self) -> Dict:
        return get_settings()

    def _cmd_set_settings(self, settings: Dict) -> Dict:
        count = set_settings(settings)
        return {"success": True, "updated": count}

    def _cmd_get_emulator_config(self) -> Dict:
        orch = self._ensure_orchestrator()
        return orch.config.to_dict()

    def _cmd_set_emulator_config(self, config_data: Dict) -> Dict:
        orch = self._ensure_orchestrator()
        new_config = EmulatorConfig.from_dict(config_data)
        orch._config = new_config
        new_config.save()
        return {"success": True}

    # ========================================================================
    # COMMANDS — Logs
    # ========================================================================

    def _cmd_get_recent_logs(self, count: int = 100) -> List[str]:
        """Повертає останні N рядків лог-файлу."""
        log_file = os.path.join(LOGS_DIR, "epicbot.log")
        if not os.path.exists(log_file):
            return []
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return [l.rstrip() for l in lines[-count:]]
        except Exception:
            return []


# ============================================================================
# PUBLIC API
# ============================================================================

def handle_command(request: Dict) -> Optional[Dict]:
    """Обробляє одну JSON-RPC команду (для тестування)."""
    server = IPCServer()
    return server._handle(request)


def main() -> None:
    """Точка входу IPC сервера."""
    import logging as _logging

    # !! КРИТИЧНО: stdout зайнятий JSON-RPC, тому ВСЕ логування → stderr + файл
    # Скидаємо флаг ініціалізації щоб setup_logging працювало без console
    from ..core import logger as _logmod
    _logmod._initialized = False
    _logmod._root_logger = None

    setup_logging(log_to_console=False, log_to_file=True)

    # Додаємо stderr-хендлер (замість stdout)
    root = _logging.getLogger('epicbot')
    stderr_handler = _logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(_logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    stderr_handler.setLevel(_logging.INFO)
    root.addHandler(stderr_handler)

    init_db()
    logger.info("Starting IPC Server...")
    server = IPCServer()
    server.run()
