"""
Тести для IPC модуля (JSON-RPC сервер).

Тестуємо handle_command з мокнутим оркестратором та БД.
"""

import os
import sys
import json
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ipc.server import IPCServer, handle_command, _ok, _error, _notification
from src.ipc.server import (
    PARSE_ERROR,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    INVALID_PARAMS,
    INTERNAL_ERROR,
    APP_ERROR,
)


# ============================================================================
# HELPERS
# ============================================================================

def rpc(method: str, params=None, req_id=1):
    """Створює JSON-RPC запит."""
    r = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        r["params"] = params
    return r


# ============================================================================
# JSON-RPC HELPERS
# ============================================================================

class TestJSONRPCHelpers:
    """Тести допоміжних функцій JSON-RPC."""

    def test_ok_response(self):
        resp = _ok("pong", 1)
        assert resp == {"jsonrpc": "2.0", "result": "pong", "id": 1}

    def test_ok_with_dict(self):
        resp = _ok({"key": "value"}, 42)
        assert resp["result"]["key"] == "value"
        assert resp["id"] == 42

    def test_error_response(self):
        resp = _error(-32601, "Method not found", 5)
        assert resp["error"]["code"] == -32601
        assert resp["error"]["message"] == "Method not found"
        assert resp["id"] == 5

    def test_error_with_data(self):
        resp = _error(-32000, "App error", 1, data={"detail": "info"})
        assert resp["error"]["data"]["detail"] == "info"

    def test_error_null_id(self):
        resp = _error(PARSE_ERROR, "Bad JSON", None)
        assert resp["id"] is None

    def test_notification(self):
        msg = _notification("event.status", {"message": "OK"})
        assert msg["method"] == "event.status"
        assert "id" not in msg
        assert msg["params"]["message"] == "OK"

    def test_notification_no_params(self):
        msg = _notification("event.ready")
        assert "params" not in msg


# ============================================================================
# IPC SERVER — _handle()
# ============================================================================

class TestIPCServerHandle:
    """Тести обробки JSON-RPC запитів."""

    def setup_method(self):
        self.server = IPCServer()

    def test_missing_method(self):
        resp = self.server._handle({"id": 1})
        assert resp["error"]["code"] == INVALID_REQUEST

    def test_unknown_method(self):
        resp = self.server._handle(rpc("nonexistent_method"))
        assert resp["error"]["code"] == METHOD_NOT_FOUND

    def test_ping(self):
        resp = self.server._handle(rpc("ping"))
        assert resp["result"] == "pong"
        assert resp["id"] == 1

    def test_get_version(self):
        resp = self.server._handle(rpc("get_version"))
        assert resp["result"]["version"] == "4.0.0"
        assert resp["result"]["mode"] == "emulator"

    def test_invalid_params_type_error(self):
        """Метод без параметрів отримав зайві."""
        # ping() не приймає аргументів — TypeError при виклику з params
        resp = self.server._handle(rpc("ping", {"unexpected": "param"}))
        assert resp["error"]["code"] == INVALID_PARAMS

    def test_dict_params(self):
        """Передача параметрів як dict."""
        resp = self.server._handle(rpc("ping", {}))
        assert resp["result"] == "pong"

    def test_list_params(self):
        """Передача параметрів як list."""
        resp = self.server._handle(rpc("ping", []))
        assert resp["result"] == "pong"

    def test_null_id_still_returns(self):
        """Запит з id=null — notification, але ми все одно повертаємо."""
        resp = self.server._handle({"jsonrpc": "2.0", "method": "ping", "id": None})
        assert resp["result"] == "pong"


# ============================================================================
# ACCOUNTS COMMANDS
# ============================================================================

class TestAccountCommands:
    """Тести команд роботи з акаунтами."""

    def setup_method(self):
        self.server = IPCServer()

    @patch("src.ipc.server.fetch_accounts")
    def test_get_accounts(self, mock_fetch):
        mock_fetch.return_value = [
            {"login": "user@test.com", "password": "enc_pass"},
        ]
        resp = self.server._handle(rpc("get_accounts"))
        assert len(resp["result"]) == 1
        assert resp["result"][0]["login"] == "user@test.com"

    @patch("src.ipc.server.add_account")
    def test_add_account(self, mock_add):
        mock_add.return_value = True
        resp = self.server._handle(rpc("add_account", {
            "login": "new@test.com",
            "password": "secret123",
        }))
        assert resp["result"]["success"] is True
        assert resp["result"]["login"] == "new@test.com"
        mock_add.assert_called_once_with("new@test.com", "secret123")

    @patch("src.ipc.server.delete_account")
    def test_delete_account(self, mock_del):
        mock_del.return_value = True
        resp = self.server._handle(rpc("delete_account", {"login": "old@test.com"}))
        assert resp["result"]["success"] is True
        mock_del.assert_called_once_with("old@test.com")

    @patch("src.ipc.server.upsert_accounts")
    def test_import_accounts(self, mock_upsert):
        mock_upsert.return_value = 3
        text = "a@test.com:pass1\nb@test.com:pass2\nc@test.com:pass3"
        resp = self.server._handle(rpc("import_accounts", {"text": text}))
        assert resp["result"]["imported"] == 3
        assert resp["result"]["total_lines"] == 3

    @patch("src.ipc.server.upsert_accounts")
    def test_import_accounts_with_comments(self, mock_upsert):
        mock_upsert.return_value = 1
        text = "# Comment line\na@test.com:pass1\n\n"
        resp = self.server._handle(rpc("import_accounts", {"text": text}))
        # Коментарі та пусті рядки пропускаються
        call_args = mock_upsert.call_args[0][0]
        assert len(call_args) == 1
        assert call_args[0]["login"] == "a@test.com"

    @patch("src.ipc.server.upsert_accounts")
    def test_import_accounts_pipe_separator(self, mock_upsert):
        mock_upsert.return_value = 1
        text = "user@test.com|password123"
        resp = self.server._handle(rpc("import_accounts", {"text": text}))
        call_args = mock_upsert.call_args[0][0]
        assert call_args[0]["login"] == "user@test.com"
        assert call_args[0]["password"] == "password123"

    @patch("src.ipc.server.upsert_accounts")
    def test_import_empty(self, mock_upsert):
        resp = self.server._handle(rpc("import_accounts", {"text": ""}))
        assert resp["result"]["imported"] == 0
        mock_upsert.assert_not_called()


# ============================================================================
# SETTINGS COMMANDS
# ============================================================================

class TestSettingsCommands:
    """Тести команд налаштувань."""

    def setup_method(self):
        self.server = IPCServer()

    @patch("src.ipc.server.get_settings")
    def test_get_settings(self, mock_get):
        mock_get.return_value = {"island_code": "1234-5678-9012", "log_level": "INFO"}
        resp = self.server._handle(rpc("get_settings"))
        assert resp["result"]["island_code"] == "1234-5678-9012"

    @patch("src.ipc.server.set_settings")
    def test_set_settings(self, mock_set):
        mock_set.return_value = 2
        resp = self.server._handle(rpc("set_settings", {
            "settings": {"island_code": "new-code", "log_level": "DEBUG"},
        }))
        assert resp["result"]["success"] is True
        assert resp["result"]["updated"] == 2


# ============================================================================
# INSTANCE COMMANDS (mocked orchestrator)
# ============================================================================

class TestInstanceCommands:
    """Тести команд роботи з інстансами."""

    def setup_method(self):
        self.server = IPCServer()
        # Mock orchestrator
        self.mock_orch = MagicMock()
        self.server._orchestrator = self.mock_orch

    def test_list_instances(self):
        mock_inst = MagicMock()
        mock_inst.name = "emulator-1"
        mock_inst.to_dict.return_value = {"name": "emulator-1", "index": 0, "status": "running"}
        self.mock_orch.ldplayer.list_instances.return_value = [mock_inst]
        self.mock_orch._active_sessions = {}

        resp = self.server._handle(rpc("list_instances"))
        assert len(resp["result"]) == 1
        assert resp["result"][0]["name"] == "emulator-1"
        assert resp["result"][0]["farm_state"] == "idle"

    def test_list_instances_with_active_session(self):
        mock_inst = MagicMock()
        mock_inst.name = "emulator-1"
        mock_inst.to_dict.return_value = {"name": "emulator-1", "index": 0, "status": "running"}
        self.mock_orch.ldplayer.list_instances.return_value = [mock_inst]

        mock_session = MagicMock()
        mock_session.state.value = "farming"
        self.mock_orch._active_sessions = {"emulator-1": mock_session}

        resp = self.server._handle(rpc("list_instances"))
        assert resp["result"][0]["farm_state"] == "farming"

    def test_setup_instance(self):
        mock_inst = MagicMock()
        mock_inst.to_dict.return_value = {"name": "new-emu", "index": 1}
        self.mock_orch.setup_instance.return_value = mock_inst

        resp = self.server._handle(rpc("setup_instance", {"name": "new-emu"}))
        assert resp["result"]["name"] == "new-emu"
        self.mock_orch.setup_instance.assert_called_once_with("new-emu")

    def test_clone_instance(self):
        mock_inst = MagicMock()
        mock_inst.to_dict.return_value = {"name": "clone-1", "index": 2}
        self.mock_orch.clone_and_setup.return_value = mock_inst

        resp = self.server._handle(rpc("clone_instance", {
            "source": "emulator-1",
            "new_name": "clone-1",
        }))
        assert resp["result"]["name"] == "clone-1"
        self.mock_orch.clone_and_setup.assert_called_once_with("emulator-1", "clone-1")

    def test_remove_instance_found(self):
        mock_inst = MagicMock()
        self.mock_orch.ldplayer.get_instance.return_value = mock_inst

        resp = self.server._handle(rpc("remove_instance", {"name": "old-emu"}))
        assert resp["result"]["success"] is True
        self.mock_orch.stop_instance.assert_called_once_with("old-emu")
        self.mock_orch.ldplayer.remove_instance.assert_called_once_with(mock_inst)

    def test_remove_instance_not_found(self):
        self.mock_orch.ldplayer.get_instance.return_value = None

        resp = self.server._handle(rpc("remove_instance", {"name": "ghost"}))
        assert resp["result"]["success"] is False


# ============================================================================
# FARM COMMANDS (mocked orchestrator)
# ============================================================================

class TestFarmCommands:
    """Тести команд фарму."""

    def setup_method(self):
        self.server = IPCServer()
        self.mock_orch = MagicMock()
        self.server._orchestrator = self.mock_orch

    @patch("src.ipc.server.fetch_accounts")
    def test_start_farm(self, mock_fetch):
        mock_fetch.return_value = [
            {"login": "user@test.com", "password": "enc_pass"},
        ]
        self.mock_orch.account_storage.get_all_accounts.return_value = []

        resp = self.server._handle(rpc("start_farm", {
            "instance_name": "emu-1",
            "email": "user@test.com",
        }))
        # Перевіримо що помилки немає
        assert "error" not in resp, f"Unexpected error: {resp.get('error')}"
        assert resp["result"]["success"] is True
        self.mock_orch.start_farming.assert_called_once()

    @patch("src.ipc.server.fetch_accounts")
    def test_start_farm_account_not_found(self, mock_fetch):
        mock_fetch.return_value = []
        self.mock_orch.account_storage.get_all_accounts.return_value = []

        resp = self.server._handle(rpc("start_farm", {
            "instance_name": "emu-1",
            "email": "nonexistent@test.com",
        }))
        assert resp["result"]["success"] is False

    def test_stop_farm(self):
        resp = self.server._handle(rpc("stop_farm", {"instance_name": "emu-1"}))
        assert resp["result"]["success"] is True
        self.mock_orch.stop_instance.assert_called_once_with("emu-1")

    def test_stop_all(self):
        resp = self.server._handle(rpc("stop_all"))
        assert resp["result"]["success"] is True
        self.mock_orch.stop_all.assert_called_once()

    def test_shutdown_all(self):
        resp = self.server._handle(rpc("shutdown_all"))
        assert resp["result"]["success"] is True
        self.mock_orch.shutdown_everything.assert_called_once()


# ============================================================================
# CONFIG COMMANDS (mocked orchestrator)
# ============================================================================

class TestEmulatorConfigCommands:
    """Тести команд конфігурації емулятора."""

    def setup_method(self):
        self.server = IPCServer()
        self.mock_orch = MagicMock()
        self.server._orchestrator = self.mock_orch

    def test_get_emulator_config(self):
        self.mock_orch.config.to_dict.return_value = {
            "ldplayer": {"install_path": "C:\\LDPlayer"},
        }
        resp = self.server._handle(rpc("get_emulator_config"))
        assert resp["result"]["ldplayer"]["install_path"] == "C:\\LDPlayer"

    @patch("src.ipc.server.EmulatorConfig")
    def test_set_emulator_config(self, mock_config_class):
        mock_new_config = MagicMock()
        mock_config_class.from_dict.return_value = mock_new_config

        resp = self.server._handle(rpc("set_emulator_config", {
            "config_data": {"ldplayer": {"install_path": "D:\\LDPlayer"}},
        }))
        assert resp["result"]["success"] is True
        mock_new_config.save.assert_called_once()


# ============================================================================
# LOGS COMMAND
# ============================================================================

class TestLogCommands:
    """Тести команд логування."""

    def setup_method(self):
        self.server = IPCServer()

    @patch("src.ipc.server.os.path.exists")
    def test_get_recent_logs_no_file(self, mock_exists):
        mock_exists.return_value = False
        resp = self.server._handle(rpc("get_recent_logs"))
        assert resp["result"] == []

    @patch("builtins.open")
    @patch("src.ipc.server.os.path.exists")
    def test_get_recent_logs(self, mock_exists, mock_open):
        mock_exists.return_value = True
        lines = [f"2024-01-01 Line {i}\n" for i in range(200)]
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.readlines.return_value = lines

        resp = self.server._handle(rpc("get_recent_logs", {"count": 50}))
        assert len(resp["result"]) == 50
        # Повертаються останні 50
        assert "Line 199" in resp["result"][-1]

    @patch("builtins.open")
    @patch("src.ipc.server.os.path.exists")
    def test_get_recent_logs_default_count(self, mock_exists, mock_open):
        mock_exists.return_value = True
        lines = [f"Log line {i}\n" for i in range(10)]
        mock_open.return_value.__enter__ = lambda s: s
        mock_open.return_value.__exit__ = MagicMock(return_value=False)
        mock_open.return_value.readlines.return_value = lines

        resp = self.server._handle(rpc("get_recent_logs"))
        assert len(resp["result"]) == 10


# ============================================================================
# ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    """Тести обробки помилок."""

    def setup_method(self):
        self.server = IPCServer()

    def test_emulator_error_returns_app_error(self):
        """EmulatorError повертається як APP_ERROR."""
        from src.emulator.exceptions import EmulatorError

        with patch.object(self.server, '_methods', {
            "fail": MagicMock(side_effect=EmulatorError("LDPlayer crashed")),
        }):
            resp = self.server._handle(rpc("fail"))
            assert resp["error"]["code"] == APP_ERROR
            assert "LDPlayer crashed" in resp["error"]["message"]

    def test_generic_exception_returns_internal_error(self):
        """Довільна помилка → INTERNAL_ERROR."""
        with patch.object(self.server, '_methods', {
            "crash": MagicMock(side_effect=RuntimeError("unexpected")),
        }):
            resp = self.server._handle(rpc("crash"))
            assert resp["error"]["code"] == INTERNAL_ERROR

    def test_handle_command_standalone(self):
        """handle_command() — standalone функція для зовнішнього тестування."""
        resp = handle_command(rpc("ping"))
        assert resp["result"] == "pong"


# ============================================================================
# ALL METHODS REGISTERED
# ============================================================================

class TestMethodRegistry:
    """Перевіряємо що всі методи зареєстровані."""

    def test_all_methods_exist(self):
        server = IPCServer()
        expected = [
            "ping", "get_version", "get_status",
            "get_accounts", "add_account", "delete_account", "import_accounts",
            "list_instances", "setup_instance", "clone_instance", "remove_instance",
            "start_farm", "stop_farm", "stop_all", "shutdown_all",
            "get_settings", "set_settings",
            "get_emulator_config", "set_emulator_config",
            "get_recent_logs",
        ]
        for method in expected:
            assert method in server._methods, f"Method '{method}' not registered"

    def test_method_count(self):
        server = IPCServer()
        assert len(server._methods) == 20
