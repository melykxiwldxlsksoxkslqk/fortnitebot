"""
Тесты для IPC сервера.
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict


class TestUtilities:
    """Тесты утилит конвертации."""
    
    def test_to_bool_true_values(self):
        """Тест конвертации в True."""
        from src.ipc.server import _to_bool
        
        assert _to_bool(True) is True
        assert _to_bool(1) is True
        assert _to_bool("1") is True
        assert _to_bool("true") is True
        assert _to_bool("True") is True
        assert _to_bool("TRUE") is True
        assert _to_bool("yes") is True
        assert _to_bool("on") is True
    
    def test_to_bool_false_values(self):
        """Тест конвертации в False."""
        from src.ipc.server import _to_bool
        
        assert _to_bool(False) is False
        assert _to_bool(0) is False
        assert _to_bool("0") is False
        assert _to_bool("false") is False
        assert _to_bool("False") is False
        assert _to_bool("no") is False
        assert _to_bool("off") is False
        assert _to_bool("") is False
    
    def test_to_bool_default(self):
        """Тест значения по умолчанию."""
        from src.ipc.server import _to_bool
        
        assert _to_bool(None, True) is True
        assert _to_bool(None, False) is False
        assert _to_bool("invalid", True) is True
    
    def test_to_int_valid(self):
        """Тест конвертации в int."""
        from src.ipc.server import _to_int
        
        assert _to_int(42) == 42
        assert _to_int("42") == 42
        assert _to_int(3.14) == 3
        assert _to_int("3.14") == 3
    
    def test_to_int_default(self):
        """Тест значения по умолчанию для int."""
        from src.ipc.server import _to_int
        
        assert _to_int(None, 10) == 10
        assert _to_int("invalid", 5) == 5
        assert _to_int("", 0) == 0
    
    def test_to_float_valid(self):
        """Тест конвертации в float."""
        from src.ipc.server import _to_float
        
        assert _to_float(3.14) == 3.14
        assert _to_float("3.14") == 3.14
        assert _to_float(42) == 42.0
    
    def test_to_float_default(self):
        """Тест значения по умолчанию для float."""
        from src.ipc.server import _to_float
        
        assert _to_float(None, 1.5) == 1.5
        assert _to_float("invalid", 2.5) == 2.5


class TestBotMetrics:
    """Тесты класса BotMetrics."""
    
    def test_metrics_creation(self):
        """Тест создания метрик."""
        from src.ipc.server import BotMetrics
        
        metrics = BotMetrics(login="test_user")
        assert metrics.login == "test_user"
        assert metrics.errors_count == 0
        assert metrics.current_state == "idle"
    
    def test_metrics_to_dict(self):
        """Тест конвертации в словарь."""
        from src.ipc.server import BotMetrics
        
        metrics = BotMetrics(login="test", errors_count=5)
        d = metrics.to_dict()
        
        assert isinstance(d, dict)
        assert d["login"] == "test"
        assert d["errors_count"] == 5
    
    def test_metrics_uptime(self):
        """Тест расчета uptime."""
        from src.ipc.server import BotMetrics
        
        metrics = BotMetrics(login="test", started_at=time.time() - 100)
        uptime = metrics.get_uptime()
        
        assert uptime >= 100
        assert uptime < 110  # с запасом
    
    def test_metrics_uptime_not_started(self):
        """Тест uptime для незапущенного бота."""
        from src.ipc.server import BotMetrics
        
        metrics = BotMetrics(login="test")
        assert metrics.get_uptime() == 0.0
    
    def test_metrics_time_to_lobby(self):
        """Тест расчета времени до лобби."""
        from src.ipc.server import BotMetrics
        
        start = time.time()
        metrics = BotMetrics(
            login="test",
            started_at=start,
            lobby_reached_at=start + 30
        )
        
        assert metrics.get_time_to_lobby() == 30.0
    
    def test_metrics_time_to_lobby_not_reached(self):
        """Тест времени до лобби если не достигнуто."""
        from src.ipc.server import BotMetrics
        
        metrics = BotMetrics(login="test", started_at=time.time())
        assert metrics.get_time_to_lobby() is None


class TestEventType:
    """Тесты типов событий."""
    
    def test_event_types_exist(self):
        """Тест существования типов событий."""
        from src.ipc.server import EventType
        
        assert EventType.STATUS.value == "status"
        assert EventType.ERROR.value == "error"
        assert EventType.METRIC.value == "metric"
        assert EventType.DETECTION.value == "detection"


class TestEventSending:
    """Тесты отправки событий."""
    
    @patch('sys.stdout')
    def test_send_event(self, mock_stdout):
        """Тест отправки события."""
        from src.ipc.server import _send_event
        
        _send_event("test_event", data="test_data")
        
        mock_stdout.write.assert_called_once()
        call_arg = mock_stdout.write.call_args[0][0]
        event_data = json.loads(call_arg.strip())
        
        assert event_data["event"] == "test_event"
        assert event_data["data"] == "test_data"
        assert "timestamp" in event_data
    
    @patch('sys.stdout')
    def test_send_typed_event(self, mock_stdout):
        """Тест отправки типизированного события."""
        from src.ipc.server import _send_typed_event, EventType
        
        _send_typed_event(EventType.STATUS, login="test", text="Hello")
        
        mock_stdout.write.assert_called_once()
        call_arg = mock_stdout.write.call_args[0][0]
        event_data = json.loads(call_arg.strip())
        
        assert event_data["event"] == "status"
        assert event_data["login"] == "test"
        assert event_data["text"] == "Hello"


class TestCommandRouting:
    """Тесты роутинга команд."""
    
    def test_methods_dict_exists(self):
        """Тест существования словаря методов."""
        from src.ipc.server import _METHODS
        
        assert isinstance(_METHODS, dict)
        assert "start" in _METHODS
        assert "stop" in _METHODS
        assert "get_status" in _METHODS
        assert "ping" in _METHODS
    
    def test_new_methods_exist(self):
        """Тест наличия новых методов."""
        from src.ipc.server import _METHODS
        
        # Метрики
        assert "get_metrics" in _METHODS
        assert "get_bot_state" in _METHODS
        assert "reset_metrics" in _METHODS
        
        # Vision
        assert "detect_screen_state" in _METHODS
        assert "run_yolo_detection" in _METHODS
        
        # Canvas
        assert "canvas_press_button" in _METHODS
        assert "canvas_navigate" in _METHODS
        assert "canvas_get_screen_state" in _METHODS
        
        # Управление ботом
        assert "start_one" in _METHODS
        assert "stop_one" in _METHODS
        assert "restart_bot" in _METHODS
        
        # Система
        assert "get_server_info" in _METHODS
        assert "get_logs" in _METHODS


class TestHandleCommand:
    """Тесты обработки команд."""
    
    def test_handle_unknown_method(self):
        """Тест обработки неизвестного метода."""
        from src.ipc.server import handle_command
        
        result = handle_command({"id": 1, "method": "unknown_method"})
        
        assert result["id"] == 1
        assert "error" in result
        assert result["error"] == "method_not_found"
    
    def test_handle_ping(self):
        """Тест команды ping."""
        from src.ipc.server import handle_command
        
        result = handle_command({"id": 1, "method": "ping"})
        
        assert result["id"] == 1
        assert "result" in result
        assert result["result"]["ok"] is True
        assert "pong" in result["result"]
    
    def test_handle_get_server_info(self):
        """Тест получения информации о сервере."""
        from src.ipc.server import handle_command
        
        result = handle_command({"id": 2, "method": "get_server_info"})
        
        assert result["id"] == 2
        assert result["result"]["ok"] is True
        assert "info" in result["result"]
        assert "uptime" in result["result"]["info"]


class TestMetricsCommands:
    """Тесты команд метрик."""
    
    def test_get_metrics_all(self):
        """Тест получения всех метрик."""
        from src.ipc.server import get_metrics
        
        result = get_metrics()
        
        assert result["ok"] is True
        assert "metrics" in result
        assert "server_uptime" in result
    
    def test_get_metrics_specific_not_found(self):
        """Тест получения метрик несуществующего бота."""
        from src.ipc.server import get_metrics
        
        result = get_metrics({"login": "nonexistent"})
        
        assert result["ok"] is False
        assert "error" in result
    
    def test_reset_metrics_all(self):
        """Тест сброса всех метрик."""
        from src.ipc.server import reset_metrics, _METRICS, BotMetrics
        
        # Добавим метрики
        _METRICS["test1"] = BotMetrics(login="test1", errors_count=5)
        _METRICS["test2"] = BotMetrics(login="test2", errors_count=3)
        
        result = reset_metrics()
        
        assert result["ok"] is True
        assert result["reset"] == "all"
        assert len(_METRICS) == 0


class TestBotStateCommand:
    """Тесты команды get_bot_state."""
    
    def test_get_bot_state_no_login(self):
        """Тест без указания логина."""
        from src.ipc.server import get_bot_state
        
        result = get_bot_state()
        
        assert result["ok"] is False
        assert "login required" in result["error"]
    
    def test_get_bot_state_not_found(self):
        """Тест для несуществующего бота."""
        from src.ipc.server import get_bot_state
        
        result = get_bot_state({"login": "nonexistent_bot"})
        
        assert result["ok"] is False


class TestCanvasCommands:
    """Тесты canvas команд."""
    
    def test_canvas_press_button_no_login(self):
        """Тест нажатия кнопки без логина."""
        from src.ipc.server import canvas_press_button
        
        result = canvas_press_button()
        
        assert result["ok"] is False
        assert "login required" in result["error"]
    
    def test_canvas_navigate_no_login(self):
        """Тест навигации без логина."""
        from src.ipc.server import canvas_navigate
        
        result = canvas_navigate()
        
        assert result["ok"] is False
        assert "login required" in result["error"]
    
    def test_canvas_get_screen_state_no_login(self):
        """Тест состояния экрана без логина."""
        from src.ipc.server import canvas_get_screen_state
        
        result = canvas_get_screen_state()
        
        assert result["ok"] is False
        assert "login required" in result["error"]


class TestVisionCommands:
    """Тесты vision команд."""
    
    def test_detect_screen_state_no_login(self):
        """Тест детекции без логина."""
        from src.ipc.server import detect_screen_state_cmd
        
        result = detect_screen_state_cmd()
        
        assert result["ok"] is False
        assert "login required" in result["error"]
    
    def test_run_yolo_detection_no_login(self):
        """Тест YOLO детекции без логина."""
        from src.ipc.server import run_yolo_detection
        
        result = run_yolo_detection()
        
        assert result["ok"] is False
        assert "login required" in result["error"]


class TestBotManagementCommands:
    """Тесты команд управления ботами."""
    
    def test_start_one_no_login(self):
        """Тест запуска бота без логина."""
        from src.ipc.server import start_one_bot
        
        result = start_one_bot()
        
        assert result["ok"] is False
        assert "login required" in result["error"]
    
    def test_stop_one_no_login(self):
        """Тест остановки бота без логина."""
        from src.ipc.server import stop_one_bot
        
        result = stop_one_bot()
        
        assert result["ok"] is False
        assert "login required" in result["error"]
    
    def test_restart_bot_no_login(self):
        """Тест перезапуска бота без логина."""
        from src.ipc.server import restart_bot
        
        result = restart_bot()
        
        assert result["ok"] is False
        assert "login required" in result["error"]


class TestLogsCommand:
    """Тесты команды получения логов."""
    
    def test_get_logs_all(self):
        """Тест получения всех логов."""
        from src.ipc.server import get_logs
        
        result = get_logs()
        
        assert result["ok"] is True
        assert "logs" in result
    
    def test_get_logs_with_limit(self):
        """Тест получения логов с лимитом."""
        from src.ipc.server import get_logs
        
        result = get_logs({"limit": 10})
        
        assert result["ok"] is True
        assert len(result["logs"]) <= 10
    
    def test_get_logs_specific_login(self):
        """Тест получения логов конкретного бота."""
        from src.ipc.server import get_logs, _STATUS
        
        _STATUS["test_bot"] = {"status": "running", "ts": time.time()}
        
        result = get_logs({"login": "test_bot"})
        
        assert result["ok"] is True


class TestSettingsCommands:
    """Тесты команд настроек."""
    
    @patch('src.ipc.server.dbmod')
    def test_load_settings(self, mock_db):
        """Тест загрузки настроек."""
        from src.ipc.server import _load_settings, _SETTINGS, _SETTINGS_LOADED
        import src.ipc.server as server
        
        mock_db.get_settings.return_value = {
            'island_code': 'TEST-CODE',
            'time_on_island_min': 20,
        }
        
        server._SETTINGS_LOADED = False
        _load_settings(force=True)
        
        # Проверяем что настройки загружены
        assert server._SETTINGS_LOADED is True
    
    @patch('src.ipc.server.dbmod')
    def test_get_settings(self, mock_db):
        """Тест получения настроек."""
        from src.ipc.server import get_settings
        
        mock_db.get_settings.return_value = {'island_code': 'TEST'}
        
        result = get_settings()
        
        assert isinstance(result, dict)


class TestIntegration:
    """Интеграционные тесты."""
    
    def test_full_command_flow(self):
        """Тест полного потока команды."""
        from src.ipc.server import handle_command
        
        # Ping
        req = {"id": 100, "method": "ping", "params": {}}
        resp = handle_command(req)
        
        assert resp["id"] == 100
        assert resp["result"]["ok"] is True
    
    def test_command_with_params(self):
        """Тест команды с параметрами."""
        from src.ipc.server import handle_command
        
        req = {"id": 101, "method": "get_logs", "params": {"limit": 5}}
        resp = handle_command(req)
        
        assert resp["id"] == 101
        assert "result" in resp
    
    def test_json_parsing(self):
        """Тест парсинга JSON команды."""
        from src.ipc.server import handle_command
        
        json_str = '{"id": 102, "method": "ping"}'
        req = json.loads(json_str)
        resp = handle_command(req)
        
        assert resp["id"] == 102
        
        # Проверяем что ответ сериализуется
        json_resp = json.dumps(resp)
        assert isinstance(json_resp, str)
