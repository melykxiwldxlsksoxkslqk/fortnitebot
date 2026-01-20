"""
IPC Server - JSON-RPC сервер для общения с Electron.

Обрабатывает команды от десктопного приложения.
Поддерживает:
- Управление ботами (старт/стоп)
- Управление аккаунтами и прокси
- Настройки
- Canvas навигацию
- Vision/YOLO детекцию
- Статистику и метрики
"""

import sys
import json
import asyncio
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from ..bot.parallel import BotWorkerPool

from ..core import db as dbmod
from ..core import get_logger
from ..bot import BotLogic

logger = get_logger(__name__)


# === Типы событий ===
class EventType(str, Enum):
    """Типы событий для UI."""
    STATUS = "status"
    ERROR = "error"
    METRIC = "metric"
    BOT_STATE = "bot_state"
    DETECTION = "detection"
    SCREEN_STATE = "screen_state"
    PROGRESS = "progress"


# === Статистика бота ===
@dataclass
class BotMetrics:
    """Метрики работы бота."""
    login: str
    started_at: Optional[float] = None
    lobby_reached_at: Optional[float] = None
    island_launched_at: Optional[float] = None
    errors_count: int = 0
    reconnects_count: int = 0
    afk_actions_count: int = 0
    detections_count: int = 0
    last_detection_time: Optional[float] = None
    current_state: str = "idle"
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_uptime(self) -> float:
        """Время работы в секундах."""
        if self.started_at:
            return time.time() - self.started_at
        return 0.0
    
    def get_time_to_lobby(self) -> Optional[float]:
        """Время до лобби в секундах."""
        if self.started_at and self.lobby_reached_at:
            return self.lobby_reached_at - self.started_at
        return None


# === Глобальное состояние ===
_BOTS: List[BotLogic] = []
_THREADS: Dict[str, threading.Thread] = {}
_SETTINGS: Dict[str, Any] = {}
_STATUS: Dict[str, Dict[str, Any]] = {}
_METRICS: Dict[str, BotMetrics] = {}
_SETTINGS_LOADED = False
_START_TIME = time.time()

# Кэш для vision операций
_VISION_CACHE: Dict[str, Any] = {}
_VISION_CACHE_TTL = 1.0  # секунд


# === Утилиты конвертации ===
def _to_bool(val, default: bool = False) -> bool:
    """Конвертация значения в bool."""
    try:
        if isinstance(val, bool):
            return val
        s = str(val).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off", ""):
            return False
        return bool(int(val))
    except Exception:
        return bool(default)


def _to_int(val, default: int = 0) -> int:
    """Конвертация значения в int."""
    try:
        return int(val)
    except Exception:
        try:
            return int(float(val))
        except Exception:
            return int(default)


def _to_float(val, default: float = 0.0) -> float:
    """Конвертация значения в float."""
    try:
        return float(val)
    except Exception:
        return float(default)


# === Статус и уведомления ===
def _send_event(event: str, **data):
    """Отправка события в stdout для Electron."""
    try:
        msg = {"event": event, "timestamp": time.time(), **data}
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()
    except Exception as e:
        logger.debug(f"Ошибка отправки события: {e}")


def _send_typed_event(event_type: EventType, **data):
    """Отправка типизированного события."""
    _send_event(event_type.value, **data)


def _update_status(login: str, text: str):
    """Обновление статуса бота и отправка в UI."""
    _STATUS[login] = {"status": text, "ts": time.time()}
    _send_typed_event(EventType.STATUS, login=login, text=text)
    
    # Обновление метрик на основе статуса
    if login in _METRICS:
        _METRICS[login].current_state = text


def _update_metrics(login: str, **updates):
    """Обновление метрик бота."""
    if login not in _METRICS:
        _METRICS[login] = BotMetrics(login=login)
    
    for key, value in updates.items():
        if hasattr(_METRICS[login], key):
            setattr(_METRICS[login], key, value)
    
    _send_typed_event(EventType.METRIC, login=login, metrics=_METRICS[login].to_dict())


def _status_sys(login: str, text: str):
    """Системный статус."""
    _send_typed_event(EventType.STATUS, login=login or "system", text=text)
    logger.info(f"[{login or 'system'}] {text}")


def _send_error(login: str, error: str, details: Optional[str] = None):
    """Отправка ошибки в UI."""
    _send_typed_event(
        EventType.ERROR, 
        login=login, 
        error=error, 
        details=details,
        traceback=traceback.format_exc() if details else None
    )
    
    # Обновление метрик
    if login in _METRICS:
        _METRICS[login].errors_count += 1
        _METRICS[login].last_error = error


# === Настройки ===
def _load_settings(force: bool = False):
    """Загрузка настроек из БД."""
    global _SETTINGS, _SETTINGS_LOADED
    if _SETTINGS_LOADED and not force:
        return
    
    try:
        dbmod.init_db()
        s = dbmod.get_settings()
        _SETTINGS = {
            "island_code": s.get('island_code', ""),
            "time_on_island_min": _to_int(s.get('time_on_island_min', 15), 15),
            "headless": _to_bool(s.get('headless', 1), True),
            "appearance": s.get('appearance', "Dark"),
            "theme": s.get('theme', "blue"),
            "ingame_mode": s.get('ingame_mode', "passive"),
            "invert_bg": _to_bool(s.get('invert_bg', 0), False),
        }
        _SETTINGS_LOADED = True
    except Exception:
        _SETTINGS = {
            "island_code": "",
            "time_on_island_min": 15,
            "headless": True,
            "appearance": "Dark",
            "theme": "blue",
            "ingame_mode": "passive",
            "invert_bg": False,
        }


# === Команды ===
def start_all() -> Dict[str, Any]:
    """Запуск всех ботов."""
    _load_settings()
    
    # Загрузка аккаунтов
    try:
        accounts = dbmod.fetch_accounts()
    except Exception as e:
        return {"ok": False, "error": f"DB accounts: {e}"}
    
    _status_sys("system", f"Запуск ботов: {len(accounts)} аккаунтов")
    
    # Загрузка прокси
    try:
        proxies = dbmod.fetch_proxies()
    except Exception:
        proxies = []
    
    if not accounts:
        return {"ok": False, "error": "Нет аккаунтов"}
    
    # Загрузка биндингов прокси
    bindings = {}
    try:
        for b in dbmod.fetch_proxy_bindings():
            bindings[b['login'].strip().lower()] = f"{b['host']}:{b['port']}"
    except Exception:
        pass
    
    # Распределение прокси
    proxies_by_key = {f"{p['host']}:{p['port']}": p for p in proxies}
    used_keys = set()
    assignments = {}
    
    # Валидация существующих биндингов
    for login, key in list(bindings.items()):
        if key in proxies_by_key:
            used_keys.add(key)
        else:
            try:
                dbmod.delete_proxy_binding_for_login(login)
            except Exception:
                pass
            bindings.pop(login, None)
    
    # Назначение прокси аккаунтам
    for account in accounts:
        login = (account.get('login') or '').strip().lower()
        assigned_proxy = None
        key = bindings.get(login)
        
        if key and key in proxies_by_key:
            assigned_proxy = proxies_by_key[key]
            used_keys.add(key)
        else:
            # Найти свободный прокси
            free_key = None
            for k in proxies_by_key.keys():
                if k not in used_keys:
                    free_key = k
                    break
            
            if free_key:
                assigned_proxy = proxies_by_key[free_key]
                try:
                    dbmod.upsert_proxy_binding(login, assigned_proxy['host'], assigned_proxy['port'])
                    bindings[login] = free_key
                except Exception:
                    pass
                used_keys.add(free_key)
        
        assignments[login] = assigned_proxy
    
    # Запуск ботов
    for acc in accounts:
        px = assignments.get((acc.get('login') or '').strip().lower())
        _start_single_bot(acc, px)
    
    return {"ok": True, "started": len(accounts)}


def stop_all() -> Dict[str, Any]:
    """Остановка всех ботов."""
    _status_sys("system", "Остановка всех ботов...")
    
    # Сначала помечаем все боты на остановку
    for b in list(_BOTS):
        try:
            b.request_stop()
        except Exception:
            pass
    
    # Закрытие браузеров (это быстро прервёт все операции)
    try:
        from ..main import close_all_active_browsers
        close_all_active_browsers()
    except Exception:
        pass
    
    # Ждём потоки, но не долго (2 секунды максимум на каждый)
    for login, th in list(_THREADS.items()):
        try:
            if th and th.is_alive():
                th.join(timeout=2)
        except Exception:
            pass
        _THREADS.pop(login, None)
    
    # Очищаем список ботов
    _BOTS.clear()
    
    _status_sys("system", "Все боты остановлены")
    return {"ok": True}


def get_status() -> Dict[str, Any]:
    """Получение текущего статуса."""
    active = set(_STATUS.keys()) | set(_THREADS.keys())
    
    try:
        accs = dbmod.fetch_accounts()
    except Exception:
        accs = []
    
    return {
        "bots": [(b.account or {}).get('login', 'unknown') for b in _BOTS],
        "threads": list(_THREADS.keys()),
        "accounts": [],
        "accounts_all": [a.get('login') for a in accs],
        "active": list(active),
        "status": _STATUS,
        "settings": _SETTINGS,
    }


def get_settings() -> Dict[str, Any]:
    """Получение настроек."""
    _load_settings()
    return _SETTINGS


def save_settings(payload: Dict) -> Dict[str, Any]:
    """Сохранение настроек."""
    _load_settings()
    s = _SETTINGS.copy()
    s.update({
        'island_code': payload.get('island_code', s['island_code']),
        'time_on_island_min': _to_int(
            payload.get('time_on_island_min', s['time_on_island_min'] or 15),
            s['time_on_island_min'] or 15
        ),
        'headless': 1 if _to_bool(payload.get('headless', s['headless'])) else 0,
        'appearance': payload.get('appearance', s['appearance']),
        'theme': payload.get('theme', s['theme']),
        'ingame_mode': str(payload.get('ingame_mode', s['ingame_mode'])).strip().lower(),
        'invert_bg': 1 if _to_bool(payload.get('invert_bg', s.get('invert_bg', False))) else 0,
    })
    
    try:
        dbmod.set_settings(s)
        global _SETTINGS_LOADED
        _SETTINGS_LOADED = False
        _load_settings(force=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    
    return {"ok": True}


def signal_lobby_ready(login: Optional[str]) -> Dict[str, Any]:
    """Сигнал о готовности лобби."""
    count = 0
    for b in _BOTS:
        try:
            if login and (b.account or {}).get('login', '').strip().lower() != login.strip().lower():
                continue
            if hasattr(b, 'signal_lobby_ready'):
                b.signal_lobby_ready()
                count += 1
        except Exception:
            pass
    return {"ok": True, "signaled": count}


def get_accounts() -> Dict[str, Any]:
    """Получение списка аккаунтов."""
    try:
        accs = dbmod.fetch_accounts()
        return {"ok": True, "accounts": accs}
    except Exception as e:
        _status_sys("system", f"accounts load error: {e}")
        return {"ok": False, "error": str(e)}


def save_accounts(payload: Dict) -> Dict[str, Any]:
    """Сохранение аккаунтов."""
    items = payload.get('accounts') or []
    try:
        n = dbmod.upsert_accounts(items)
        _status_sys("system", f"accounts saved: {n}")
        return {"ok": True, "saved": n}
    except Exception as e:
        _status_sys("system", f"accounts save error: {e}")
        return {"ok": False, "error": str(e)}


def get_proxies() -> Dict[str, Any]:
    """Получение списка прокси."""
    try:
        px = dbmod.fetch_proxies()
        return {"ok": True, "proxies": px}
    except Exception as e:
        _status_sys("system", f"proxies load error: {e}")
        return {"ok": False, "error": str(e)}


def save_proxies(payload: Dict) -> Dict[str, Any]:
    """Сохранение прокси."""
    items = payload.get('proxies') or []
    try:
        n = dbmod.upsert_proxies(items)
        _status_sys("system", f"proxies saved: {n}")
        return {"ok": True, "saved": n}
    except Exception as e:
        _status_sys("system", f"proxies save error: {e}")
        return {"ok": False, "error": str(e)}


# === Новые команды: Метрики и статистика ===
def get_metrics(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Получение метрик всех ботов."""
    login = (params or {}).get('login')
    
    if login:
        if login in _METRICS:
            return {"ok": True, "metrics": _METRICS[login].to_dict()}
        return {"ok": False, "error": "Bot not found"}
    
    return {
        "ok": True,
        "metrics": {k: v.to_dict() for k, v in _METRICS.items()},
        "server_uptime": time.time() - _START_TIME,
        "active_bots": len([t for t in _THREADS.values() if t.is_alive()]),
        "total_bots": len(_BOTS),
    }


def get_bot_state(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Получение детального состояния бота."""
    login = (params or {}).get('login')
    if not login:
        return {"ok": False, "error": "login required"}
    
    login = login.strip().lower()
    
    # Найти бота
    bot = None
    for b in _BOTS:
        if (b.account or {}).get('login', '').strip().lower() == login:
            bot = b
            break
    
    if not bot:
        return {"ok": False, "error": "Bot not found"}
    
    # Собрать состояние
    thread = _THREADS.get(login)
    metrics = _METRICS.get(login)
    
    return {
        "ok": True,
        "state": {
            "login": login,
            "running": thread.is_alive() if thread else False,
            "stop_requested": bot.stop_requested,
            "status": _STATUS.get(login, {}).get("status", "unknown"),
            "metrics": metrics.to_dict() if metrics else None,
            "has_browser": bot.browser is not None,
            "manual_lobby_flag": bot.manual_lobby_event.is_set(),
        }
    }


def reset_metrics(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Сброс метрик."""
    login = (params or {}).get('login')
    
    if login:
        if login in _METRICS:
            _METRICS[login] = BotMetrics(login=login)
            return {"ok": True, "reset": login}
        return {"ok": False, "error": "Bot not found"}
    
    _METRICS.clear()
    return {"ok": True, "reset": "all"}


# === Новые команды: Vision/YOLO ===
def detect_screen_state_cmd(params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Определение состояния экрана через vision.
    
    Params:
        login: str - логин бота (для получения его страницы)
        use_yolo: bool - использовать YOLO детекцию
    """
    login = (params or {}).get('login')
    use_yolo = _to_bool((params or {}).get('use_yolo', False))
    
    if not login:
        return {"ok": False, "error": "login required"}
    
    # Найти бота и его страницу
    bot = None
    for b in _BOTS:
        if (b.account or {}).get('login', '').strip().lower() == login.strip().lower():
            bot = b
            break
    
    if not bot:
        return {"ok": False, "error": "Bot not found"}
    
    # Попробовать получить страницу из бота
    page = getattr(bot, 'page', None)
    if not page:
        return {"ok": False, "error": "No page available"}
    
    try:
        from ..vision import detect_screen_state, capture_page_bgr
        
        # Захват экрана
        image = capture_page_bgr(page)
        if image is None:
            return {"ok": False, "error": "Failed to capture screen"}
        
        # Детекция состояния
        state = detect_screen_state(image)
        
        result = {
            "ok": True,
            "state": state.value if hasattr(state, 'value') else str(state),
        }
        
        # YOLO детекция если запрошена
        if use_yolo:
            try:
                from ..vision.yolo_detector import yolo_detect_game_state
                yolo_state = yolo_detect_game_state(image)
                result["yolo_state"] = yolo_state
            except Exception as e:
                result["yolo_error"] = str(e)
        
        # Обновить метрики
        _update_metrics(login, 
            detections_count=_METRICS.get(login, BotMetrics(login=login)).detections_count + 1,
            last_detection_time=time.time()
        )
        
        return result
        
    except Exception as e:
        _send_error(login, f"Detection error: {e}")
        return {"ok": False, "error": str(e)}


def run_yolo_detection(params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Запуск YOLO детекции на текущем экране бота.
    
    Params:
        login: str - логин бота
        classes: list[str] - классы для детекции (опционально)
        confidence: float - минимальная уверенность (0.0-1.0)
    """
    login = (params or {}).get('login')
    classes = (params or {}).get('classes', [])
    confidence = _to_float((params or {}).get('confidence', 0.5), 0.5)
    
    if not login:
        return {"ok": False, "error": "login required"}
    
    # Найти бота
    bot = None
    for b in _BOTS:
        if (b.account or {}).get('login', '').strip().lower() == login.strip().lower():
            bot = b
            break
    
    if not bot:
        return {"ok": False, "error": "Bot not found"}
    
    page = getattr(bot, 'page', None)
    if not page:
        return {"ok": False, "error": "No page available"}
    
    try:
        from ..vision import capture_page_bgr
        from ..vision.yolo_detector import yolo_detect, yolo_detect_ui_elements
        
        image = capture_page_bgr(page)
        if image is None:
            return {"ok": False, "error": "Failed to capture screen"}
        
        # Детекция
        if classes:
            detections = yolo_detect(image, classes=classes, conf=confidence)
        else:
            detections = yolo_detect_ui_elements(image, conf=confidence)
        
        # Отправка события
        _send_typed_event(EventType.DETECTION, login=login, detections=detections)
        
        return {
            "ok": True,
            "detections": detections,
            "count": len(detections),
        }
        
    except Exception as e:
        _send_error(login, f"YOLO error: {e}")
        return {"ok": False, "error": str(e)}


# === Новые команды: Canvas навигация ===
def canvas_press_button(params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Нажатие кнопки геймпада через canvas navigator.
    
    Params:
        login: str - логин бота
        button: str - кнопка (A, B, X, Y, DPAD_UP, etc.)
        hold_time: float - время удержания
    """
    login = (params or {}).get('login')
    button = (params or {}).get('button', 'A')
    hold_time = _to_float((params or {}).get('hold_time', 0.1), 0.1)
    
    if not login:
        return {"ok": False, "error": "login required"}
    
    # Найти бота
    bot = None
    for b in _BOTS:
        if (b.account or {}).get('login', '').strip().lower() == login.strip().lower():
            bot = b
            break
    
    if not bot:
        return {"ok": False, "error": "Bot not found"}
    
    page = getattr(bot, 'page', None)
    if not page:
        return {"ok": False, "error": "No page available"}
    
    try:
        from ..bot.canvas import CanvasNavigator, GamepadButton
        
        # Получить или создать navigator
        navigator = getattr(bot, '_canvas_navigator', None)
        if not navigator:
            navigator = CanvasNavigator(page)
            bot._canvas_navigator = navigator
        
        # Найти кнопку
        try:
            btn = GamepadButton[button.upper()]
        except KeyError:
            return {"ok": False, "error": f"Unknown button: {button}"}
        
        # Нажать
        asyncio.get_event_loop().run_until_complete(
            navigator.press_button(btn, hold_time=hold_time)
        )
        
        return {"ok": True, "button": button}
        
    except Exception as e:
        _send_error(login, f"Canvas button error: {e}")
        return {"ok": False, "error": str(e)}


def canvas_navigate(params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Навигация в canvas.
    
    Params:
        login: str - логин бота
        direction: str - направление (UP, DOWN, LEFT, RIGHT)
        count: int - количество нажатий
    """
    login = (params or {}).get('login')
    direction = (params or {}).get('direction', 'DOWN')
    count = _to_int((params or {}).get('count', 1), 1)
    
    if not login:
        return {"ok": False, "error": "login required"}
    
    bot = None
    for b in _BOTS:
        if (b.account or {}).get('login', '').strip().lower() == login.strip().lower():
            bot = b
            break
    
    if not bot:
        return {"ok": False, "error": "Bot not found"}
    
    page = getattr(bot, 'page', None)
    if not page:
        return {"ok": False, "error": "No page available"}
    
    try:
        from ..bot.canvas import CanvasNavigator, NavigationDirection
        
        navigator = getattr(bot, '_canvas_navigator', None)
        if not navigator:
            navigator = CanvasNavigator(page)
            bot._canvas_navigator = navigator
        
        try:
            nav_dir = NavigationDirection[direction.upper()]
        except KeyError:
            return {"ok": False, "error": f"Unknown direction: {direction}"}
        
        asyncio.get_event_loop().run_until_complete(
            navigator.navigate(nav_dir, count=count)
        )
        
        return {"ok": True, "direction": direction, "count": count}
        
    except Exception as e:
        _send_error(login, f"Canvas navigate error: {e}")
        return {"ok": False, "error": str(e)}


def canvas_get_screen_state(params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Получение состояния экрана через canvas navigator.
    
    Params:
        login: str - логин бота
    """
    login = (params or {}).get('login')
    
    if not login:
        return {"ok": False, "error": "login required"}
    
    bot = None
    for b in _BOTS:
        if (b.account or {}).get('login', '').strip().lower() == login.strip().lower():
            bot = b
            break
    
    if not bot:
        return {"ok": False, "error": "Bot not found"}
    
    page = getattr(bot, 'page', None)
    if not page:
        return {"ok": False, "error": "No page available"}
    
    try:
        from ..bot.canvas import CanvasNavigator
        
        navigator = getattr(bot, '_canvas_navigator', None)
        if not navigator:
            navigator = CanvasNavigator(page)
            bot._canvas_navigator = navigator
        
        state = asyncio.get_event_loop().run_until_complete(
            navigator.detect_screen_state()
        )
        
        # Отправить событие
        _send_typed_event(EventType.SCREEN_STATE, login=login, state=state.value)
        
        return {
            "ok": True,
            "state": state.value,
            "state_name": state.name,
        }
        
    except Exception as e:
        _send_error(login, f"Canvas state error: {e}")
        return {"ok": False, "error": str(e)}


# === Новые команды: Управление отдельным ботом ===
def start_one_bot(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Запуск одного конкретного бота по логину."""
    login = (params or {}).get('login')
    if not login:
        return {"ok": False, "error": "login required"}
    
    _load_settings()
    
    try:
        accounts = dbmod.fetch_accounts()
        account = None
        for acc in accounts:
            if (acc.get('login') or '').strip().lower() == login.strip().lower():
                account = acc
                break
        
        if not account:
            return {"ok": False, "error": "Account not found"}
        
        # Найти прокси
        proxies = dbmod.fetch_proxies()
        bindings = {}
        for b in dbmod.fetch_proxy_bindings():
            bindings[b['login'].strip().lower()] = f"{b['host']}:{b['port']}"
        
        proxy = None
        key = bindings.get(login.strip().lower())
        if key:
            for p in proxies:
                if f"{p['host']}:{p['port']}" == key:
                    proxy = p
                    break
        
        # Запустить бота
        _start_single_bot(account, proxy)
        
        return {"ok": True, "started": login}
        
    except Exception as e:
        _send_error("system", f"Start bot error: {e}")
        return {"ok": False, "error": str(e)}


def stop_one_bot(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Остановка одного конкретного бота."""
    login = (params or {}).get('login')
    if not login:
        return {"ok": False, "error": "login required"}
    
    login = login.strip().lower()
    
    # Найти и остановить бота
    stopped = False
    for b in list(_BOTS):
        if (b.account or {}).get('login', '').strip().lower() == login:
            b.request_stop()
            stopped = True
    
    # Дождаться потока
    th = _THREADS.get(login)
    if th and th.is_alive():
        th.join(timeout=10)
    _THREADS.pop(login, None)
    
    if stopped:
        _status_sys(login, "Остановлен")
        return {"ok": True, "stopped": login}
    
    return {"ok": False, "error": "Bot not found or not running"}


def restart_bot(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Перезапуск бота."""
    login = (params or {}).get('login')
    if not login:
        return {"ok": False, "error": "login required"}
    
    # Остановить
    stop_result = stop_one_bot(params)
    if not stop_result.get('ok'):
        # Бота может не быть запущенным, это нормально
        pass
    
    # Небольшая пауза
    time.sleep(1)
    
    # Запустить
    return start_one_bot(params)


# === Вспомогательные функции ===
def _start_single_bot(account: Dict, proxy: Optional[Dict]):
    """Запуск одного бота."""
    login = (account.get('login') or '').strip().lower()
    old = _THREADS.get(login)
    
    # Проверка на уже запущенный поток
    if old and old.is_alive():
        _status_sys(login, "Уже запущен — пропускаю повторный старт")
        return
    
    # Остановка старого бота
    try:
        for b in list(_BOTS):
            if (b.account or {}).get('login', '').strip().lower() == login:
                b.request_stop()
        if old and old.is_alive():
            old.join(timeout=10)
    except Exception:
        pass
    
    # Создание метрик
    _METRICS[login] = BotMetrics(login=login, started_at=time.time())
    
    # Создание и запуск нового бота
    bot = BotLogic(account, proxy, _SETTINGS, _update_status)
    _BOTS.append(bot)
    _status_sys(login, "Запуск...")
    
    def run_in_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(bot.run())
        except Exception as e:
            _send_error(login, f"Bot error: {e}", str(e))
        finally:
            loop.close()
            # Обновить метрики
            if login in _METRICS:
                _METRICS[login].current_state = "stopped"
    
    th = threading.Thread(target=run_in_loop, daemon=True)
    th.start()
    _THREADS[login] = th


# === Команды системы ===
def get_server_info() -> Dict[str, Any]:
    """Информация о сервере."""
    return {
        "ok": True,
        "info": {
            "uptime": time.time() - _START_TIME,
            "started_at": _START_TIME,
            "bots_count": len(_BOTS),
            "active_threads": len([t for t in _THREADS.values() if t.is_alive()]),
            "settings_loaded": _SETTINGS_LOADED,
            "python_version": sys.version,
            "parallel_pool_active": _PARALLEL_POOL is not None,
        }
    }


def ping() -> Dict[str, Any]:
    """Проверка связи."""
    return {"ok": True, "pong": time.time()}


def get_logs(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Получение последних логов."""
    login = (params or {}).get('login')
    limit = _to_int((params or {}).get('limit', 100), 100)
    
    # Возвращаем статусы как "логи"
    if login:
        status = _STATUS.get(login.strip().lower())
        return {"ok": True, "logs": [status] if status else []}
    
    # Все статусы
    logs = sorted(_STATUS.items(), key=lambda x: x[1].get('ts', 0), reverse=True)[:limit]
    return {"ok": True, "logs": [{"login": k, **v} for k, v in logs]}


# === Команды параллельного запуска ===
_PARALLEL_POOL: Optional['BotWorkerPool'] = None


def start_parallel(params: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Запуск ботов через параллельный пул.
    
    Params:
        max_workers: int - макс. одновременных ботов (по умолчанию 5)
        logins: list[str] - список логинов для запуска (опционально, все если не указано)
        priority: str - приоритет (high, normal, low)
    """
    global _PARALLEL_POOL
    
    _load_settings()
    
    max_workers = _to_int((params or {}).get('max_workers', 5), 5)
    target_logins = (params or {}).get('logins', [])
    priority_str = (params or {}).get('priority', 'normal').lower()
    
    try:
        from ..bot.parallel import BotWorkerPool, BotPriority
        
        # Определяем приоритет
        priority_map = {
            'high': BotPriority.HIGH,
            'normal': BotPriority.NORMAL,
            'low': BotPriority.LOW,
        }
        priority = priority_map.get(priority_str, BotPriority.NORMAL)
        
        # Загружаем аккаунты
        accounts = dbmod.fetch_accounts()
        
        if target_logins:
            target_logins_lower = [l.lower() for l in target_logins]
            accounts = [a for a in accounts if a.get('login', '').lower() in target_logins_lower]
        
        if not accounts:
            return {"ok": False, "error": "Нет аккаунтов для запуска"}
        
        # Загружаем прокси
        proxies = dbmod.fetch_proxies()
        bindings = {}
        for b in dbmod.fetch_proxy_bindings():
            bindings[b['login'].strip().lower()] = f"{b['host']}:{b['port']}"
        
        proxies_by_key = {f"{p['host']}:{p['port']}": p for p in proxies}
        
        # Создаём или перезапускаем пул
        if _PARALLEL_POOL:
            _PARALLEL_POOL.stop(wait=False)
        
        _PARALLEL_POOL = BotWorkerPool(
            max_workers=max_workers,
            settings=_SETTINGS,
            status_callback=_update_status,
        )
        _PARALLEL_POOL.start()
        
        # Добавляем ботов
        for account in accounts:
            login = account.get('login', '').strip().lower()
            
            # Находим прокси
            proxy = None
            key = bindings.get(login)
            if key and key in proxies_by_key:
                proxy = proxies_by_key[key]
            
            _PARALLEL_POOL.submit(
                account=account,
                proxy=proxy,
                priority=priority,
                headless=_SETTINGS.get('headless', True),
            )
        
        _status_sys("system", f"Параллельный запуск: {len(accounts)} ботов, max_workers={max_workers}")
        
        return {
            "ok": True,
            "started": len(accounts),
            "max_workers": max_workers,
            "logins": [a.get('login') for a in accounts],
        }
        
    except Exception as e:
        logger.error(f"Parallel start error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


def stop_parallel() -> Dict[str, Any]:
    """Остановка параллельного пула."""
    global _PARALLEL_POOL
    
    if not _PARALLEL_POOL:
        return {"ok": False, "error": "Пул не запущен"}
    
    try:
        _PARALLEL_POOL.cancel_all()
        _PARALLEL_POOL.stop(wait=True, timeout=30)
        _PARALLEL_POOL = None
        
        _status_sys("system", "Параллельный пул остановлен")
        return {"ok": True}
        
    except Exception as e:
        logger.error(f"Parallel stop error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


def get_parallel_stats() -> Dict[str, Any]:
    """Получение статистики параллельного пула."""
    if not _PARALLEL_POOL:
        return {"ok": False, "error": "Пул не запущен"}
    
    try:
        stats = _PARALLEL_POOL.get_stats()
        statuses = _PARALLEL_POOL.get_all_statuses()
        
        return {
            "ok": True,
            "stats": {
                "total_tasks": stats.total_tasks,
                "completed_tasks": stats.completed_tasks,
                "failed_tasks": stats.failed_tasks,
                "active_workers": stats.active_workers,
                "queued_tasks": stats.queued_tasks,
                "average_duration": stats.average_duration,
                "total_retries": stats.total_retries,
                "started_at": stats.started_at,
            },
            "statuses": {k: v.value for k, v in statuses.items()},
        }
        
    except Exception as e:
        logger.error(f"Parallel stats error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


def set_parallel_workers(params: Optional[Dict] = None) -> Dict[str, Any]:
    """Изменение количества воркеров."""
    if not _PARALLEL_POOL:
        return {"ok": False, "error": "Пул не запущен"}
    
    max_workers = _to_int((params or {}).get('max_workers', 5), 5)
    
    try:
        # Изменение требует перезапуска пула
        _PARALLEL_POOL.max_workers = max_workers
        
        return {"ok": True, "max_workers": max_workers}
        
    except Exception as e:
        return {"ok": False, "error": str(e)}


# === Роутинг методов ===
_METHODS: Dict[str, Callable] = {
    # Основные
    "start": lambda params: start_all(),
    "stop": lambda params: stop_all(),
    "get_status": lambda params: get_status(),
    "get_settings": lambda params: get_settings(),
    "save_settings": lambda params: save_settings(params or {}),
    "signal_lobby_ready": lambda params: signal_lobby_ready((params or {}).get('login')),
    "get_accounts": lambda params: get_accounts(),
    "save_accounts": lambda params: save_accounts(params or {}),
    "get_proxies": lambda params: get_proxies(),
    "save_proxies": lambda params: save_proxies(params or {}),
    
    # Метрики
    "get_metrics": lambda params: get_metrics(params),
    "get_bot_state": lambda params: get_bot_state(params),
    "reset_metrics": lambda params: reset_metrics(params),
    
    # Vision/YOLO
    "detect_screen_state": lambda params: detect_screen_state_cmd(params),
    "run_yolo_detection": lambda params: run_yolo_detection(params),
    
    # Canvas
    "canvas_press_button": lambda params: canvas_press_button(params),
    "canvas_navigate": lambda params: canvas_navigate(params),
    "canvas_get_screen_state": lambda params: canvas_get_screen_state(params),
    
    # Управление одним ботом
    "start_one": lambda params: start_one_bot(params),
    "stop_one": lambda params: stop_one_bot(params),
    "restart_bot": lambda params: restart_bot(params),
    
    # Параллельный запуск
    "start_parallel": lambda params: start_parallel(params),
    "stop_parallel": lambda params: stop_parallel(),
    "get_parallel_stats": lambda params: get_parallel_stats(),
    "set_parallel_workers": lambda params: set_parallel_workers(params),
    
    # Система
    "get_server_info": lambda params: get_server_info(),
    "ping": lambda params: ping(),
    "get_logs": lambda params: get_logs(params),
}


def handle_command(req: Dict) -> Dict[str, Any]:
    """Обработка одной команды."""
    method_name = req.get('method')
    method = _METHODS.get(method_name)
    
    if not method:
        logger.warning(f"Unknown method: {method_name}")
        return {"id": req.get('id'), "error": "method_not_found", "method": method_name}
    
    try:
        start_time = time.time()
        result = method(req.get('params'))
        elapsed = time.time() - start_time
        
        # Логируем медленные команды
        if elapsed > 1.0:
            logger.warning(f"Slow command {method_name}: {elapsed:.2f}s")
        
        return {"id": req.get('id'), "result": result}
    except Exception as e:
        logger.error(f"Error in {method_name}: {e}", exc_info=True)
        return {"id": req.get('id'), "error": str(e), "traceback": traceback.format_exc()}


def main():
    """Главный цикл IPC сервера."""
    # Инициализация БД
    try:
        dbmod.init_db()
    except Exception as e:
        _status_sys("system", f"DB init warning: {e}")
    
    _load_settings()
    _status_sys("system", "IPC server ready")
    logger.info("IPC server started")
    
    # Обработка команд из stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            req = json.loads(line)
            resp = handle_command(req)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON: {line[:100]}")
            resp = {"id": None, "error": f"Invalid JSON: {e}"}
        except Exception as e:
            logger.error(f"Error handling command: {e}", exc_info=True)
            resp = {"id": None, "error": str(e)}
        
        try:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Error writing response: {e}")


if __name__ == '__main__':
    main()
