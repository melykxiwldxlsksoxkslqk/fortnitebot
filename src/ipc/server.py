"""
IPC Server - JSON-RPC сервер для общения с Electron.

Обрабатывает команды от десктопного приложения.
"""

import sys
import json
import asyncio
import threading
import time
from typing import Dict, List, Optional, Any, Callable

from ..core import db as dbmod
from ..bot import BotLogic


# === Глобальное состояние ===
_BOTS: List[BotLogic] = []
_THREADS: Dict[str, threading.Thread] = {}
_SETTINGS: Dict[str, Any] = {}
_STATUS: Dict[str, Dict[str, Any]] = {}
_SETTINGS_LOADED = False


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


# === Статус и уведомления ===
def _send_event(event: str, **data):
    """Отправка события в stdout для Electron."""
    try:
        msg = {"event": event, **data}
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _update_status(login: str, text: str):
    """Обновление статуса бота и отправка в UI."""
    _STATUS[login] = {"status": text, "ts": time.time()}
    _send_event("status", login=login, text=text)


def _status_sys(login: str, text: str):
    """Системный статус."""
    _send_event("status", login=login or "system", text=text)


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
    def start_one(acc: Dict, px: Optional[Dict]):
        login = (acc.get('login') or '').strip().lower()
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
        
        # Создание и запуск нового бота
        bot = BotLogic(acc, px, _SETTINGS, _update_status)
        _BOTS.append(bot)
        _status_sys(login, "Запуск...")
        
        def run_in_loop():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(bot.run())
            finally:
                loop.close()
        
        th = threading.Thread(target=run_in_loop, daemon=True)
        th.start()
        _THREADS[login] = th
    
    for acc in accounts:
        px = assignments.get((acc.get('login') or '').strip().lower())
        start_one(acc, px)
    
    return {"ok": True, "started": len(accounts)}


def stop_all() -> Dict[str, Any]:
    """Остановка всех ботов."""
    # Остановка логики ботов
    for b in list(_BOTS):
        try:
            b.request_stop()
        except Exception:
            pass
    
    # Закрытие браузеров
    try:
        from ..main import close_all_active_browsers
        close_all_active_browsers()
    except Exception:
        pass
    
    # Ожидание завершения потоков
    for login, th in list(_THREADS.items()):
        try:
            if th and th.is_alive():
                th.join(timeout=10)
        except Exception:
            pass
        _THREADS.pop(login, None)
    
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


# === Роутинг методов ===
_METHODS: Dict[str, Callable] = {
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
}


def handle_command(req: Dict) -> Dict[str, Any]:
    """Обработка одной команды."""
    method = _METHODS.get(req.get('method'))
    if not method:
        return {"id": req.get('id'), "error": "method_not_found"}
    
    try:
        result = method(req.get('params'))
        return {"id": req.get('id'), "result": result}
    except Exception as e:
        return {"id": req.get('id'), "error": str(e)}


def main():
    """Главный цикл IPC сервера."""
    # Инициализация БД
    try:
        dbmod.init_db()
    except Exception as e:
        _status_sys("system", f"DB init warning: {e}")
    
    _load_settings()
    _status_sys("system", "IPC server ready")
    
    # Обработка команд из stdin
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        
        try:
            req = json.loads(line)
            resp = handle_command(req)
        except Exception as e:
            resp = {"id": None, "error": str(e)}
        
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == '__main__':
    main()
