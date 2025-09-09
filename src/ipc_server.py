import sys
import json
import asyncio
import threading
import time
from typing import Dict, List

from .bot_logic import BotLogic
from . import db as dbmod

_BOTS: List[BotLogic] = []
_THREADS: Dict[str, threading.Thread] = {}
_SETTINGS: Dict[str, object] = {}
_STATUS: Dict[str, Dict[str, object]] = {}


def _load_settings():
    global _SETTINGS
    try:
        dbmod.init_db()
        s = dbmod.get_settings()
        _SETTINGS = {
            "island_code": s.get('island_code', ""),
            "time_on_island_min": int(s.get('time_on_island_min') or 15),
            "headless": bool(int(s.get('headless', 1))),
            "appearance": s.get('appearance', "Dark"),
            "theme": s.get('theme', "blue"),
            "ingame_mode": s.get('ingame_mode', "passive"),
            "invert_bg": bool(int(s.get('invert_bg', 0))) if isinstance(s.get('invert_bg', 0), (int, str)) else bool(s.get('invert_bg', False)),
        }
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


def _update_status(login: str, text: str):
    _STATUS[login] = {"status": text, "ts": time.time()}
    try:
        sys.stdout.write(json.dumps({"event": "status", "login": login, "text": text}) + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def _status_sys(login: str, text: str):
	try:
		sys.stdout.write(json.dumps({"event": "status", "login": login or "system", "text": text}) + "\n")
		sys.stdout.flush()
	except Exception:
		pass


def start_all():
    _load_settings()
    try:
        accounts = dbmod.fetch_accounts()
    except Exception as e:
        return {"ok": False, "error": f"DB accounts: {e}"}
    # Лог в UI
    try:
        sys.stdout.write(json.dumps({"event": "status", "login": "system", "text": f"Запуск ботов: {len(accounts)} аккаунтов"}) + "\n")
        sys.stdout.flush()
    except Exception:
        pass
    try:
        proxies = dbmod.fetch_proxies()
    except Exception:
        proxies = []
    if not accounts:
        return {"ok": False, "error": "Нет аккаунтов"}

    # биндинги
    bindings = {}
    try:
        for b in dbmod.fetch_proxy_bindings():
            bindings[b['login'].strip().lower()] = f"{b['host']}:{b['port']}"
    except Exception:
        pass
    proxies_by_key = {f"{p['host']}:{p['port']}": p for p in proxies}
    used_keys = set()
    assignments = {}
    for login, key in list(bindings.items()):
        if key in proxies_by_key:
            used_keys.add(key)
        else:
            try:
                dbmod.delete_proxy_binding_for_login(login)
            except Exception:
                pass
            bindings.pop(login, None)
    for account in accounts:
        login = (account.get('login') or '').strip().lower()
        assigned_proxy = None
        key = bindings.get(login)
        if key and key in proxies_by_key:
            assigned_proxy = proxies_by_key[key]
            used_keys.add(key)
        else:
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

    def start_one(acc, px):
        login = (acc.get('login') or '').strip().lower()
        old = _THREADS.get(login)
        if old and old.is_alive():
            try:
                for b in list(_BOTS):
                    if (b.account or {}).get('login', '').strip().lower() == login:
                        b.request_stop()
                old.join(timeout=10)
            except Exception:
                pass
        bot = BotLogic(acc, px, _SETTINGS, _update_status)
        _BOTS.append(bot)
        # Лог старта конкретного бота
        try:
            sys.stdout.write(json.dumps({"event": "status", "login": login, "text": "Запуск..."}) + "\n")
            sys.stdout.flush()
        except Exception:
            pass

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


def stop_all():
	for b in list(_BOTS):
		try:
			b.request_stop()
		except Exception:
			pass
	# Попробуем также закрыть все активные браузеры
	try:
		from .main import close_all_active_browsers
		close_all_active_browsers()
	except Exception:
		pass
	# Дождаться завершения потоков
	for login, th in list(_THREADS.items()):
		try:
			if th and th.is_alive():
				th.join(timeout=10)
		except Exception:
			pass
		_THREADS.pop(login, None)
	return {"ok": True}


def get_status():
    # Активные логины — те, у кого есть статус или активный поток
    active = set(_STATUS.keys()) | set(_THREADS.keys())
    try:
        accs = dbmod.fetch_accounts()
    except Exception:
        accs = []
    return {
        "bots": [(b.account or {}).get('login', 'unknown') for b in _BOTS],
        "threads": list(_THREADS.keys()),
        "accounts": [],  # не засоряем дашборд всеми аккаунтами
        "accounts_all": [a.get('login') for a in accs],
        "active": list(active),
        "status": _STATUS,
        "settings": _SETTINGS,
    }


def get_settings():
    _load_settings()
    return _SETTINGS


def save_settings(payload: dict):
    _load_settings()
    s = _SETTINGS.copy()
    s.update({
        'island_code': payload.get('island_code', s['island_code']),
        'time_on_island_min': int(payload.get('time_on_island_min', s['time_on_island_min'] or 15)),
        'headless': int(bool(payload.get('headless', s['headless']))),
        'appearance': payload.get('appearance', s['appearance']),
        'theme': payload.get('theme', s['theme']),
        'ingame_mode': str(payload.get('ingame_mode', s['ingame_mode'])).strip().lower(),
        'invert_bg': int(bool(payload.get('invert_bg', s.get('invert_bg', False)))),
    })
    try:
        dbmod.set_settings(s)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True}


def signal_lobby_ready(login: str | None):
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


def get_accounts():
	try:
		dbmod.init_db()
		accs = dbmod.fetch_accounts()
		_status_sys("system", f"accounts loaded: {len(accs)}")
		return {"ok": True, "accounts": accs}
	except Exception as e:
		_status_sys("system", f"accounts load error: {e}")
		return {"ok": False, "error": str(e)}


def save_accounts(payload: dict):
	items = payload.get('accounts') or []
	try:
		dbmod.init_db()
		n = dbmod.upsert_accounts(items)
		_status_sys("system", f"accounts saved: {n}")
		return {"ok": True, "saved": n}
	except Exception as e:
		_status_sys("system", f"accounts save error: {e}")
		return {"ok": False, "error": str(e)}


def get_proxies():
	try:
		dbmod.init_db()
		px = dbmod.fetch_proxies()
		_status_sys("system", f"proxies loaded: {len(px)}")
		return {"ok": True, "proxies": px}
	except Exception as e:
		_status_sys("system", f"proxies load error: {e}")
		return {"ok": False, "error": str(e)}


def save_proxies(payload: dict):
	items = payload.get('proxies') or []
	try:
		dbmod.init_db()
		n = dbmod.upsert_proxies(items)
		_status_sys("system", f"proxies saved: {n}")
		return {"ok": True, "saved": n}
	except Exception as e:
		_status_sys("system", f"proxies save error: {e}")
		return {"ok": False, "error": str(e)}


_METHODS = {
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


def main():
    _load_settings()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            method = _METHODS.get(req.get('method'))
            if not method:
                resp = {"id": req.get('id'), "error": "method_not_found"}
            else:
                result = method(req.get('params'))
                resp = {"id": req.get('id'), "result": result}
        except Exception as e:
            resp = {"id": None, "error": str(e)}
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == '__main__':
    main() 