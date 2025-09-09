import os
import sqlite3
import time
from typing import List, Dict, Any

_DB_PATH = os.path.join('config', 'epicbot.db')


def _ensure_dir():
	os.makedirs('config', exist_ok=True)


def get_connection(db_path: str = _DB_PATH) -> sqlite3.Connection:
	_ensure_dir()
	conn = sqlite3.connect(db_path, check_same_thread=False)
	conn.execute('PRAGMA journal_mode=WAL;')
	conn.execute('PRAGMA synchronous=NORMAL;')
	conn.execute('PRAGMA temp_store=MEMORY;')
	conn.execute('PRAGMA foreign_keys=ON;')
	return conn


def init_db(db_path: str = _DB_PATH) -> None:
	conn = get_connection(db_path)
	with conn:
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS accounts (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				login TEXT NOT NULL UNIQUE,
				password TEXT,
				created_at INTEGER,
				updated_at INTEGER
			);
			"""
		)
		conn.execute("CREATE INDEX IF NOT EXISTS idx_accounts_login ON accounts(login);")
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS proxies (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				host TEXT NOT NULL,
				port TEXT NOT NULL,
				username TEXT,
				password TEXT,
				created_at INTEGER,
				updated_at INTEGER,
				UNIQUE(host, port)
			);
			"""
		)
		conn.execute("CREATE INDEX IF NOT EXISTS idx_proxies_host_port ON proxies(host, port);")
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS settings (
				key TEXT PRIMARY KEY,
				value TEXT
			);
			"""
		)
		# Mapping: one account -> one proxy, and one proxy -> at most one account
		conn.execute(
			"""
			CREATE TABLE IF NOT EXISTS account_proxy_bindings (
				account_login TEXT NOT NULL UNIQUE,
				proxy_host TEXT NOT NULL,
				proxy_port TEXT NOT NULL,
				created_at INTEGER,
				updated_at INTEGER,
				UNIQUE(proxy_host, proxy_port),
				FOREIGN KEY (account_login) REFERENCES accounts(login) ON DELETE CASCADE
			);
			"""
		)


def set_settings(settings: Dict[str, Any], db_path: str = _DB_PATH) -> int:
	conn = get_connection(db_path)
	with conn:
		for k, v in (settings or {}).items():
			conn.execute(
				"INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
				(str(k), str(v) if v is not None else ''),
			)
	return len(settings or {})


def get_settings(db_path: str = _DB_PATH) -> Dict[str, Any]:
	conn = get_connection(db_path)
	cur = conn.cursor()
	cur.execute("SELECT key, value FROM settings")
	rows = cur.fetchall()
	return {k: v for (k, v) in rows}


def get_setting(key: str, default: Any = None, db_path: str = _DB_PATH) -> Any:
	conn = get_connection(db_path)
	cur = conn.cursor()
	cur.execute("SELECT value FROM settings WHERE key=?", (key,))
	row = cur.fetchone()
	return row[0] if row and row[0] is not None else default


def upsert_accounts(accounts: List[Dict], db_path: str = _DB_PATH) -> int:
	conn = get_connection(db_path)
	now = int(time.time())
	count = 0
	with conn:
		seen = set()
		for acc in accounts:
			login = (acc.get('login') or acc.get('email') or '').strip().lower()
			password = (acc.get('password') or '').strip()
			if not login:
				continue
			if login in seen:
				continue
			seen.add(login)
			conn.execute(
				"""
				INSERT INTO accounts(login, password, created_at, updated_at)
				VALUES(?, ?, ?, ?)
				ON CONFLICT(login) DO UPDATE SET
					password=excluded.password,
					updated_at=excluded.updated_at
				""",
				(login, password, now, now),
			)
			count += 1
	return count


def fetch_accounts(db_path: str = _DB_PATH) -> List[Dict]:
	conn = get_connection(db_path)
	cur = conn.cursor()
	cur.execute("SELECT login, password FROM accounts ORDER BY id ASC")
	rows = cur.fetchall()
	return [{'login': r[0], 'password': r[1] or ''} for r in rows]


def upsert_proxies(proxies: List[Dict], db_path: str = _DB_PATH) -> int:
	conn = get_connection(db_path)
	now = int(time.time())
	count = 0
	with conn:
		for p in proxies:
			host = (p.get('host') or '').strip()
			port = (p.get('port') or '').strip()
			username = (p.get('username') or p.get('login') or '').strip()
			password = (p.get('password') or '').strip()
			if not host or not port:
				continue
			conn.execute(
				"""
				INSERT INTO proxies(host, port, username, password, created_at, updated_at)
				VALUES(?, ?, ?, ?, ?, ?)
				ON CONFLICT(host, port) DO UPDATE SET
					username=excluded.username,
					password=excluded.password,
					updated_at=excluded.updated_at
				""",
				(host, port, username, password, now, now),
			)
			count += 1
	return count


def fetch_proxies(db_path: str = _DB_PATH) -> List[Dict]:
	conn = get_connection(db_path)
	cur = conn.cursor()
	cur.execute("SELECT host, port, username, password FROM proxies ORDER BY id ASC")
	rows = cur.fetchall()
	return [
		{
			'host': r[0],
			'port': r[1],
			'username': r[2] or '',
			'password': r[3] or ''
		}
		for r in rows
	]


# --- Proxy bindings helpers ---
def upsert_proxy_binding(account_login: str, proxy_host: str, proxy_port: str, db_path: str = _DB_PATH) -> bool:
	"""Assigns proxy to account (one-to-one). Returns True if written, False on constraint error."""
	if not account_login or not proxy_host or not proxy_port:
		return False
	conn = get_connection(db_path)
	now = int(time.time())
	try:
		with conn:
			conn.execute(
				"""
				INSERT INTO account_proxy_bindings(account_login, proxy_host, proxy_port, created_at, updated_at)
				VALUES(?, ?, ?, ?, ?)
				ON CONFLICT(account_login) DO UPDATE SET
					proxy_host=excluded.proxy_host,
					proxy_port=excluded.proxy_port,
					updated_at=excluded.updated_at
				""",
				(account_login.strip().lower(), proxy_host.strip(), proxy_port.strip(), now, now),
			)
		return True
	except sqlite3.IntegrityError:
		# Likely violates UNIQUE(proxy_host, proxy_port) because proxy already bound to someone else
		return False


def delete_proxy_binding_for_login(account_login: str, db_path: str = _DB_PATH) -> int:
	conn = get_connection(db_path)
	with conn:
		cur = conn.execute("DELETE FROM account_proxy_bindings WHERE account_login=?", (account_login.strip().lower(),))
		return cur.rowcount or 0


def fetch_proxy_bindings(db_path: str = _DB_PATH) -> List[Dict]:
	conn = get_connection(db_path)
	cur = conn.cursor()
	cur.execute("SELECT account_login, proxy_host, proxy_port FROM account_proxy_bindings")
	rows = cur.fetchall()
	return [{'login': r[0], 'host': r[1], 'port': r[2]} for r in rows] 