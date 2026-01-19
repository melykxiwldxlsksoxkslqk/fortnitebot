"""
Модуль работы с базой данных SQLite.

Обеспечивает хранение аккаунтов, прокси и настроек.
Пароли автоматически шифруются при сохранении.
"""

import os
import sqlite3
import time
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from .config import DB_PATH, CONFIG_DIR
from .logger import get_logger
from .security import encrypt_password, decrypt_password

logger = get_logger(__name__)

_DB_PATH = DB_PATH


def _ensure_dir() -> None:
    """Создаёт директорию config, если не существует."""
    os.makedirs(CONFIG_DIR, exist_ok=True)


@contextmanager
def get_connection(db_path: str = _DB_PATH):
    """
    Контекстный менеджер для безопасного подключения к БД.
    
    Usage:
        with get_connection() as conn:
            conn.execute(...)
    """
    _ensure_dir()
    conn = None
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        conn.execute('PRAGMA temp_store=MEMORY;')
        conn.execute('PRAGMA foreign_keys=ON;')
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Ошибка подключения к БД: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def init_db(db_path: str = _DB_PATH) -> None:
    """Инициализирует схему базы данных."""
    try:
        with get_connection(db_path) as conn:
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
            conn.commit()
        logger.debug("База данных инициализирована")
    except Exception as e:
        logger.error(f"Ошибка инициализации БД: {e}")
        raise


# ============================================================================
# SETTINGS
# ============================================================================

def set_settings(settings: Dict[str, Any], db_path: str = _DB_PATH) -> int:
    """Сохраняет настройки в базу данных."""
    if not settings:
        return 0
    try:
        with get_connection(db_path) as conn:
            for k, v in settings.items():
                conn.execute(
                    "INSERT INTO settings(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
                    (str(k), str(v) if v is not None else ''),
                )
            conn.commit()
        logger.debug(f"Сохранено {len(settings)} настроек")
        return len(settings)
    except Exception as e:
        logger.error(f"Ошибка сохранения настроек: {e}")
        return 0


def set_setting(key: str, value: Any, db_path: str = _DB_PATH) -> bool:
    """Сохраняет одну настройку."""
    return set_settings({key: value}, db_path) > 0


def get_settings(db_path: str = _DB_PATH) -> Dict[str, Any]:
    """Загружает все настройки из базы данных."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT key, value FROM settings")
            rows = cur.fetchall()
            return {k: v for (k, v) in rows}
    except Exception as e:
        logger.error(f"Ошибка чтения настроек: {e}")
        return {}


def get_setting(key: str, default: Any = None, db_path: str = _DB_PATH) -> Any:
    """Получает значение настройки по ключу."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT value FROM settings WHERE key=?", (key,))
            row = cur.fetchone()
            return row[0] if row and row[0] is not None else default
    except Exception as e:
        logger.error(f"Ошибка чтения настройки {key}: {e}")
        return default


# ============================================================================
# ACCOUNTS
# ============================================================================

def add_account(login: str, password: str, db_path: str = _DB_PATH) -> bool:
    """Добавляет один аккаунт."""
    return upsert_accounts([{'login': login, 'password': password}], db_path) > 0


def upsert_accounts(accounts: List[Dict], db_path: str = _DB_PATH) -> int:
    """
    Добавляет или обновляет аккаунты.
    Пароли автоматически шифруются перед сохранением.
    """
    if not accounts:
        return 0
    
    now = int(time.time())
    count = 0
    
    try:
        with get_connection(db_path) as conn:
            seen = set()
            for acc in accounts:
                login = (acc.get('login') or acc.get('email') or '').strip().lower()
                password = (acc.get('password') or '').strip()
                
                if not login:
                    continue
                if login in seen:
                    continue
                seen.add(login)
                
                # Шифруем пароль перед сохранением
                encrypted_password = encrypt_password(password) if password else ''
                
                conn.execute(
                    """
                    INSERT INTO accounts(login, password, created_at, updated_at)
                    VALUES(?, ?, ?, ?)
                    ON CONFLICT(login) DO UPDATE SET
                        password=excluded.password,
                        updated_at=excluded.updated_at
                    """,
                    (login, encrypted_password, now, now),
                )
                count += 1
            conn.commit()
        
        logger.info(f"Сохранено {count} аккаунтов")
        return count
    except Exception as e:
        logger.error(f"Ошибка сохранения аккаунтов: {e}")
        return 0


def fetch_accounts(db_path: str = _DB_PATH) -> List[Dict]:
    """
    Загружает все аккаунты из базы данных.
    Пароли автоматически расшифровываются.
    """
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT login, password FROM accounts ORDER BY id ASC")
            rows = cur.fetchall()
            
            accounts = []
            for r in rows:
                # Расшифровываем пароль при чтении
                password = decrypt_password(r[1]) if r[1] else ''
                accounts.append({'login': r[0], 'password': password})
            
            return accounts
    except Exception as e:
        logger.error(f"Ошибка загрузки аккаунтов: {e}")
        return []


def delete_account(login: str, db_path: str = _DB_PATH) -> bool:
    """Удаляет аккаунт из базы данных."""
    if not login:
        return False
    
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "DELETE FROM accounts WHERE login=?",
                (login.strip().lower(),)
            )
            conn.commit()
            deleted = cur.rowcount > 0
            if deleted:
                logger.info(f"Удалён аккаунт: {login}")
            return deleted
    except Exception as e:
        logger.error(f"Ошибка удаления аккаунта: {e}")
        return False


def get_account_count(db_path: str = _DB_PATH) -> int:
    """Возвращает количество аккаунтов в базе."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM accounts")
            row = cur.fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


# ============================================================================
# PROXIES
# ============================================================================

def add_proxy(host: str, port: str, username: str = '', password: str = '', db_path: str = _DB_PATH) -> bool:
    """Добавляет один прокси."""
    return upsert_proxies([{
        'host': host,
        'port': port,
        'username': username,
        'password': password
    }], db_path) > 0


def upsert_proxies(proxies: List[Dict], db_path: str = _DB_PATH) -> int:
    """Добавляет или обновляет прокси-серверы."""
    if not proxies:
        return 0
    
    now = int(time.time())
    count = 0
    
    try:
        with get_connection(db_path) as conn:
            for p in proxies:
                host = (p.get('host') or '').strip()
                port = (p.get('port') or '').strip()
                username = (p.get('username') or p.get('login') or '').strip()
                password = (p.get('password') or '').strip()
                
                if not host or not port:
                    continue
                
                # Шифруем пароль прокси
                encrypted_password = encrypt_password(password) if password else ''
                
                conn.execute(
                    """
                    INSERT INTO proxies(host, port, username, password, created_at, updated_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    ON CONFLICT(host, port) DO UPDATE SET
                        username=excluded.username,
                        password=excluded.password,
                        updated_at=excluded.updated_at
                    """,
                    (host, port, username, encrypted_password, now, now),
                )
                count += 1
            conn.commit()
        
        logger.info(f"Сохранено {count} прокси")
        return count
    except Exception as e:
        logger.error(f"Ошибка сохранения прокси: {e}")
        return 0


def fetch_proxies(db_path: str = _DB_PATH) -> List[Dict]:
    """Загружает все прокси-серверы из базы данных."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT host, port, username, password FROM proxies ORDER BY id ASC")
            rows = cur.fetchall()
            
            proxies = []
            for r in rows:
                # Расшифровываем пароль
                password = decrypt_password(r[3]) if r[3] else ''
                proxies.append({
                    'host': r[0],
                    'port': r[1],
                    'username': r[2] or '',
                    'password': password
                })
            
            return proxies
    except Exception as e:
        logger.error(f"Ошибка загрузки прокси: {e}")
        return []


def delete_proxy(host: str, port: str, db_path: str = _DB_PATH) -> bool:
    """Удаляет прокси из базы данных."""
    if not host or not port:
        return False
    
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "DELETE FROM proxies WHERE host=? AND port=?",
                (host.strip(), port.strip())
            )
            conn.commit()
            deleted = cur.rowcount > 0
            if deleted:
                logger.info(f"Удалён прокси: {host}:{port}")
            return deleted
    except Exception as e:
        logger.error(f"Ошибка удаления прокси: {e}")
        return False


def get_proxy_count(db_path: str = _DB_PATH) -> int:
    """Возвращает количество прокси в базе."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM proxies")
            row = cur.fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


# ============================================================================
# PROXY BINDINGS
# ============================================================================

def upsert_proxy_binding(account_login: str, proxy_host: str, proxy_port: str, db_path: str = _DB_PATH) -> bool:
    """Привязывает прокси к аккаунту (один к одному)."""
    if not account_login or not proxy_host or not proxy_port:
        return False
    
    now = int(time.time())
    try:
        with get_connection(db_path) as conn:
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
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        logger.warning(f"Прокси уже привязан к другому аккаунту: {proxy_host}:{proxy_port}")
        return False
    except Exception as e:
        logger.error(f"Ошибка привязки прокси: {e}")
        return False


def delete_proxy_binding_for_login(account_login: str, db_path: str = _DB_PATH) -> int:
    """Удаляет привязку прокси к аккаунту."""
    if not account_login:
        return 0
    
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(
                "DELETE FROM account_proxy_bindings WHERE account_login=?",
                (account_login.strip().lower(),)
            )
            conn.commit()
            return cur.rowcount or 0
    except Exception as e:
        logger.error(f"Ошибка удаления привязки прокси: {e}")
        return 0


def fetch_proxy_bindings(db_path: str = _DB_PATH) -> List[Dict]:
    """Загружает все привязки прокси к аккаунтам."""
    try:
        with get_connection(db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT account_login, proxy_host, proxy_port FROM account_proxy_bindings")
            rows = cur.fetchall()
            return [{'login': r[0], 'host': r[1], 'port': r[2]} for r in rows]
    except Exception as e:
        logger.error(f"Ошибка загрузки привязок прокси: {e}")
        return []
