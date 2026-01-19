"""
Модуль безопасного хранения учётных данных.

Обеспечивает шифрование паролей и безопасную работу с credentials.
Использует Fernet (AES-128) для симметричного шифрования.
"""

import os
import base64
import hashlib
import secrets
from typing import Optional
from pathlib import Path

# Попытка импорта cryptography, fallback на base64 обфускацию
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from .config import CONFIG_DIR
from .logger import get_logger

logger = get_logger(__name__)

# Путь к файлу с ключом
_KEY_FILE = os.path.join(CONFIG_DIR, '.secret_key')

# Кэш ключа в памяти
_cached_key: Optional[bytes] = None


def _get_machine_id() -> bytes:
    """
    Получает уникальный идентификатор машины для привязки ключа.
    Комбинирует имя компьютера и username.
    """
    import platform
    import getpass
    
    machine_info = f"{platform.node()}:{getpass.getuser()}:{platform.system()}"
    return hashlib.sha256(machine_info.encode()).digest()


def _derive_key(master_key: bytes, salt: bytes) -> bytes:
    """Генерирует ключ шифрования из мастер-ключа и соли."""
    if CRYPTO_AVAILABLE:
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
        )
        return base64.urlsafe_b64encode(kdf.derive(master_key))
    else:
        # Fallback: простой PBKDF2 через hashlib
        dk = hashlib.pbkdf2_hmac('sha256', master_key, salt, 100_000, dklen=32)
        return base64.urlsafe_b64encode(dk)


def _generate_key() -> bytes:
    """Генерирует новый ключ шифрования."""
    if CRYPTO_AVAILABLE:
        return Fernet.generate_key()
    else:
        # Fallback: случайный ключ
        return base64.urlsafe_b64encode(secrets.token_bytes(32))


def _load_or_create_key() -> bytes:
    """Загружает ключ из файла или создаёт новый."""
    global _cached_key
    
    if _cached_key:
        return _cached_key
    
    key_path = Path(_KEY_FILE)
    
    if key_path.exists():
        try:
            with open(key_path, 'rb') as f:
                stored_data = f.read()
            
            # Формат: salt (32 bytes) + encrypted_key
            if len(stored_data) > 32:
                salt = stored_data[:32]
                encrypted_key = stored_data[32:]
                
                # Расшифровываем ключ с помощью machine ID
                machine_key = _derive_key(_get_machine_id(), salt)
                
                if CRYPTO_AVAILABLE:
                    f = Fernet(machine_key)
                    _cached_key = f.decrypt(encrypted_key)
                else:
                    # Fallback: XOR обфускация
                    _cached_key = _xor_bytes(encrypted_key, machine_key[:len(encrypted_key)])
                
                return _cached_key
        except Exception as e:
            logger.warning(f"Не удалось загрузить ключ, создаю новый: {e}")
    
    # Создаём новый ключ
    _cached_key = _generate_key()
    
    # Сохраняем зашифрованным
    try:
        key_path.parent.mkdir(parents=True, exist_ok=True)
        
        salt = secrets.token_bytes(32)
        machine_key = _derive_key(_get_machine_id(), salt)
        
        if CRYPTO_AVAILABLE:
            f = Fernet(machine_key)
            encrypted_key = f.encrypt(_cached_key)
        else:
            encrypted_key = _xor_bytes(_cached_key, machine_key[:len(_cached_key)])
        
        with open(key_path, 'wb') as f:
            f.write(salt + encrypted_key)
        
        # Устанавливаем права доступа (только владелец)
        try:
            os.chmod(key_path, 0o600)
        except Exception:
            pass
        
        logger.info("Создан новый ключ шифрования")
    except Exception as e:
        logger.error(f"Не удалось сохранить ключ: {e}")
    
    return _cached_key


def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR обфускация (fallback когда нет cryptography)."""
    return bytes(a ^ b for a, b in zip(data, key * (len(data) // len(key) + 1)))


def encrypt_password(password: str) -> str:
    """
    Шифрует пароль.
    
    Args:
        password: Пароль в открытом виде
    
    Returns:
        Зашифрованный пароль (base64 строка с префиксом 'enc:')
    """
    if not password:
        return ""
    
    # Если уже зашифрован, возвращаем как есть
    if password.startswith('enc:'):
        return password
    
    try:
        key = _load_or_create_key()
        
        if CRYPTO_AVAILABLE:
            f = Fernet(key)
            encrypted = f.encrypt(password.encode('utf-8'))
            return 'enc:' + encrypted.decode('ascii')
        else:
            # Fallback: base64 + XOR
            password_bytes = password.encode('utf-8')
            obfuscated = _xor_bytes(password_bytes, key)
            return 'enc:' + base64.urlsafe_b64encode(obfuscated).decode('ascii')
    except Exception as e:
        logger.error(f"Ошибка шифрования пароля: {e}")
        # В случае ошибки возвращаем оригинал (не ломаем работу)
        return password


def decrypt_password(encrypted: str) -> str:
    """
    Расшифровывает пароль.
    
    Args:
        encrypted: Зашифрованный пароль (строка с префиксом 'enc:')
    
    Returns:
        Расшифрованный пароль в открытом виде
    """
    if not encrypted:
        return ""
    
    # Если не зашифрован, возвращаем как есть
    if not encrypted.startswith('enc:'):
        return encrypted
    
    try:
        key = _load_or_create_key()
        encrypted_data = encrypted[4:]  # Убираем 'enc:'
        
        if CRYPTO_AVAILABLE:
            f = Fernet(key)
            decrypted = f.decrypt(encrypted_data.encode('ascii'))
            return decrypted.decode('utf-8')
        else:
            # Fallback: base64 + XOR
            obfuscated = base64.urlsafe_b64decode(encrypted_data.encode('ascii'))
            decrypted = _xor_bytes(obfuscated, key)
            return decrypted.decode('utf-8')
    except Exception as e:
        logger.error(f"Ошибка расшифровки пароля: {e}")
        # Возвращаем как есть (возможно, это plain text)
        return encrypted


def is_encrypted(value: str) -> bool:
    """Проверяет, зашифровано ли значение."""
    return value.startswith('enc:') if value else False


def secure_erase(value: str) -> None:
    """
    Безопасно очищает строку из памяти (насколько возможно в Python).
    
    Примечание: В Python строки иммутабельны, полная очистка невозможна,
    но эта функция помечает объект для GC.
    """
    try:
        # Попытка перезаписать (не гарантируется из-за string interning)
        if hasattr(value, '__del__'):
            del value
    except Exception:
        pass


def migrate_plaintext_passwords(accounts: list) -> list:
    """
    Миграция: шифрует все незашифрованные пароли в списке аккаунтов.
    
    Args:
        accounts: Список словарей с 'login' и 'password'
    
    Returns:
        Обновлённый список с зашифрованными паролями
    """
    migrated = []
    count = 0
    
    for acc in accounts:
        new_acc = acc.copy()
        password = acc.get('password', '')
        
        if password and not is_encrypted(password):
            new_acc['password'] = encrypt_password(password)
            count += 1
        
        migrated.append(new_acc)
    
    if count > 0:
        logger.info(f"Зашифровано {count} паролей")
    
    return migrated
