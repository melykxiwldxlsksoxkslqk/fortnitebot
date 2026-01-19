"""
Тесты для модуля db.py
"""

import os
import sys
import tempfile
import pytest

# Добавляем путь к src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import db as dbmod
from src.core.security import encrypt_password, decrypt_password, is_encrypted


class TestDatabase:
    """Тесты для модуля базы данных."""
    
    @pytest.fixture
    def temp_db(self):
        """Создаёт временную БД для тестов."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        dbmod.init_db(db_path)
        yield db_path
        
        # Очистка
        try:
            os.unlink(db_path)
        except Exception:
            pass
    
    def test_init_db(self, temp_db):
        """Тест инициализации БД."""
        # БД должна быть создана
        assert os.path.exists(temp_db)
    
    def test_settings_crud(self, temp_db):
        """Тест CRUD операций для настроек."""
        # Сохранение
        count = dbmod.set_settings({
            'island_code': '1234-5678-9012',
            'headless': 'true',
            'time_on_island_min': '30'
        }, temp_db)
        assert count == 3
        
        # Чтение всех
        settings = dbmod.get_settings(temp_db)
        assert settings['island_code'] == '1234-5678-9012'
        assert settings['headless'] == 'true'
        
        # Чтение одной
        value = dbmod.get_setting('island_code', db_path=temp_db)
        assert value == '1234-5678-9012'
        
        # Чтение несуществующей с default
        value = dbmod.get_setting('nonexistent', default='default_value', db_path=temp_db)
        assert value == 'default_value'
    
    def test_accounts_crud(self, temp_db):
        """Тест CRUD операций для аккаунтов."""
        accounts = [
            {'login': 'test1@example.com', 'password': 'pass1'},
            {'login': 'test2@example.com', 'password': 'pass2'},
        ]
        
        # Сохранение
        count = dbmod.upsert_accounts(accounts, temp_db)
        assert count == 2
        
        # Чтение
        loaded = dbmod.fetch_accounts(temp_db)
        assert len(loaded) == 2
        assert loaded[0]['login'] == 'test1@example.com'
        # Пароль должен быть расшифрован
        assert loaded[0]['password'] == 'pass1'
    
    def test_accounts_duplicate(self, temp_db):
        """Тест дедупликации аккаунтов."""
        accounts = [
            {'login': 'test@example.com', 'password': 'pass1'},
            {'login': 'test@example.com', 'password': 'pass2'},  # Дубликат
        ]
        
        count = dbmod.upsert_accounts(accounts, temp_db)
        assert count == 1  # Только уникальный
        
        loaded = dbmod.fetch_accounts(temp_db)
        assert len(loaded) == 1
    
    def test_accounts_update(self, temp_db):
        """Тест обновления аккаунта."""
        # Создаём
        dbmod.upsert_accounts([{'login': 'test@example.com', 'password': 'old_pass'}], temp_db)
        
        # Обновляем
        dbmod.upsert_accounts([{'login': 'test@example.com', 'password': 'new_pass'}], temp_db)
        
        loaded = dbmod.fetch_accounts(temp_db)
        assert len(loaded) == 1
        assert loaded[0]['password'] == 'new_pass'
    
    def test_proxies_crud(self, temp_db):
        """Тест CRUD операций для прокси."""
        proxies = [
            {'host': '192.168.1.1', 'port': '8080', 'username': 'user', 'password': 'pass'},
            {'host': '192.168.1.2', 'port': '8081'},
        ]
        
        count = dbmod.upsert_proxies(proxies, temp_db)
        assert count == 2
        
        loaded = dbmod.fetch_proxies(temp_db)
        assert len(loaded) == 2
        assert loaded[0]['host'] == '192.168.1.1'
    
    def test_proxy_bindings(self, temp_db):
        """Тест привязки прокси к аккаунтам."""
        # Создаём аккаунт
        dbmod.upsert_accounts([{'login': 'test@example.com', 'password': 'pass'}], temp_db)
        
        # Привязываем прокси
        result = dbmod.upsert_proxy_binding('test@example.com', '192.168.1.1', '8080', temp_db)
        assert result is True
        
        # Проверяем
        bindings = dbmod.fetch_proxy_bindings(temp_db)
        assert len(bindings) == 1
        assert bindings[0]['login'] == 'test@example.com'
        
        # Удаляем привязку
        deleted = dbmod.delete_proxy_binding_for_login('test@example.com', temp_db)
        assert deleted == 1
        
        bindings = dbmod.fetch_proxy_bindings(temp_db)
        assert len(bindings) == 0
    
    def test_account_count(self, temp_db):
        """Тест подсчёта аккаунтов."""
        assert dbmod.get_account_count(temp_db) == 0
        
        dbmod.upsert_accounts([{'login': 'test@example.com', 'password': 'pass'}], temp_db)
        assert dbmod.get_account_count(temp_db) == 1
    
    def test_delete_account(self, temp_db):
        """Тест удаления аккаунта."""
        dbmod.upsert_accounts([{'login': 'test@example.com', 'password': 'pass'}], temp_db)
        
        result = dbmod.delete_account('test@example.com', temp_db)
        assert result is True
        
        assert dbmod.get_account_count(temp_db) == 0


class TestSecurity:
    """Тесты для модуля безопасности."""
    
    def test_encrypt_decrypt(self):
        """Тест шифрования и расшифровки."""
        original = "my_secret_password_123!"
        
        encrypted = encrypt_password(original)
        assert encrypted != original
        assert encrypted.startswith('enc:')
        
        decrypted = decrypt_password(encrypted)
        assert decrypted == original
    
    def test_empty_password(self):
        """Тест пустого пароля."""
        assert encrypt_password("") == ""
        assert decrypt_password("") == ""
    
    def test_already_encrypted(self):
        """Тест уже зашифрованного пароля."""
        encrypted = encrypt_password("password")
        
        # Повторное шифрование не должно менять
        double_encrypted = encrypt_password(encrypted)
        assert double_encrypted == encrypted
    
    def test_plain_text_decrypt(self):
        """Тест расшифровки незашифрованного текста."""
        plain = "plain_password"
        
        # Должен вернуть как есть
        result = decrypt_password(plain)
        assert result == plain
    
    def test_is_encrypted(self):
        """Тест проверки шифрования."""
        assert is_encrypted("enc:some_data") is True
        assert is_encrypted("plain_text") is False
        assert is_encrypted("") is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
