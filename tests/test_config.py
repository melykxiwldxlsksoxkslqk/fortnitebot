"""
Тесты для модуля config.py
"""

import os
import sys
import pytest

# Добавляем путь к src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import (
    TIMEOUTS,
    EMULATOR,
    DEFAULT_SETTINGS,
    validate_island_code,
    validate_email,
)


class TestConfig:
    """Тесты для конфигурации."""
    
    def test_timeouts_values(self):
        """Тест значений таймаутов."""
        assert TIMEOUTS.page_load > 0
        assert TIMEOUTS.element_wait > 0
    
    def test_emulator_config(self):
        """Тест конфигурации эмулятора."""
        assert "xbox.com" in EMULATOR.xbox_cloud_url
        assert EMULATOR.max_instances > 0
        assert EMULATOR.adb_timeout > 0
    
    def test_default_settings(self):
        """Тест значений по умолчанию."""
        assert 'island_code' in DEFAULT_SETTINGS
        assert 'log_level' in DEFAULT_SETTINGS
        assert isinstance(DEFAULT_SETTINGS['time_on_island_min'], int)


class TestValidation:
    """Тесты для функций валидации."""
    
    def test_validate_island_code_valid(self):
        """Тест валидного кода острова."""
        assert validate_island_code("1234-5678-9012") is True
        assert validate_island_code("0000-0000-0000") is True
        assert validate_island_code("9999-9999-9999") is True
    
    def test_validate_island_code_invalid(self):
        """Тест невалидного кода острова."""
        assert validate_island_code("") is False
        assert validate_island_code("1234-5678") is False
        assert validate_island_code("1234-5678-901") is False
        assert validate_island_code("1234-5678-90123") is False
        assert validate_island_code("abcd-efgh-ijkl") is False
        assert validate_island_code("123456789012") is False
    
    def test_validate_email_valid(self):
        """Тест валидного email."""
        assert validate_email("test@example.com") is True
        assert validate_email("user.name@domain.org") is True
        assert validate_email("user+tag@example.co.uk") is True
    
    def test_validate_email_invalid(self):
        """Тест невалидного email."""
        assert validate_email("") is False
        assert validate_email("not_an_email") is False
        assert validate_email("@example.com") is False
        assert validate_email("user@") is False
        assert validate_email("user@.com") is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
