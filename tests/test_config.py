"""
Тесты для модуля config.py
"""

import os
import sys
import pytest

# Добавляем путь к src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    TIMEOUTS,
    VISION,
    RL,
    BROWSER,
    ASSETS,
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
        assert TIMEOUTS.episode_max_duration > 0
    
    def test_vision_values(self):
        """Тест значений для vision."""
        assert 0 < VISION.default_confidence <= 1.0
        assert VISION.observation_width > 0
        assert VISION.observation_height > 0
        assert len(VISION.pyramid_scales) > 0
    
    def test_rl_values(self):
        """Тест значений для RL."""
        assert RL.num_actions == 12
        assert RL.base_step_penalty < 0
        assert RL.attack_with_target_reward > 0
    
    def test_browser_config(self):
        """Тест конфигурации браузера."""
        assert "xbox.com" in BROWSER.xbox_cloud_url
        assert BROWSER.navigation_timeout > 0
    
    def test_assets_required(self):
        """Тест списка обязательных ассетов."""
        required = ASSETS.get_required()
        assert len(required) >= 4
        assert all('.png' in asset for asset in required)
    
    def test_default_settings(self):
        """Тест значений по умолчанию."""
        assert 'island_code' in DEFAULT_SETTINGS
        assert 'headless' in DEFAULT_SETTINGS
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
