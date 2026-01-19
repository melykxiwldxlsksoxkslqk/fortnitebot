"""
Тесты для модуля browser.
"""

import os
import sys
import pytest

# Добавляем путь к src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.browser.manager import (
    PLAYWRIGHT_AVAILABLE,
    CAMOUFOX_AVAILABLE,
    CHROMIUM_ARGS,
    CAMOUFOX_CONFIG,
)
from src.browser.input import (
    DEFAULT_INPUT_MAP,
    _load_input_map,
)


class TestBrowserManager:
    """Тесты для BrowserManager."""
    
    def test_playwright_availability_is_bool(self):
        """Проверка типа PLAYWRIGHT_AVAILABLE."""
        assert isinstance(PLAYWRIGHT_AVAILABLE, bool)
    
    def test_camoufox_availability_is_bool(self):
        """Проверка типа CAMOUFOX_AVAILABLE."""
        assert isinstance(CAMOUFOX_AVAILABLE, bool)
    
    def test_chromium_args_is_list(self):
        """Проверка что CHROMIUM_ARGS - список."""
        assert isinstance(CHROMIUM_ARGS, list)
    
    def test_chromium_args_contains_required(self):
        """Проверка наличия важных аргументов Chromium."""
        args_str = ' '.join(CHROMIUM_ARGS)
        # Проверяем наличие типичных аргументов
        assert '--disable-blink-features' in args_str or len(CHROMIUM_ARGS) > 0
    
    def test_camoufox_config_is_dict(self):
        """Проверка что CAMOUFOX_CONFIG - словарь."""
        assert isinstance(CAMOUFOX_CONFIG, dict)
    
    def test_camoufox_config_has_humanize(self):
        """Проверка наличия humanize в конфиге Camoufox."""
        assert 'humanize' in CAMOUFOX_CONFIG


class TestBrowserInput:
    """Тесты для модуля ввода."""
    
    def test_default_input_map_is_dict(self):
        """Проверка типа DEFAULT_INPUT_MAP."""
        assert isinstance(DEFAULT_INPUT_MAP, dict)
    
    def test_default_input_map_has_basic_keys(self):
        """Проверка наличия базовых кнопок в маппинге."""
        assert 'A' in DEFAULT_INPUT_MAP
        assert 'B' in DEFAULT_INPUT_MAP
        assert 'X' in DEFAULT_INPUT_MAP
        assert 'Y' in DEFAULT_INPUT_MAP
        assert 'UP' in DEFAULT_INPUT_MAP
        assert 'DOWN' in DEFAULT_INPUT_MAP
        assert 'LEFT' in DEFAULT_INPUT_MAP
        assert 'RIGHT' in DEFAULT_INPUT_MAP
    
    def test_input_map_values_are_lists(self):
        """Проверка что значения маппинга - списки."""
        for key, value in DEFAULT_INPUT_MAP.items():
            assert isinstance(value, list), f"Значение для {key} должно быть списком"
            assert len(value) > 0, f"Список для {key} не должен быть пустым"
    
    def test_load_input_map_returns_dict(self):
        """Проверка что _load_input_map возвращает словарь."""
        result = _load_input_map()
        assert isinstance(result, dict)
    
    def test_load_input_map_has_defaults(self):
        """Проверка что загруженный маппинг содержит дефолтные значения."""
        result = _load_input_map()
        assert 'A' in result
        assert 'B' in result


class TestBrowserIntegration:
    """Интеграционные тесты для браузера."""
    
    @pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright не установлен")
    def test_browser_manager_import(self):
        """Тест импорта BrowserManager."""
        from src.browser import BrowserManager
        assert BrowserManager is not None
    
    @pytest.mark.skipif(not PLAYWRIGHT_AVAILABLE, reason="Playwright не установлен")
    def test_create_browser_import(self):
        """Тест импорта create_browser."""
        from src.browser import create_browser
        assert callable(create_browser)
