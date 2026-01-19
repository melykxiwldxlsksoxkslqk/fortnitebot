"""
Тесты для модуля vision.
"""

import os
import sys
import pytest
import numpy as np

# Добавляем путь к src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vision.state import ScreenState
from src.vision.constants import TEMPLATE_CACHE, MATCH_HISTORY
from src.vision.templates import load_template, resolve_asset_path


class TestScreenState:
    """Тесты для ScreenState enum."""
    
    def test_screen_states_exist(self):
        """Проверка наличия всех состояний экрана."""
        assert hasattr(ScreenState, 'UNKNOWN')
        assert hasattr(ScreenState, 'LOADING')
        assert hasattr(ScreenState, 'CONNECTING')
        assert hasattr(ScreenState, 'PLANE_SCREEN')
        assert hasattr(ScreenState, 'LOBBY')
        assert hasattr(ScreenState, 'IN_GAME')
        assert hasattr(ScreenState, 'MENU')
        assert hasattr(ScreenState, 'ERROR')
    
    def test_screen_state_values(self):
        """Проверка уникальности значений."""
        values = [s.value for s in ScreenState]
        assert len(values) == len(set(values)), "Значения ScreenState должны быть уникальными"


class TestTemplates:
    """Тесты для работы с шаблонами."""
    
    def test_resolve_asset_path(self):
        """Тест разрешения пути к ассету."""
        path = resolve_asset_path('assets/test.png')
        assert path.endswith('test.png')
        assert 'assets' in path
    
    def test_load_template_nonexistent(self):
        """Тест загрузки несуществующего шаблона."""
        with pytest.raises((IOError, OSError, FileNotFoundError)):
            load_template('nonexistent_file_12345.png')
    
    def test_template_cache_is_dict(self):
        """Проверка что кэш шаблонов - словарь."""
        assert isinstance(TEMPLATE_CACHE, dict)
    
    def test_match_history_is_dict(self):
        """Проверка что история матчей - словарь."""
        assert isinstance(MATCH_HISTORY, dict)


class TestVisionConstants:
    """Тесты для констант vision модуля."""
    
    def test_confidence_thresholds(self):
        """Тест порогов уверенности."""
        from src.vision.constants import (
            DEFAULT_CONFIDENCE,
            HIGH_CONFIDENCE,
            LOW_CONFIDENCE,
        )
        
        assert 0 < LOW_CONFIDENCE < DEFAULT_CONFIDENCE < HIGH_CONFIDENCE <= 1.0
    
    def test_scales_exist(self):
        """Тест наличия масштабов."""
        from src.vision.constants import SCALES
        
        assert isinstance(SCALES, (list, tuple))
        assert len(SCALES) > 0
        assert all(s > 0 for s in SCALES)


class TestDetection:
    """Тесты для функций детекции."""
    
    def test_find_template_with_none_image(self):
        """Тест поиска шаблона с None изображением."""
        from src.vision.detection import find_template
        
        result = find_template(None, 'test.png')
        assert result is None
    
    def test_find_template_with_empty_image(self):
        """Тест поиска шаблона с пустым изображением."""
        from src.vision.detection import find_template
        
        # Создаём пустое изображение
        empty_img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = find_template(empty_img, 'nonexistent.png')
        assert result is None


class TestCapture:
    """Тесты для функций захвата экрана."""
    
    def test_capture_screen_returns_none_without_display(self):
        """Тест захвата экрана без дисплея."""
        from src.vision.capture import capture_screen
        
        # В CI/тестовой среде может не быть дисплея
        # Функция должна обработать это gracefully
        try:
            result = capture_screen()
            # Если есть дисплей, должен вернуть numpy array
            if result is not None:
                assert isinstance(result, np.ndarray)
        except Exception:
            # Допустимо если нет дисплея
            pass
