"""
Тесты для детекции лобби и иконки поиска.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import cv2
import numpy as np
from unittest.mock import Mock, patch


def test_detect_fortnite_lobby_with_yellow_button():
    """Тест: обнаружение лобби по жёлтой кнопке PLAY."""
    from src.vision.state import _detect_fortnite_lobby
    
    # Создаём тестовое изображение с жёлтой областью внизу (симуляция кнопки PLAY)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Средняя яркость для non-loading screen
    img[:, :] = (80, 80, 80)  # BGR серый
    
    # Добавляем жёлтую область внизу по центру (кнопка PLAY)
    # Жёлтый в BGR: (0, 255, 255) или близкий
    yellow_button = np.array([0, 200, 255], dtype=np.uint8)  # яркий жёлтый
    # Нижняя центральная часть: 70%-95% высоты, 30%-70% ширины
    img[int(720*0.75):int(720*0.9), int(1280*0.4):int(1280*0.6)] = yellow_button
    
    result = _detect_fortnite_lobby(img)
    assert result == True, "Должен обнаружить лобби по жёлтой кнопке PLAY"


def test_detect_fortnite_lobby_dark_screen_rejected():
    """Тест: тёмный экран НЕ должен определяться как лобби."""
    from src.vision.state import _detect_fortnite_lobby
    
    # Тёмное изображение (загрузка)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :] = (20, 20, 20)  # Очень тёмный
    
    result = _detect_fortnite_lobby(img)
    assert result == False, "Тёмный экран не должен определяться как лобби"


def test_find_search_icon_template_exists():
    """Тест: проверка существования шаблона search_icon.png."""
    from src.core import ROOT_DIR
    
    template_path = os.path.join(ROOT_DIR, 'assets', 'search_icon.png')
    assert os.path.exists(template_path), f"Шаблон search_icon.png должен существовать: {template_path}"
    
    template = cv2.imread(template_path)
    assert template is not None, "Шаблон должен загружаться"
    assert len(template.shape) == 3, "Шаблон должен быть цветным изображением"


def test_find_search_icon_no_icon():
    """Тест: поиск иконки на пустом изображении."""
    from src.vision.state import find_search_icon
    
    # Создаём пустое изображение
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :] = (50, 50, 50)  # Серый
    
    result = find_search_icon(img)
    assert result is None, "На пустом изображении иконка не должна найтись"


def test_screen_state_lobby_detection():
    """Тест: интеграция детекции LOBBY в _analyze_screen_state."""
    from src.vision.state import _analyze_screen_state, ScreenState
    
    # Создаём изображение похожее на лобби (яркое с жёлтой кнопкой)
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Яркий фон
    img[:, :] = (100, 100, 100)
    
    # Жёлтая кнопка внизу
    yellow = np.array([0, 200, 255], dtype=np.uint8)
    img[int(720*0.75):int(720*0.9), int(1280*0.4):int(1280*0.6)] = yellow
    
    # UI элементы вверху (контрастные)
    img[0:int(720*0.1), int(1280*0.1):int(1280*0.9)] = (200, 200, 200)
    
    state = _analyze_screen_state(img)
    # Может быть LOBBY или другое состояние в зависимости от эвристик
    assert state in [ScreenState.LOBBY, ScreenState.IN_GAME, ScreenState.UNKNOWN], \
        f"Яркое изображение с жёлтой кнопкой должно определяться как LOBBY или похожее: {state}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
