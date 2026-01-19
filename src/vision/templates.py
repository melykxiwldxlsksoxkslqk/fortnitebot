"""
Работа с шаблонами изображений.
"""

import os
import cv2
import numpy as np
from typing import Tuple, Optional, Dict

from .constants import (
    ASSETS_DIR,
    ROOT_DIR,
    TEMPLATE_CACHE,
    TEMPLATE_HASH_CACHE,
)


def resolve_asset_path(relative_or_absolute: str) -> str:
    """
    Преобразует относительный путь к ассету в абсолютный.
    Поддерживает форматы:
    - 'assets/image.png'
    - 'image.png' (ищет в assets/)
    - '/absolute/path/image.png'
    """
    if os.path.isabs(relative_or_absolute):
        return relative_or_absolute
    
    # Пробуем как есть относительно ROOT_DIR
    path1 = os.path.join(ROOT_DIR, relative_or_absolute)
    if os.path.exists(path1):
        return path1
    
    # Пробуем в папке assets
    path2 = os.path.join(ASSETS_DIR, relative_or_absolute)
    if os.path.exists(path2):
        return path2
    
    # Пробуем без 'assets/' префикса
    if relative_or_absolute.startswith('assets/'):
        path3 = os.path.join(ROOT_DIR, relative_or_absolute)
        if os.path.exists(path3):
            return path3
    
    # Возвращаем первый вариант (для сообщения об ошибке)
    return path1


def _compute_template_hash(template_gray: np.ndarray) -> np.ndarray:
    """Вычисляет perceptual hash шаблона для быстрого сравнения."""
    try:
        # pHash через cv2 если доступен
        if hasattr(cv2, 'img_hash') and hasattr(cv2.img_hash, 'pHash'):
            return cv2.img_hash.pHash(template_gray)
    except Exception:
        pass
    # Фолбэк: простой средний хэш
    resized = cv2.resize(template_gray, (8, 8), interpolation=cv2.INTER_AREA)
    mean_val = resized.mean()
    return (resized > mean_val).flatten().astype(np.uint8)


def load_template(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Загружает шаблон изображения с кэшированием.
    
    Args:
        path: Путь к изображению (относительный или абсолютный)
    
    Returns:
        (template_bgr, alpha_channel) - BGR изображение и опциональный альфа-канал
    """
    resolved = resolve_asset_path(path)
    
    if resolved in TEMPLATE_CACHE:
        return TEMPLATE_CACHE[resolved]
    
    template = cv2.imread(resolved, cv2.IMREAD_UNCHANGED)
    if template is None:
        raise IOError(f"Не удалось прочитать шаблон: {resolved}")
    
    if len(template.shape) == 3 and template.shape[2] == 4:
        template_bgr = template[:, :, :3]
        alpha_channel = template[:, :, 3]
    else:
        template_bgr = template
        alpha_channel = None
    
    TEMPLATE_CACHE[resolved] = (template_bgr, alpha_channel)
    
    # Вычисляем и кэшируем хэш
    try:
        gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        TEMPLATE_HASH_CACHE[resolved] = _compute_template_hash(gray)
    except Exception:
        pass
    
    return (template_bgr, alpha_channel)


def clear_template_cache() -> None:
    """Очищает кэш шаблонов."""
    TEMPLATE_CACHE.clear()
    TEMPLATE_HASH_CACHE.clear()


def get_template_size(path: str) -> Tuple[int, int]:
    """Возвращает размер шаблона (width, height)."""
    template, _ = load_template(path)
    h, w = template.shape[:2]
    return (w, h)


def compare_templates_hash(path1: str, path2: str) -> float:
    """
    Сравнивает два шаблона по их хэшам.
    Возвращает значение от 0 (разные) до 1 (одинаковые).
    """
    resolved1 = resolve_asset_path(path1)
    resolved2 = resolve_asset_path(path2)
    
    # Загружаем если нет в кэше
    if resolved1 not in TEMPLATE_HASH_CACHE:
        load_template(path1)
    if resolved2 not in TEMPLATE_HASH_CACHE:
        load_template(path2)
    
    hash1 = TEMPLATE_HASH_CACHE.get(resolved1)
    hash2 = TEMPLATE_HASH_CACHE.get(resolved2)
    
    if hash1 is None or hash2 is None:
        return 0.0
    
    # Hamming distance
    if hasattr(cv2, 'img_hash') and hasattr(cv2.img_hash, 'pHash'):
        try:
            dist = cv2.norm(hash1, hash2, cv2.NORM_HAMMING)
            return 1.0 - (dist / 64.0)  # pHash = 64 bits
        except Exception:
            pass
    
    # Fallback: простое сравнение
    same = np.sum(hash1 == hash2)
    total = len(hash1)
    return same / total if total > 0 else 0.0


def preload_templates(*paths: str) -> int:
    """
    Предзагружает несколько шаблонов в кэш.
    Возвращает количество успешно загруженных.
    """
    loaded = 0
    for path in paths:
        try:
            load_template(path)
            loaded += 1
        except Exception:
            pass
    return loaded
