"""
Захват экрана и видео.
"""

import numpy as np
import cv2
from typing import Optional, Tuple

try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
except ImportError:
    PYAUTOGUI_AVAILABLE = False

# Кэш для OBS камер
_obs_captures = {}


def capture_screen(region: Tuple[int, int, int, int] = None) -> np.ndarray:
    """
    Захватывает скриншот экрана и возвращает BGR изображение.
    
    Args:
        region: (x, y, width, height) для захвата области
    
    Returns:
        BGR изображение (numpy array)
    """
    if not PYAUTOGUI_AVAILABLE:
        raise RuntimeError("pyautogui не установлен")
    
    if region:
        screenshot = pyautogui.screenshot(region=region)
    else:
        screenshot = pyautogui.screenshot()
    
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def capture_page_bgr(page, full_page: bool = False) -> np.ndarray:
    """
    Захватывает скриншот Playwright страницы и возвращает BGR изображение.
    
    Args:
        page: Playwright page object
        full_page: Захватывать всю страницу (включая скролл)
    
    Returns:
        BGR изображение (numpy array)
    """
    try:
        # Пробуем захватить весь viewport
        png_bytes = page.screenshot(type='png', timeout=5000, full_page=full_page)
        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        # Если изображение слишком маленькое, пробуем захватить экран напрямую
        if img is not None:
            h, w = img.shape[:2]
            # Минимальный размер - если меньше, используем pyautogui
            if w < 800 or h < 600:
                return capture_screen()
        
        return img
    except Exception:
        # Fallback на скриншот экрана
        return capture_screen()


def get_obs_camera(index: int = 0, preferred_size: Tuple[int, int] = (1280, 720)):
    """
    Возвращает (и кэширует) VideoCapture для OBS Virtual Camera.
    Использует DirectShow на Windows.
    """
    global _obs_captures
    if index not in _obs_captures or not _obs_captures[index].isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if preferred_size:
            w, h = preferred_size
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        # Снизим задержку: маленький буфер и целевой FPS
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_FPS, 30)
        except Exception:
            pass
        _obs_captures[index] = cap
    return _obs_captures[index]


def capture_obs_frame(index: int = 0, preferred_size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
    """
    Захватывает кадр из OBS Virtual Camera. Возвращает BGR-изображение.
    При сбое возвращается скриншот экрана как фолбэк.
    """
    cap = get_obs_camera(index, preferred_size)
    ret, frame = cap.read()
    if not ret or frame is None:
        # Фолбэк на скриншот
        frame = capture_screen()
    if preferred_size and (frame.shape[1], frame.shape[0]) != preferred_size:
        frame = cv2.resize(frame, preferred_size)
    return frame


def release_obs_cameras():
    """Освобождает все захваченные камеры."""
    global _obs_captures
    for cap in _obs_captures.values():
        try:
            cap.release()
        except Exception:
            pass
    _obs_captures.clear()


def center_brightness(frame_bgr: np.ndarray, center_frac: float = 0.3) -> float:
    """
    Средняя яркость центрального окна кадра. center_frac=0.3 означает 30% ширины/высоты.
    """
    h, w = frame_bgr.shape[:2]
    cw = int(w * center_frac)
    ch = int(h * center_frac)
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    roi = frame_bgr[y0:y0+ch, x0:x0+cw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))
