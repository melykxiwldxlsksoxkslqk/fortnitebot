import time
from typing import Optional, Tuple, Dict, List
import json
import os

# Этот модуль использует только page (Playwright sync API) и наш vision
try:
    from . import vision
except Exception:  # pragma: no cover
    from src import vision  # type: ignore

# ----- Настройки маппинга действий (геймпад → клавиатура) -----
_DEFAULT_INPUT_MAP: Dict[str, List[str]] = {
	"A": ["Enter", "KeyA"],
	"B": ["Escape", "Backspace", "KeyB"],
	"X": ["KeyX"],
	"Y": ["Slash", "KeyY"],
	"UP": ["ArrowUp"],
	"DOWN": ["ArrowDown"],
	"LEFT": ["ArrowLeft"],
	"RIGHT": ["ArrowRight"],
	"LB": ["BracketLeft"],
	"RB": ["BracketRight"],
	"LT": ["Minus"],
	"RT": ["Equal"],
	"MENU": ["KeyM", "Tab"],
	"VIEW": ["KeyV", "F1"],
	"NEXUS": ["KeyN"],
}

_INPUT_MAP: Dict[str, List[str]] = {}


def _load_input_map() -> Dict[str, List[str]]:
    global _INPUT_MAP
    if _INPUT_MAP:
        return _INPUT_MAP
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'input_map.json')
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                # нормализуем значения к спискам строк
                mapped: Dict[str, List[str]] = {}
                for k, v in data.items():
                    if isinstance(v, list):
                        mapped[str(k).upper()] = [str(x) for x in v]
                    else:
                        mapped[str(k).upper()] = [str(v)]
                _INPUT_MAP = {**_DEFAULT_INPUT_MAP, **mapped}
                return _INPUT_MAP
    except Exception:
        pass
    _INPUT_MAP = _DEFAULT_INPUT_MAP
    return _INPUT_MAP


def press_action(page, action: str) -> None:
    """Нажимает действие из маппинга (можно объединять несколько клавиш)."""
    mapping = _load_input_map()
    keys = mapping.get(str(action).upper())
    if not keys:
        return
    for key in keys:
        try:
            page.keyboard.press(key)
            page.wait_for_timeout(60)
        except Exception:
            pass

# ----- Поиск канваса -----

def _locator_visible(locator) -> bool:
    try:
        return bool(locator) and locator.count() > 0 and locator.first.is_visible()
    except Exception:
        return False


def _get_canvas_locator_anywhere(page):
    # Сначала на верхнем уровне
    try:
        loc = page.locator('canvas')
        if _locator_visible(loc):
            return loc.first
    except Exception:
        pass
    # Внутри фреймов
    try:
        for fr in page.frames:
            try:
                loc = fr.locator('canvas')
                if _locator_visible(loc):
                    return loc.first
            except Exception:
                continue
    except Exception:
        pass
    return None


def _get_canvas_bbox(page) -> Optional[dict]:
    try:
        cnv = _get_canvas_locator_anywhere(page)
        if cnv:
            return cnv.bounding_box()
    except Exception:
        pass
    return None

# ----- Фокус/PointerLock/Fullscreen -----

def request_pointer_lock(page) -> None:
    """Пытается включить Pointer Lock на первом canvas (в нужном фрейме)."""
    try:
        cnv = _get_canvas_locator_anywhere(page)
        if cnv:
            try:
                cnv.evaluate("el => el.requestPointerLock && el.requestPointerLock()")
                return
            except Exception:
                pass
        # Фолбэк на main document
        page.evaluate(
            "() => { const c = document.querySelector('canvas'); if (c && document.pointerLockElement !== c) { c.requestPointerLock?.(); } }"
        )
    except Exception:
        pass


def ensure_fullscreen(page) -> None:
    """Переключает страницу или канвас в полноэкранный режим (если не активен)."""
    try:
        page.evaluate(
            "() => { const doc = document; const el = document.body; if (!doc.fullscreenElement) { (el.requestFullscreen||el.webkitRequestFullscreen||el.msRequestFullscreen)?.call(el); } }"
        )
    except Exception:
        pass


def ensure_stream_focus(page) -> None:
    """Захватывает фокус канваса стрима: Esc → клик по центру канваса → пытается PointerLock."""
    try:
        page.keyboard.press('Escape')
    except Exception:
        pass
    try:
        box = _get_canvas_bbox(page)
        if box:
            cx = int((box.get('x') or 0) + (box.get('width') or 0) / 2)
            cy = int((box.get('y') or 0) + (box.get('height') or 0) / 2)
            page.mouse.move(cx, cy)
            page.mouse.down()
            page.wait_for_timeout(60)
            page.mouse.up()
            page.wait_for_timeout(60)
            request_pointer_lock(page)
    except Exception:
        pass

# ----- Вспомогательные действия -----

def move_relative(page, dx: int, dy: int, steps: int = 10) -> None:
    """Плавное относительное перемещение курсора, начиная с центра канваса."""
    try:
        box = _get_canvas_bbox(page)
        if not box:
            return
        sx = int((box.get('x') or 0) + (box.get('width') or 0) / 2)
        sy = int((box.get('y') or 0) + (box.get('height') or 0) / 2)
        page.mouse.move(sx, sy, steps=max(1, steps // 2))
        page.mouse.move(sx + dx, sy + dy, steps=max(1, steps))
    except Exception:
        pass

# ----- Основные сценарии -----

def click_image_in_stream(page, template_path: str, confidence: float = 0.78,
                           timeout: int = 3, roi=None, scales=None,
                           debug_label: Optional[str] = None) -> bool:
    """Надёжный клик по изображению, отрисованному внутри стрим‑канваса.
    1) Ищем через vision.find_image_on_page (скриншот страницы)
    2) Кликаем по абсолютным CSS‑координатам
    3) Если не сработало, кликаем относительно boundingBox канваса
    """
    rect = vision.find_image_on_page(
        page,
        template_path,
        confidence=confidence,
        timeout=timeout,
        roi=roi,
        scales=scales,
        return_rect=True,
        debug_mode=True,
        debug_label=debug_label,
    )
    if not rect:
        return False

    x, y, w, h = rect
    cx_css, cy_css = vision._to_css_coords(page, x + w // 2, y + h // 2)  # type: ignore

    try:
        page.mouse.move(cx_css, cy_css, steps=8)
        page.mouse.click(cx_css, cy_css)
        page.wait_for_timeout(120)
    except Exception:
        pass

    try:
        still = vision.find_image_on_page(
            page, template_path, confidence=confidence, timeout=0.6,
            roi=None, scales=scales, return_rect=False, debug_mode=False
        )
        if not still:
            return True
    except Exception:
        pass

    try:
        box = _get_canvas_bbox(page)
        if box:
            cx = int(cx_css)
            cy = int(cy_css)
            bx = int(box.get('x') or 0)
            by = int(box.get('y') or 0)
            bw = int(box.get('width') or 0)
            bh = int(box.get('height') or 0)
            if not (bx <= cx <= bx + bw and by <= cy <= by + bh):
                cx = bx + bw // 2
                cy = by + bh // 2
            page.mouse.move(cx, cy, steps=6)
            page.mouse.down()
            page.wait_for_timeout(80)
            page.mouse.up()
            page.wait_for_timeout(100)
            return True
    except Exception:
        pass

    try:
        page.mouse.dblclick(cx_css, cy_css, delay=60)
        page.wait_for_timeout(120)
        return True
    except Exception:
        pass

    return False


def open_search(page) -> bool:
    """Открывает поиск ИСКЛЮЧИТЕЛЬНО как геймпад:
    1) Фокус стрима
    2) Несколько коротких нажатий кнопки Y (через vgamepad, если доступен)
    3) Проверка появления поля ввода кода по ассету
    """
    ensure_stream_focus(page)

    # 1) Попытка через виртуальный геймпад (Y)
    used_gamepad = False
    try:
        if vision.init_gamepad():
            used_gamepad = True
            for _ in range(3):
                vision.gp_tap_button('Y')
                page.wait_for_timeout(140)
    except Exception:
        used_gamepad = False

    # 2) Минимальный fallback: маппинг действия Y (на случай отсутствия ViGEmBus)
    if not used_gamepad:
        try:
            for _ in range(3):
                press_action(page, 'Y')
                page.wait_for_timeout(140)
        except Exception:
            pass

    # 3) Проверяем наличие поля ввода
    ipt = vision.find_image_on_page(
        page,
        'assets/island_code_input_field.png',
        confidence=0.66,
        timeout=4,
        roi=(0.03, 0.06, 0.97, 0.46),
        scales=[0.6, 0.8, 1.0, 1.2],
    )
    return bool(ipt) 