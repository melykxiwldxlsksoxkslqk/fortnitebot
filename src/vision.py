import pyautogui
import cv2
import numpy as np
import time
from typing import Dict, Tuple
import os
try:
    from ultralytics import YOLO as _YOLO
except Exception:
    _YOLO = None

_yolo_model = None

# Кэш загруженных шаблонов (RGB, optional alpha)
_TEMPLATE_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
# Кэш последнего найденного прямоугольника по шаблону (в абсолютных координатах кадра)
_LAST_RECT_CACHE: Dict[str, Tuple[int, int, int, int]] = {}


def _load_template_cached(resolved_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if resolved_path in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[resolved_path]
    template = cv2.imread(resolved_path, cv2.IMREAD_UNCHANGED)
    if template is None:
        raise IOError(f"Не удалось прочитать шаблон: {resolved_path}")
    if template.shape[2] == 4:
        template_rgb = template[:, :, :3]
        alpha_channel = template[:, :, 3]
    else:
        template_rgb = template
        alpha_channel = None  # type: ignore
    _TEMPLATE_CACHE[resolved_path] = (template_rgb, alpha_channel)
    return _TEMPLATE_CACHE[resolved_path]


def yolo_load_model(weights_path: str = os.path.join('config', 'yolo', 'model.pt')):
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    if _YOLO is None:
        raise RuntimeError("ultralytics не установлен")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"YOLO веса не найдены: {weights_path}")
    _yolo_model = _YOLO(weights_path)
    return _yolo_model

def yolo_detect(frame_bgr, conf: float = 0.35):
    """
    Выполняет детекцию YOLO по кадру BGR. Возвращает список детекций:
    [{"cls": int, "name": str, "conf": float, "xyxy": (x1,y1,x2,y2)}]
    """
    if _YOLO is None:
        return []
    try:
        model = yolo_load_model()
    except Exception:
        # Нет ultralytics или нет весов — тихий фолбэк
        return []
    # Ultralytics ожидает RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = model.predict(source=frame_rgb, verbose=False, conf=conf, device='cpu')
    out = []
    try:
        r0 = res[0]
        boxes = r0.boxes
        names = model.names or {}
        for i in range(len(boxes)):
            b = boxes[i]
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            conf_v = float(b.conf[0].item()) if hasattr(b.conf[0], 'item') else float(b.conf[0])
            cls_id = int(b.cls[0].item()) if hasattr(b.cls[0], 'item') else int(b.cls[0])
            out.append({"cls": cls_id, "name": names.get(cls_id, str(cls_id)), "conf": conf_v, "xyxy": (x1, y1, x2, y2)})
    except Exception:
        return []
    return out

_obs_captures: Dict[int, cv2.VideoCapture] = {}

# --- NEW: helpers for Playwright page screenshots ---
# Глобальный флаг: принудительно включать debug-режим в функциях поиска
_GLOBAL_DEBUG_ALWAYS = False

# Дополнительные пороги для подавления ложных срабатываний шаблонного сопоставления
_PEAK_RATIO_MIN = 1.03  # отношение лучшего пика ко второму лучшему (было 1.07)
_STABLE_FRAMES_NEEDED = 1  # достаточно одного кадра при хорошем совпадении (было 2)
_STABLE_RADIUS_PX = 16     # радиус стабильности центра, px (было 12)


def set_global_debug(enabled: bool) -> None:
    global _GLOBAL_DEBUG_ALWAYS
    _GLOBAL_DEBUG_ALWAYS = bool(enabled)

def _capture_page_bgr(page) -> np.ndarray:
    """
    Делает screenshot текущего viewport через Playwright page и возвращает BGR-изображение.
    """
    png_bytes = page.screenshot(type='png')
    nparr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    return img


def _resolve_roi_abs(img_w: int, img_h: int, roi) -> Tuple[int, int, int, int]:
    if roi and len(roi) == 4:
        x0f, y0f, x1f, y1f = roi
        x0 = max(0, min(img_w - 1, int(img_w * float(x0f))))
        y0 = max(0, min(img_h - 1, int(img_h * float(y0f))))
        x1 = max(0, min(img_w, int(img_w * float(x1f))))
        y1 = max(0, min(img_h, int(img_h * float(y1f))))
        if x1 <= x0 or y1 <= y0:
            return (0, 0, img_w, img_h)
        return (x0, y0, x1, y1)
    return (0, 0, img_w, img_h)


# --- helpers: pyramids + light smoothing for robust matching ---
def _build_pyramid(img: np.ndarray, max_levels: int = 2):
    """
    Строит простую гауссову пирамиду (изображение, масштаб относительно исходника).
    Используем 2 уровня по умолчанию (1.0, 0.5, 0.25), но останавливаемся, если кадр слишком мал.
    """
    pyr = [(img, 1.0)]
    for _ in range(max_levels):
        last_img, last_scale = pyr[-1]
        if min(last_img.shape[:2]) <= 40:
            break
        try:
            down = cv2.pyrDown(last_img)
        except Exception:
            break
        pyr.append((down, last_scale * 0.5))
    return pyr


def _gauss3(src: np.ndarray) -> np.ndarray:
    try:
        return cv2.GaussianBlur(src, (3, 3), 0, borderType=cv2.BORDER_REPLICATE)
    except Exception:
        return src


def _rotate_image_keep_bounds(img: np.ndarray, angle_deg: float, border=cv2.BORDER_REPLICATE) -> np.ndarray:
    (h, w) = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - c[0]
    M[1, 2] += (new_h / 2) - c[1]
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=border)


def _match_on_image(screen_bgr: np.ndarray, template_rgb: np.ndarray, alpha_channel: np.ndarray, confidence: float, scales, return_rect: bool, angles=None):
    screen_h, screen_w = screen_bgr.shape[:2]
    search_scales = scales if (isinstance(scales, (list, tuple)) and len(scales) > 0) else [1.0]
    angle_list = angles if (isinstance(angles, (list, tuple)) and len(angles) > 0) else [0.0]

    best_match = None  # (score, (rx, ry), tw, th)
    second_score = 0.0
    method = cv2.TM_CCORR_NORMED if alpha_channel is not None else cv2.TM_CCOEFF_NORMED

    # Пирамида экрана + лёгкое сглаживание
    screen_pyr = _build_pyramid(screen_bgr, max_levels=2)
    tpl_base = _gauss3(template_rgb)

    # --- Основной проход по цвету на пирамиде с поворотами ---
    for level_img, level_scale in screen_pyr:
        scr = _gauss3(level_img)
        for angle in angle_list:
            # Вращаем шаблон и маску, если требуется
            if abs(float(angle)) > 1e-3:
                tpl_rot = _rotate_image_keep_bounds(tpl_base, float(angle))
                msk_rot = _rotate_image_keep_bounds(alpha_channel, float(angle)) if alpha_channel is not None else None
            else:
                tpl_rot = tpl_base
                msk_rot = alpha_channel
            for s in search_scales:
                eff = float(s) * float(level_scale)
                scaled_w = max(1, int(round(tpl_rot.shape[1] * eff)))
                scaled_h = max(1, int(round(tpl_rot.shape[0] * eff)))
                if scr.shape[0] < scaled_h or scr.shape[1] < scaled_w:
                    continue
                try:
                    if eff != 1.0:
                        interp = cv2.INTER_AREA if eff < 1.0 else cv2.INTER_LINEAR
                        tpl = cv2.resize(tpl_rot, (scaled_w, scaled_h), interpolation=interp)
                        msk = None
                        if msk_rot is not None:
                            msk = cv2.resize(msk_rot, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
                    else:
                        tpl = tpl_rot
                        msk = msk_rot

                    result = cv2.matchTemplate(scr, tpl, method, mask=msk)
                    # лучший максимум
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    # второй пик (подавление окрестности вокруг максимума)
                    score1 = float(max_val)
                    second_val = 0.0
                    try:
                        res_copy = result.copy()
                        x0 = max(0, max_loc[0] - scaled_w // 2)
                        y0 = max(0, max_loc[1] - scaled_h // 2)
                        x1 = min(res_copy.shape[1], max_loc[0] + scaled_w // 2)
                        y1 = min(res_copy.shape[0], max_loc[1] + scaled_h // 2)
                        res_copy[y0:y1, x0:x1] = 0
                        _, second_val, _, _ = cv2.minMaxLoc(res_copy)
                        second_val = float(second_val)
                    except Exception:
                        second_val = 0.0

                    if best_match is None or score1 > best_match[0]:
                        rx = int(round(max_loc[0] / level_scale))
                        ry = int(round(max_loc[1] / level_scale))
                        tw = int(round(scaled_w / level_scale))
                        th = int(round(scaled_h / level_scale))
                        best_match = (score1, (rx, ry), tw, th)
                        second_score = max(second_score, second_val)
                    else:
                        second_score = max(second_score, score1, second_val)
                except cv2.error:
                    continue

    if best_match is not None and best_match[0] >= confidence:
        # Проверка отношения пиков для снижения ложных срабатываний
        ratio_ok = True
        if second_score > 1e-6:
            # Если совпадение очень высокое (>= 0.92), смягчаем требование по ratio
            if best_match[0] >= max(0.92, confidence + 0.1):
                ratio_ok = True
            else:
                ratio_ok = (best_match[0] / max(1e-6, second_score)) >= _PEAK_RATIO_MIN
        if not ratio_ok:
            return None
        score, (rx, ry), tw, th = best_match
        if return_rect:
            return (rx, ry, tw, th)
        return (rx + tw // 2, ry + th // 2)

    # --- Фолбэк: границы + морфология на пирамиде ---
    try:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tpl_gray_base = cv2.cvtColor(tpl_base, cv2.COLOR_BGR2GRAY)
        tpl_gray_base = _gauss3(tpl_gray_base)
        best_edge = None
        for level_img, level_scale in screen_pyr:
            gray = cv2.cvtColor(level_img, cv2.COLOR_BGR2GRAY)
            gray = _gauss3(gray)
            screen_edges = cv2.Canny(gray, 50, 150)
            try:
                screen_edges = cv2.morphologyEx(screen_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
            except Exception:
                pass
            for s in search_scales:
                eff = float(s) * float(level_scale)
                scaled_w = max(1, int(round(template_rgb.shape[1] * eff)))
                scaled_h = max(1, int(round(template_rgb.shape[0] * eff)))
                if level_img.shape[0] < scaled_h or level_img.shape[1] < scaled_w:
                    continue
                if eff != 1.0:
                    interp = cv2.INTER_AREA if eff < 1.0 else cv2.INTER_LINEAR
                    tgray = cv2.resize(tpl_gray_base, (scaled_w, scaled_h), interpolation=interp)
                else:
                    tgray = tpl_gray_base
                tpl_edges = cv2.Canny(tgray, 50, 150)
                try:
                    tpl_edges = cv2.morphologyEx(tpl_edges, cv2.MORPH_CLOSE, kernel, iterations=1)
                except Exception:
                    pass
                res = cv2.matchTemplate(screen_edges, tpl_edges, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if best_edge is None or float(max_val) > best_edge[0]:
                    rx = int(round(max_loc[0] / level_scale))
                    ry = int(round(max_loc[1] / level_scale))
                    tw = int(round(scaled_w / level_scale))
                    th = int(round(scaled_h / level_scale))
                    best_edge = (float(max_val), (rx, ry), tw, th)
        if best_edge is not None and best_edge[0] >= max(0.5, confidence - 0.15):
            _, (rx, ry), tw, th = best_edge
            if return_rect:
                return (rx, ry, tw, th)
            return (rx + tw // 2, ry + th // 2)
    except Exception:
        pass

    # --- Дополнительный фолбэк: бинарное сопоставление (CLAHE + adaptiveThreshold) ---
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        tpl_gray0 = cv2.cvtColor(tpl_base, cv2.COLOR_BGR2GRAY)
        tpl_eq = clahe.apply(tpl_gray0)
        best_bin = None
        for level_img, level_scale in screen_pyr:
            gray = cv2.cvtColor(level_img, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            scr_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            for s in search_scales:
                eff = float(s) * float(level_scale)
                scaled_w = max(1, int(round(template_rgb.shape[1] * eff)))
                scaled_h = max(1, int(round(template_rgb.shape[0] * eff)))
                if level_img.shape[0] < scaled_h or level_img.shape[1] < scaled_w:
                    continue
                if eff != 1.0:
                    interp = cv2.INTER_AREA if eff < 1.0 else cv2.INTER_LINEAR
                    tpl_eq_r = cv2.resize(tpl_eq, (scaled_w, scaled_h), interpolation=interp)
                else:
                    tpl_eq_r = tpl_eq
                tpl_bin = cv2.adaptiveThreshold(tpl_eq_r, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                res = cv2.matchTemplate(scr_bin, tpl_bin, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if best_bin is None or float(max_val) > best_bin[0]:
                    rx = int(round(max_loc[0] / level_scale))
                    ry = int(round(max_loc[1] / level_scale))
                    tw = int(round(scaled_w / level_scale))
                    th = int(round(scaled_h / level_scale))
                    best_bin = (float(max_val), (rx, ry), tw, th)
        if best_bin is not None and best_bin[0] >= max(0.55, confidence - 0.1):
            _, (rx, ry), tw, th = best_bin
            if return_rect:
                return (rx, ry, tw, th)
            return (rx + tw // 2, ry + th // 2)
    except Exception:
        pass

    # --- Последний фолбэк: ORB + BFMatcher + гомография (features2d) ---
    try:
        # Готовим градации серого
        tpl_gray = cv2.cvtColor(tpl_base, cv2.COLOR_BGR2GRAY)
        scr_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(nfeatures=800)
        kpt1, des1 = orb.detectAndCompute(tpl_gray, None)
        kpt2, des2 = orb.detectAndCompute(scr_gray, None)
        if des1 is not None and des2 is not None and len(kpt1) >= 8 and len(kpt2) >= 8:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            knn = bf.knnMatch(des1, des2, k=2)
            good = []
            for m, n in knn:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(good) >= 8:
                src_pts = np.float32([kpt1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kpt2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H, mask_inl = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if H is not None:
                    h, w = tpl_gray.shape[:2]
                    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                    proj = cv2.perspectiveTransform(corners, H)
                    xs = proj[:, 0, 0]
                    ys = proj[:, 0, 1]
                    x_min = max(0, int(np.floor(xs.min())))
                    y_min = max(0, int(np.floor(ys.min())))
                    x_max = min(screen_bgr.shape[1] - 1, int(np.ceil(xs.max())))
                    y_max = min(screen_bgr.shape[0] - 1, int(np.ceil(ys.max())))
                    if x_max > x_min and y_max > y_min:
                        tw = x_max - x_min
                        th = y_max - y_min
                        if return_rect:
                            return (x_min, y_min, tw, th)
                        return (x_min + tw // 2, y_min + th // 2)
    except Exception:
        pass

    return None


def find_image_on_page(page, template_path, confidence=0.8, timeout=10, roi=None, scales=None, return_rect: bool = False, debug_mode: bool = False, debug_label: str = None):
    """
    Ищет шаблон на скриншоте текущего viewport Playwright-страницы.
    Возвращает координаты в системе координат страницы (viewport).
    При debug_mode=True сохраняет снимок в debug/ с прямоугольником найденного региона
    (или с пометкой NO_MATCH при отсутствии совпадений).
    """
    # Учитываем глобальный флаг отладки
    debug_mode = bool(debug_mode or _GLOBAL_DEBUG_ALWAYS)
    start_time = time.time()
    # загрузка шаблона (через кэш)
    try:
        resolved_path = resolve_asset_path(template_path)
        template_rgb, alpha_channel = _load_template_cached(resolved_path)
    except Exception as e:
        print(f"Ошибка загрузки шаблона '{template_path}': {e}")
        return None

    last_page_bgr = None
    last_rect_abs = None  # (x, y, w, h)
    stable_hits = []  # [(cx, cy)] подряд

    while time.time() - start_time < timeout:
        page_bgr = _capture_page_bgr(page)
        last_page_bgr = page_bgr
        img_h, img_w = page_bgr.shape[:2]
        # Если ROI не задана — пробуем сначала небольшую окрестность вокруг последнего попадания
        x0, y0, x1, y1 = _resolve_roi_abs(img_w, img_h, roi)
        if roi is None:
            prev = _LAST_RECT_CACHE.get(resolved_path)
            if prev is not None:
                px, py, pw, ph = prev
                pad = int(0.3 * max(pw, ph))
                x0 = max(0, px - pad)
                y0 = max(0, py - pad)
                x1 = min(img_w, px + pw + pad)
                y1 = min(img_h, py + ph + pad)
        roi_img = page_bgr[y0:y1, x0:x1]

        # Если debug_mode, принудительно просим прямоугольник для отрисовки
        want_rect = return_rect or debug_mode
        found = _match_on_image(roi_img, template_rgb, alpha_channel, confidence, scales, want_rect, angles=[-10, -5, 0, 5, 10])
        if found is not None:
            if want_rect:
                rx, ry, tw, th = found
                abs_rect = (x0 + rx, y0 + ry, tw, th)
                last_rect_abs = abs_rect
                _LAST_RECT_CACHE[resolved_path] = abs_rect
                # стабилизация по двум кадрам
                cx = abs_rect[0] + abs_rect[2] // 2
                cy = abs_rect[1] + abs_rect[3] // 2
                stable_hits.append((cx, cy))
                if len(stable_hits) >= 2:
                    dx = abs(stable_hits[-1][0] - stable_hits[-2][0])
                    dy = abs(stable_hits[-1][1] - stable_hits[-2][1])
                    if max(dx, dy) <= _STABLE_RADIUS_PX:
                        if debug_mode:
                            try:
                                dbg = page_bgr.copy()
                                cv2.rectangle(dbg, (abs_rect[0], abs_rect[1]), (abs_rect[0] + abs_rect[2], abs_rect[1] + abs_rect[3]), (0, 255, 0), 2)
                                base = os.path.basename(resolve_asset_path(template_path))
                                os.makedirs('debug', exist_ok=True)
                                label = debug_label or 'page_match'
                                cv2.imwrite(f"debug/{label}_{base}", dbg)
                            except Exception:
                                pass
                        return abs_rect if return_rect else (cx, cy)
            else:
                cx, cy = found
                abs_rect = (max(0, x0 + cx - template_rgb.shape[1] // 2), max(0, y0 + cy - template_rgb.shape[0] // 2), template_rgb.shape[1], template_rgb.shape[0])
                _LAST_RECT_CACHE[resolved_path] = abs_rect
                stable_hits.append((x0 + cx, y0 + cy))
                if len(stable_hits) >= 2:
                    dx = abs(stable_hits[-1][0] - stable_hits[-2][0])
                    dy = abs(stable_hits[-1][1] - stable_hits[-2][1])
                    if max(dx, dy) <= _STABLE_RADIUS_PX:
                        if debug_mode:
                            try:
                                dbg = page_bgr.copy()
                                # приблизительный прямоугольник на основе исходного шаблона
                                tw, th = template_rgb.shape[1], template_rgb.shape[0]
                                top_left = (max(0, x0 + cx - tw // 2), max(0, y0 + cy - th // 2))
                                bottom_right = (min(img_w - 1, top_left[0] + tw), min(img_h - 1, top_left[1] + th))
                                cv2.rectangle(dbg, top_left, bottom_right, (0, 255, 0), 2)
                                base = os.path.basename(resolve_asset_path(template_path))
                                os.makedirs('debug', exist_ok=True)
                                label = debug_label or 'page_match'
                                cv2.imwrite(f"debug/{label}_{base}", dbg)
                            except Exception:
                                pass
                        return (x0 + cx, y0 + cy) if not return_rect else abs_rect

        time.sleep(0.25)
        # ускорение для коротких таймаутов
        if (timeout or 0) <= 2:
            time.sleep(0.10)

    # таймаут
    if debug_mode and last_page_bgr is not None:
        try:
            dbg = last_page_bgr.copy()
            if last_rect_abs:
                tl = (last_rect_abs[0], last_rect_abs[1])
                br = (last_rect_abs[0] + last_rect_abs[2], last_rect_abs[1] + last_rect_abs[3])
                cv2.rectangle(dbg, tl, br, (0, 0, 255), 2)
            else:
                cv2.putText(dbg, 'NO_MATCH', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            base = os.path.basename(resolve_asset_path(template_path))
            os.makedirs('debug', exist_ok=True)
            label = debug_label or 'page_nomatch'
            cv2.imwrite(f"debug/{label}_{base}", dbg)
        except Exception:
            pass

    return None


# Обновлённая обёртка клика по картинке на странице с поддержкой debug_mode

def _get_page_device_pixel_ratio(page) -> float:
    try:
        dpr = page.evaluate("() => window.devicePixelRatio")
        return float(dpr) if dpr else 1.0
    except Exception:
        return 1.0


def _to_css_coords(page, x_img: int, y_img: int) -> Tuple[int, int]:
    dpr = _get_page_device_pixel_ratio(page)
    try:
        vp = page.viewport_size or {"width": 1280, "height": 720}
        cx = max(1, min(int(vp["width"]) - 2, int(round(x_img / max(0.5, dpr)))))
        cy = max(1, min(int(vp["height"]) - 2, int(round(y_img / max(0.5, dpr)))))
        return cx, cy
    except Exception:
        return int(round(x_img / max(0.5, dpr))), int(round(y_img / max(0.5, dpr)))


def page_click_from_screenshot(page, x_img: int, y_img: int, steps: int = 12) -> None:
    cx, cy = _to_css_coords(page, x_img, y_img)
    try:
        page.mouse.move(cx, cy, steps=max(1, int(steps)))
    except Exception:
        page.mouse.move(cx, cy)
    page.mouse.click(cx, cy)


def click_on_image_on_page(page, template_path, confidence=0.8, timeout=10, roi=None, scales=None, debug_mode: bool = False, debug_label: str = None):
    """
    Находит изображение на странице и кликает по нему через page.mouse.
    При debug_mode=True сохраняет скриншот с разметкой найденного региона.
    """
    coords_or_rect = find_image_on_page(page, template_path, confidence=confidence, timeout=timeout, roi=roi, scales=scales, return_rect=True, debug_mode=debug_mode, debug_label=debug_label)
    if coords_or_rect:
        # coords_or_rect = (x, y, w, h) в пикселях скриншота страницы
        x, y, w, h = coords_or_rect
        cx, cy = _to_css_coords(page, x + w // 2, y + h // 2)
        try:
            page.mouse.move(cx, cy, steps=8)
            page.mouse.click(cx, cy)
            page.wait_for_timeout(180)
            # быстрый локальный рескан в окне ±40 px вокруг центра (в координатах скриншота)
            vp = page.viewport_size or {"width": 1280, "height": 720}
            dpr = _get_page_device_pixel_ratio(page)
            imgW = max(1, int(vp["width"] * max(1.0, dpr)))
            imgH = max(1, int(vp["height"] * max(1.0, dpr)))
            cx_img = x + w // 2
            cy_img = y + h // 2
            x0f = max(0.0, float(cx_img - 40) / float(imgW))
            y0f = max(0.0, float(cy_img - 40) / float(imgH))
            x1f = min(1.0, float(cx_img + 40) / float(imgW))
            y1f = min(1.0, float(cy_img + 40) / float(imgH))
            win = (x0f, y0f, x1f, y1f)
            still = find_image_on_page(page, template_path, confidence=confidence, timeout=0.6, roi=win, scales=scales, return_rect=False, debug_mode=False)
            if not still:
                return True
            # локальный повтор
            page.mouse.move(cx, cy)
            page.mouse.click(cx, cy)
            page.wait_for_timeout(150)
            return True
        except Exception:
            pass
    return False

# --- NEW: Heuristic detection of CONNECTING/LOGGING overlays on page ---
def detect_connecting_overlay_on_page(page) -> bool:
    """
    Эвристика: определяет, виден ли на странице оверлей CONNECTING/LOGGING IN
    по яркому циановому тексту в нижней части экрана.
    Возвращает True, если оверлей вероятно присутствует.
    """
    try:
        img = _capture_page_bgr(page)
        h, w = img.shape[:2]
        def roi_abs(fr):
            x0 = max(0, min(w - 1, int(w * fr[0]))); y0 = max(0, min(h - 1, int(h * fr[1])))
            x1 = max(0, min(w, int(w * fr[2])));     y1 = max(0, min(h, int(h * fr[3])))
            return x0, y0, x1, y1
        # Две зоны: левый-низ (CONNECTING...) и центр-низ (LOGGING IN...)
        rois = [
            (0.00, 0.82, 0.42, 0.99),  # шире слева снизу
            (0.28, 0.72, 0.76, 0.94),  # центр-низ
            (0.00, 0.74, 1.00, 0.98),  # общий низ как резерв
        ]
        # Циан/бирюза в HSV — расширим диапазон оттенков
        lower = np.array([60, 40, 100], dtype=np.uint8)
        upper = np.array([130, 255, 255], dtype=np.uint8)
        for fr in rois:
            x0, y0, x1, y1 = roi_abs(fr)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = img[y0:y1, x0:x1]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            # Сгладим и чуть закроем пробелы внутри букв, чтобы ловить тонкий шрифт
            mask = cv2.medianBlur(mask, 3)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            ratio = float(np.count_nonzero(mask)) / float(mask.size)
            if ratio > 0.003:  # ~0.3% пикселей циановые — вероятно CONNECTING/LOGGING
                return True
        return False
    except Exception:
        return False

# --- NEW: Heuristic detection of plane launch screen ---
def detect_plane_screen_on_page(page) -> bool:
    """
    Эвристика: определяет, отображается ли экран "Launching with cloud gaming"
    с зелёным самолётиком на чёрном фоне. Используем зелёную маску по центру.
    Возвращает True, если самолёт/полосы вероятны.
    """
    try:
        img = _capture_page_bgr(page)
        h, w = img.shape[:2]
        # Центральная зона, где находится самолёт и зелёные полосы
        x0 = int(w * 0.20); x1 = int(w * 0.80)
        y0 = int(h * 0.30); y1 = int(h * 0.75)
        roi = img[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Ярко-зелёный/салатовый диапазон
        lower = np.array([40, 60, 110], dtype=np.uint8)
        upper = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.medianBlur(mask, 3)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        return ratio > 0.0015  # ~0.15% зелёных пикселей в центре
    except Exception:
        return False


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

def capture_obs_frame(index: int = 0, preferred_size: Tuple[int, int] = (1280, 720)):
    """
    Захватывает кадр из OBS Virtual Camera. Возвращает BGR-изображение нужного размера.
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

def center_brightness(frame_bgr, center_frac: float = 0.3) -> float:
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

def detect_enemy_health_bar(frame_bgr) -> bool:
    """
    Примитивное распознавание красной полоски здоровья в верхней центральной части.
    Работает эвристически и может потребовать подстройки порогов.
    """
    h, w = frame_bgr.shape[:2]
    # Зона интереса: верхняя центральная полоса
    roi_y0 = int(h * 0.06)
    roi_y1 = int(h * 0.12)
    roi_x0 = int(w * 0.25)
    roi_x1 = int(w * 0.75)
    roi = frame_bgr[roi_y0:roi_y1, roi_x0:roi_x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Красный в HSV разнесён по двум диапазонам
    mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    red_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    return red_ratio > 0.02

def capture_screen():
	"""
	Захватывает основной экран и возвращает его как numpy-массив в формате BGR.
	Использует mss для корректной геометрии при масштабировании Windows; при ошибке — pyautogui.
	"""
	try:
		import mss  # local import, если пакет не установлен
		with mss.mss() as sct:
			mon = sct.monitors[1]  # основной монитор
			img = np.array(sct.grab(mon))  # BGRA
			return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
	except Exception:
		# Фолбэк на pyautogui
		screenshot = pyautogui.screenshot()
		return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def resolve_asset_path(path: str) -> str:
    """
    Возвращает существующий путь к ассету. Поддерживает варианты с .png и .png.txt.
    Если исходный путь существует — возвращается он, иначе пробует альтернативу.
    """
    if os.path.exists(path):
        return path
    if path.endswith('.png') and os.path.exists(path + '.txt'):
        return path + '.txt'
    if path.endswith('.png.txt') and os.path.exists(path[:-4]):
        return path[:-4]
    return path

def find_image(template_path, confidence=0.8, timeout=10, roi=None, scales=None, return_rect: bool = False, debug_mode: bool = False):
    """
    Ищет изображение (шаблон) на экране.

    Параметры:
    - template_path: путь к шаблону (поддерживает .png и .png.txt)
    - confidence: минимальная уверенность совпадения [0..1]
    - timeout: таймаут ожидания (сек)
    - roi: область поиска на экране (относительные доли)
    - scales: список коэффициентов масштабирования шаблона
    - return_rect: вернуть ли прямоугольник (x, y, w, h) вместо центра
    - debug_mode: сохранить отладочный снимок
    """
    # Учитываем глобальный флаг отладки
    debug_mode = bool(debug_mode or _GLOBAL_DEBUG_ALWAYS)
    start_time = time.time()
    try:
        resolved_path = resolve_asset_path(template_path)
        template_rgb, alpha_channel = _load_template_cached(resolved_path)
    except Exception as e:
        print(f"Ошибка загрузки шаблона '{template_path}': {e}")
        return None

    last_screen = None
    last_rect_abs = None
    stable_hits = []  # [(cx, cy)] подряд

    while time.time() - start_time < timeout:
        screen_bgr = capture_screen()
        last_screen = screen_bgr
        img_h, img_w = screen_bgr.shape[:2]
        # Если ROI не задана — пробуем сначала окрестность прошлого попадания
        x0, y0, x1, y1 = _resolve_roi_abs(img_w, img_h, roi)
        if roi is None:
            prev = _LAST_RECT_CACHE.get(resolved_path)
            if prev is not None:
                px, py, pw, ph = prev
                pad = int(0.3 * max(pw, ph))
                x0 = max(0, px - pad)
                y0 = max(0, py - pad)
                x1 = min(img_w, px + pw + pad)
                y1 = min(img_h, py + ph + pad)
        roi_img = screen_bgr[y0:y1, x0:x1]

        want_rect = return_rect or debug_mode
        found = _match_on_image(roi_img, template_rgb, alpha_channel, confidence, scales, want_rect, angles=[-10, -5, 0, 5, 10])
        if found is not None:
            if want_rect:
                rx, ry, tw, th = found
                abs_rect = (x0 + rx, y0 + ry, tw, th)
                last_rect_abs = abs_rect
                _LAST_RECT_CACHE[resolved_path] = abs_rect
                cx = abs_rect[0] + abs_rect[2] // 2
                cy = abs_rect[1] + abs_rect[3] // 2
                stable_hits.append((cx, cy))
                if len(stable_hits) >= _STABLE_FRAMES_NEEDED:
                    dx = abs(stable_hits[-1][0] - stable_hits[-2][0])
                    dy = abs(stable_hits[-1][1] - stable_hits[-2][1])
                    if max(dx, dy) <= _STABLE_RADIUS_PX:
                        if debug_mode:
                            try:
                                dbg = screen_bgr.copy()
                                cv2.rectangle(dbg, (abs_rect[0], abs_rect[1]), (abs_rect[0] + abs_rect[2], abs_rect[1] + abs_rect[3]), (0, 255, 0), 2)
                                base = os.path.basename(resolve_asset_path(template_path))
                                os.makedirs('debug', exist_ok=True)
                                cv2.imwrite(f"debug/match_{base}", dbg)
                            except Exception:
                                pass
                        return abs_rect if return_rect else (cx, cy)
            else:
                cx, cy = found
                abs_rect = (
                    max(0, x0 + cx - template_rgb.shape[1] // 2),
                    max(0, y0 + cy - template_rgb.shape[0] // 2),
                    template_rgb.shape[1],
                    template_rgb.shape[0],
                )
                _LAST_RECT_CACHE[resolved_path] = abs_rect
                stable_hits.append((x0 + cx, y0 + cy))
                if len(stable_hits) >= _STABLE_FRAMES_NEEDED:
                    dx = abs(stable_hits[-1][0] - stable_hits[-2][0])
                    dy = abs(stable_hits[-1][1] - stable_hits[-2][1])
                    if max(dx, dy) <= _STABLE_RADIUS_PX:
                        if debug_mode:
                            try:
                                dbg = screen_bgr.copy()
                                tw, th = template_rgb.shape[1], template_rgb.shape[0]
                                top_left = (max(0, x0 + cx - tw // 2), max(0, y0 + cy - th // 2))
                                bottom_right = (min(img_w - 1, top_left[0] + tw), min(img_h - 1, top_left[1] + th))
                                cv2.rectangle(dbg, top_left, bottom_right, (0, 255, 0), 2)
                                base = os.path.basename(resolve_asset_path(template_path))
                                os.makedirs('debug', exist_ok=True)
                                cv2.imwrite(f"debug/match_{base}", dbg)
                            except Exception:
                                pass
                        return (x0 + cx, y0 + cy) if not return_rect else abs_rect

        time.sleep(0.25)

    if debug_mode and last_screen is not None:
        try:
            dbg = last_screen.copy()
            if last_rect_abs:
                tl = (last_rect_abs[0], last_rect_abs[1])
                br = (last_rect_abs[0] + last_rect_abs[2], last_rect_abs[1] + last_rect_abs[3])
                cv2.rectangle(dbg, tl, br, (0, 0, 255), 2)
            else:
                cv2.putText(dbg, 'NO_MATCH', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            base = os.path.basename(resolve_asset_path(template_path))
            os.makedirs('debug', exist_ok=True)
            cv2.imwrite(f"debug/match_{base}", dbg)
        except Exception:
            pass

    return None

def _screen_to_os_coords(x: int, y: int) -> Tuple[int, int]:
    """
    Преобразует координаты из пространства "сырых" пикселей экрана (как в mss.grab)
    в координаты курсора PyAutoGUI с учётом масштабирования Windows (DPI scaling).
    """
    try:
        import mss
        with mss.mss() as sct:
            mon = sct.monitors[1]
            mw = int(mon.get('width') or 0)
            mh = int(mon.get('height') or 0)
    except Exception:
        mw = 0
        mh = 0
    try:
        vw, vh = pyautogui.size()
    except Exception:
        vw, vh = (mw or 1), (mh or 1)
    sx = float(vw) / float(mw or vw)
    sy = float(vh) / float(mh or vh)
    return int(round(float(x) * sx)), int(round(float(y) * sy))


def click_on_image(template_path, confidence=0.8, timeout=10, roi=None, scales=None):
    """
    Находит изображение на экране и кликает по его центру.
    Дополнительно поддерживает ограничение ROI и набор масштабов.
    """
    coords = find_image(template_path, confidence=confidence, timeout=timeout, roi=roi, scales=scales)
    if coords:
        ox, oy = _screen_to_os_coords(coords[0], coords[1])
        # Пытаемся кликнуть безопасно (учитывая флаг запрета OS ввода)
        if _safe_py_click(ox, oy):
            return True
        # Если безопасный клик не разрешён или не удался — не трогаем системную мышь
        return False
    return False

# --- Старая реализация, оставлена для обратной совместимости, если где-то используется ---
# Вы можете удалить это, если уверены, что вызовы были обновлены
def find_image_on_screen(template_path, confidence, timeout):
    return find_image(template_path, confidence, timeout)

def wait_for_image_state(template_path: str, should_appear: bool = True, confidence: float = 0.8, timeout: int = 30, poll_interval: float = 0.5, roi=None, scales=None) -> bool:
    """
    Ожидает, пока изображение появится на экране (should_appear=True) или исчезнет (should_appear=False).
    Возвращает True при успехе, False по таймауту.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        found = find_image(template_path, confidence=confidence, timeout=0, roi=roi, scales=scales)
        if bool(found) == should_appear:
            return True
        time.sleep(poll_interval)
    return False

def wait_for_scene_change(timeout: int = 60, sample_interval: float = 1.0, resize_to=(320, 180), diff_threshold: float = 8.0, method: str = 'auto') -> bool:
    """
    Ждет заметного изменения сцены на экране.
    По умолчанию выбирает лучший доступный метод:
    - 'ssim' через cv2.quality (если доступен в opencv-contrib)
    - 'hash' через cv2.img_hash (если доступен)
    - иначе среднее абсолютное отличие (как раньше)
    Возвращает True при существенной смене, False по таймауту.
    """
    def take_frame_gray():
        frame = capture_screen()
        frame_small = cv2.resize(frame, resize_to)
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        return gray

    def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
        # Попытка использовать cv2.quality если доступен
        try:
            q = cv2.quality.QualitySSIM_compute(a, b)[0]
            return float(q)
        except Exception:
            pass
        # Собственная оценка SSIM по окну целиком
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        L = 255.0
        c1 = (0.01 * L) ** 2
        c2 = (0.03 * L) ** 2
        mu_a = a.mean()
        mu_b = b.mean()
        var_a = ((a - mu_a) ** 2).mean()
        var_b = ((b - mu_b) ** 2).mean()
        cov_ab = ((a - mu_a) * (b - mu_b)).mean()
        num = (2 * mu_a * mu_b + c1) * (2 * cov_ab + c2)
        den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
        if den == 0:
            return 1.0 if num == 0 else 0.0
        return float(num / den)

    def hash_dist(a: np.ndarray, b: np.ndarray) -> float:
        # Если есть img_hash (contrib) — используем pHash
        try:
            hasher = cv2.img_hash.PHash_create()
            ha = hasher.compute(a)
            hb = hasher.compute(b)
            d = cv2.norm(ha, hb, cv2.NORM_HAMMING)
            return float(d)
        except Exception:
            return float('inf')

    baseline = take_frame_gray()
    start_time = time.time()

    use_method = method
    if method == 'auto':
        if hasattr(cv2, 'quality'):
            use_method = 'ssim'
        elif hasattr(cv2, 'img_hash'):
            use_method = 'hash'
        else:
            use_method = 'diff'

    while time.time() - start_time < timeout:
        time.sleep(sample_interval)
        current = take_frame_gray()
        if use_method == 'ssim':
            s = ssim_gray(baseline, current)
            # чем ниже SSIM, тем сильнее отличие
            if s <= 0.85:
                return True
        elif use_method == 'hash':
            d = hash_dist(baseline, current)
            # эмпирический порог для pHash расстояния (чем больше, тем сильнее отличие)
            if d >= 12.0:
                return True
        else:
            diff = cv2.absdiff(baseline, current)
            mean_diff = float(np.mean(diff))
            if mean_diff >= diff_threshold:
                return True
    return False

def find_search_icon_on_page(page, confidence: float = 0.75, timeout: int = 15, debug_mode: bool = False):
    """
    Находит иконку поиска (лупу) на странице: прямое сопоставление по снимку page с авто‑маской.
    Возвращает (x, y) центра или None.
    """
    debug_mode = bool(debug_mode or _GLOBAL_DEBUG_ALWAYS)
    try:
        resolved_path = resolve_asset_path('assets/search_icon.png')
        tpl_rgb, tpl_alpha = _load_template_cached(resolved_path)
        # Если у шаблона нет альфы — создадим маску из почти белых пикселей
        if tpl_alpha is None:
            try:
                g = cv2.cvtColor(tpl_rgb, cv2.COLOR_BGR2GRAY)
                _, m = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                tpl_alpha = m
            except Exception:
                tpl_alpha = None
    except Exception as e:
        print(f"[search_icon] template error: {e}")
        return None

    start = time.time()
    # Кандидатные ROI в верхней зоне интерфейса
    rois = [
        (0.00, 0.04, 0.22, 0.16),
        (0.00, 0.00, 0.28, 0.22),
        (0.00, 0.00, 0.35, 0.30),
        None
    ]
    scales = [0.35, 0.45, 0.55, 0.70, 0.85, 1.00, 1.15]
    angles = [-10, -5, 0, 5, 10]

    last_bgr = None
    last_rect = None

    while time.time() - start < timeout:
        page_bgr = _capture_page_bgr(page)
        last_bgr = page_bgr
        H, W = page_bgr.shape[:2]
        for idx, fr in enumerate(rois):
            x0, y0, x1, y1 = _resolve_roi_abs(W, H, fr)
            crop = page_bgr[y0:y1, x0:x1]
            want_rect = True  # для надёжного дебага/координат
            found = _match_on_image(crop, tpl_rgb, tpl_alpha, confidence, scales, want_rect, angles=angles)
            if found is not None:
                rx, ry, tw, th = found
                abs_rect = (x0 + rx, y0 + ry, tw, th)
                # debug
                if debug_mode:
                    try:
                        dbg = page_bgr.copy()
                        cv2.rectangle(dbg, (abs_rect[0], abs_rect[1]), (abs_rect[0] + abs_rect[2], abs_rect[1] + abs_rect[3]), (0, 255, 0), 2)
                        os.makedirs('debug', exist_ok=True)
                        cv2.imwrite(f"debug/page_roi{idx}_search_icon.png", dbg)
                    except Exception:
                        pass
                cx = abs_rect[0] + abs_rect[2] // 2
                cy = abs_rect[1] + abs_rect[3] // 2
                # Лог координат: прямоугольник, центр (изображение), CSS‑координаты и DPR
                try:
                    dpr = _get_page_device_pixel_ratio(page)
                    css_x, css_y = _to_css_coords(page, cx, cy)
                    print(f"[SEARCH_ICON] ROI#{idx} rect=({abs_rect[0]},{abs_rect[1]},{abs_rect[2]},{abs_rect[3]}) center_img=({cx},{cy}) center_css=({css_x},{css_y}) dpr={dpr:.2f}")
                except Exception:
                    print(f"[SEARCH_ICON] ROI#{idx} rect=({abs_rect[0]},{abs_rect[1]},{abs_rect[2]},{abs_rect[3]}) center_img=({cx},{cy})")
                _LAST_RECT_CACHE[resolved_path] = abs_rect
                return (cx, cy)
        time.sleep(0.3)

    # Таймаут: логируем NO_MATCH только по снимку страницы
    if debug_mode and last_bgr is not None:
        try:
            dbg = last_bgr.copy()
            cv2.putText(dbg, 'NO_MATCH', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            os.makedirs('debug', exist_ok=True)
            cv2.imwrite("debug/page_search_icon_nomatch.png", dbg)
        except Exception:
            pass
    return None

def detect_tap_to_continue_on_page(page) -> bool:
    """
    Эвристика: обнаруживает оверлей "Click or tap here to continue playing".
    Признаки: крупный белый текст в центральной полосе экрана на затемнённом фоне.
    Возвращает True, если оверлей вероятен.
    """
    try:
        img = _capture_page_bgr(page)
        h, w = img.shape[:2]
        # Центральная горизонтальная полоса
        x0 = int(w * 0.15); x1 = int(w * 0.85)
        y0 = int(h * 0.42); y1 = int(h * 0.58)
        if x1 <= x0 or y1 <= y0:
            return False
        roi = img[y0:y1, x0:x1]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Белый текст: низкая насыщенность, высокая яркость
        lower = np.array([0, 0, 215], dtype=np.uint8)
        upper = np.array([179, 60, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        ratio = float(np.count_nonzero(mask)) / float(mask.size)
        # Порог под крупную фразу (~несколько процентов полосы)
        return ratio > 0.01
    except Exception:
        return False


def dismiss_tap_to_continue_on_page(page) -> bool:
    """
    Если виден оверлей 'tap to continue' — кликает в центр viewport и возвращает True.
    """
    try:
        if not detect_tap_to_continue_on_page(page):
            return False
        vp = page.viewport_size or {"width": 1280, "height": 720}
        cx = int(vp["width"]) // 2
        cy = int(vp["height"]) // 2
        page.mouse.move(cx, cy)
        page.mouse.click(cx, cy)
        page.wait_for_timeout(200)
        return True
    except Exception:
        return False

# Глобальный флаг: запрет системного ввода (перехвата мыши/клавы ОС)
_DISABLE_OS_INPUT = True


def set_disable_os_input(disable: bool) -> None:
    global _DISABLE_OS_INPUT
    _DISABLE_OS_INPUT = bool(disable)


def _safe_py_click(x: int, y: int) -> bool:
    if _DISABLE_OS_INPUT:
        return False
    try:
        pyautogui.moveTo(x, y, duration=0.10)
        pyautogui.click()
        return True
    except Exception:
        return False


def _safe_py_type(text: str, interval: float = 0.05) -> bool:
    if _DISABLE_OS_INPUT:
        return False
    try:
        pyautogui.write(text, interval=interval)
        return True
    except Exception:
        return False


def _safe_py_press(key: str) -> bool:
    if _DISABLE_OS_INPUT:
        return False
    try:
        pyautogui.press(key)
        return True
    except Exception:
        return False

# --- НОВЫЙ МОДУЛЬ ВВОДА: pydirectinput ---
# Более надежный способ симуляции ввода для игр и стриминговых клиентов
import pydirectinput

# Настройка pydirectinput для работы без пауз по-умолчанию
pydirectinput.PAUSE = 0.01
pydirectinput.FAILSAFE = False


def press_key(key: str, presses: int = 1, interval: float = 0.1) -> bool:
    """
    Нажимает системную клавишу с использованием pydirectinput.
    Более надежно для игр, чем pyautogui.
    """
    if _DISABLE_OS_INPUT:
        print("[Input] OS input is disabled, skipping key press.")
        return False
    try:
        for i in range(presses):
            pydirectinput.press(key)
            if presses > 1 and i < presses - 1:
                time.sleep(interval)
        return True
    except Exception as e:
        print(f"[Input] Error pressing key '{key}': {e}")
        return False


def type_text(text: str, interval: float = 0.1) -> bool:
    """
    Печатает текст с использованием pydirectinput.
    """
    if _DISABLE_OS_INPUT:
        print("[Input] OS input is disabled, skipping typing.")
        return False
    try:
        pydirectinput.write(text, interval=interval)
        return True
    except Exception as e:
        print(f"[Input] Error typing text: {e}")
        return False

# --- НОВЫЙ НАВИГАЦИОННЫЙ МОДУЛЬ: "РЕЖИМ ГЕЙМПАДА" ---

# Обёртки для vgamepad (виртуальный Xbox360 контроллер)
try:
    from vgamepad import VX360Gamepad, XUSB_BUTTON
    _VGP_AVAILABLE = True
except Exception:
    VX360Gamepad = None  # type: ignore
    XUSB_BUTTON = None   # type: ignore
    _VGP_AVAILABLE = False

_gamepad = None

def init_gamepad() -> bool:
    """Ленивая инициализация виртуального геймпада. Требуется установленный ViGEmBus (Windows)."""
    global _gamepad
    if _gamepad is not None:
        return True
    if not _VGP_AVAILABLE:
        print("[Gamepad] vgamepad недоступен. Переходим к клавиатуре.")
        return False
    try:
        _gamepad = VX360Gamepad()
        # небольшая задержка, чтобы устройство появилось в системе
        time.sleep(0.1)
        return True
    except Exception as e:
        print(f"[Gamepad] Не удалось создать виртуальный геймпад: {e}")
        _gamepad = None
        return False

def gp_tap_dpad(direction: str, duration: float = 0.08) -> bool:
    """Короткое нажатие на D-Pad: 'up'|'down'|'left'|'right'."""
    if not init_gamepad():
        return False
    try:
        btn_map = {
            'up': XUSB_BUTTON.XUSB_GAMEPAD_DPAD_UP,
            'down': XUSB_BUTTON.XUSB_GAMEPAD_DPAD_DOWN,
            'left': XUSB_BUTTON.XUSB_GAMEPAD_DPAD_LEFT,
            'right': XUSB_BUTTON.XUSB_GAMEPAD_DPAD_RIGHT,
        }
        b = btn_map.get(direction.lower())
        if b is None:
            return False
        _gamepad.press_button(button=b)
        _gamepad.update()
        time.sleep(max(0.02, duration))
        _gamepad.release_button(button=b)
        _gamepad.update()
        return True
    except Exception as e:
        print(f"[Gamepad] Ошибка D-Pad '{direction}': {e}")
        return False

def gp_tap_button(name: str, duration: float = 0.08) -> bool:
    """Короткое нажатие кнопки: 'A','B','X','Y','START','BACK'."""
    if not init_gamepad():
        return False
    try:
        nm = name.upper()
        btn_map = {
            'A': XUSB_BUTTON.XUSB_GAMEPAD_A,
            'B': XUSB_BUTTON.XUSB_GAMEPAD_B,
            'X': XUSB_BUTTON.XUSB_GAMEPAD_X,
            'Y': XUSB_BUTTON.XUSB_GAMEPAD_Y,
            'START': XUSB_BUTTON.XUSB_GAMEPAD_START,
            'BACK': XUSB_BUTTON.XUSB_GAMEPAD_BACK,
        }
        b = btn_map.get(nm)
        if b is None:
            return False
        _gamepad.press_button(button=b)
        _gamepad.update()
        time.sleep(max(0.02, duration))
        _gamepad.release_button(button=b)
        _gamepad.update()
        return True
    except Exception as e:
        print(f"[Gamepad] Ошибка кнопки '{name}': {e}")
        return False

# --- Опциональный фокус окна (Windows) ---
try:
    import pygetwindow as gw
    _GW_AVAILABLE = True
except Exception:
    gw = None  # type: ignore
    _GW_AVAILABLE = False

def focus_window_by_title_fragment(title_fragment: str, exact: bool = False) -> bool:
    """
    Активирует первое окно, чьё название содержит (или равно) указанной строке.
    Требует pygetwindow на Windows.
    """
    if not _GW_AVAILABLE:
        return False
    try:
        wins = gw.getAllTitles()
        candidates = []
        frag = title_fragment.lower()
        for t in wins:
            if not t:
                continue
            tl = t.lower()
            if (exact and tl == frag) or (not exact and frag in tl):
                candidates.append(t)
        for title in candidates:
            try:
                w = gw.getWindowsWithTitle(title)[0]
                if w.isMinimized:
                    w.restore()
                w.activate()
                time.sleep(0.05)
                return True
            except Exception:
                continue
        return False
    except Exception:
        return False

def focus_any_window(title_fragments) -> bool:
    """
    Перебирает список фрагментов названий окон и активирует первое найденное.
    Возвращает True при успехе.
    """
    if not _GW_AVAILABLE:
        return False
    for frag in title_fragments or []:
        if focus_window_by_title_fragment(str(frag), exact=False):
            return True
    return False

def navigate_and_select_image(
    page,
    target_template_path: str,
    focused_template_path: str,
    navigation_keys: list,
    confidence=0.85,
    timeout=30,
    debug_mode: bool = False,
    use_gamepad: bool = False
) -> bool:
    """
    Навигирует по меню с помощью клавиатуры до тех пор, пока выделенный элемент
    не совпадет с целевым элементом, после чего нажимает Enter.

    Параметры:
    - page: Playwright page object.
    - target_template_path: Путь к шаблону цели (например, кнопка 'Играть').
    - focused_template_path: Путь к шаблону выделенного элемента (например, подсвеченная кнопка).
    - navigation_keys: Список клавиш для навигации (например, ['right', 'down']).
    - confidence: Уверенность для поиска изображений.
    - timeout: Общий таймаут операции.
    - debug_mode: Включить сохранение отладочных изображений.
    - use_gamepad: Использовать ли виртуальный геймпад (vgamepad). При False — клавиатура.
    """
    debug_mode = bool(debug_mode or _GLOBAL_DEBUG_ALWAYS)
    start_time = time.time()

    # Вспомогательная функция поиска: страница или весь экран
    def _seek(path: str, tmo: float, ret_rect: bool, dbg_label: str):
        if page is not None:
            return find_image_on_page(page, path, confidence=confidence, timeout=tmo, return_rect=ret_rect, debug_mode=debug_mode, debug_label=dbg_label)
        return find_image(path, confidence=confidence, timeout=tmo, return_rect=ret_rect, debug_mode=debug_mode)

    # Сначала находим, где наша цель, чтобы знать, куда двигаться
    target_rect = _seek(target_template_path, 5, True, "nav_target")

    if not target_rect:
        print(f"[Navigate] Не удалось найти цель '{os.path.basename(target_template_path)}' на экране.")
        return False

    target_cx = target_rect[0] + target_rect[2] // 2
    target_cy = target_rect[1] + target_rect[3] // 2
    print(f"[Navigate] Цель '{os.path.basename(target_template_path)}' найдена в ({target_cx}, {target_cy}). Начинаем навигацию.")

    last_focused_pos = (-1, -1)
    stuck_counter = 0

    while time.time() - start_time < timeout:
        # Ищем текущий выделенный элемент
        focused_rect = _seek(focused_template_path, 0.5, True, "nav_focus")

        if focused_rect:
            focused_cx = focused_rect[0] + focused_rect[2] // 2
            focused_cy = focused_rect[1] + focused_rect[3] // 2

            # Проверяем, не застряли ли мы
            if focused_cx == last_focused_pos[0] and focused_cy == last_focused_pos[1]:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_focused_pos = (focused_cx, focused_cy)

            if stuck_counter > len(navigation_keys) * 2:
                 print("[Navigate] Навигация застряла. Прерывание.")
                 return False

            # Проверяем, находимся ли мы на цели
            dist_sq = (focused_cx - target_cx)**2 + (focused_cy - target_cy)**2
            # Допуск в пикселях (квадрат расстояния)
            if dist_sq < (target_rect[2] * 0.5)**2: # Допуск в половину ширины цели
                print(f"[Navigate] Цель достигнута! Выделенный элемент в ({focused_cx}, {focused_cy}). Нажимаем Enter.")
                if use_gamepad and init_gamepad():
                    gp_tap_button('A')
                else:
                    press_key('enter')
                time.sleep(1) # Пауза после выбора
                return True

            # Логика навигации (очень простая: пробуем клавиши по очереди)
            key_to_press = navigation_keys[stuck_counter % len(navigation_keys)]
            print(f"[Navigate] Промах. Выделено: ({focused_cx}, {focused_cy}). Цель: ({target_cx}, {target_cy}). Нажимаем '{key_to_press}'")
            if use_gamepad and init_gamepad():
                gp_tap_dpad(key_to_press)
            else:
                press_key(key_to_press)

        else:
            # Если фокус вообще не найден, нажимаем первую навигационную клавишу
            print("[Navigate] Не найден выделенный элемент. Пробуем нажать первую клавишу...")
            if use_gamepad and init_gamepad():
                gp_tap_dpad(navigation_keys[0])
            else:
                press_key(navigation_keys[0])

        time.sleep(0.7) # Пауза между нажатиями

    print("[Navigate] Таймаут навигации.")
    return False