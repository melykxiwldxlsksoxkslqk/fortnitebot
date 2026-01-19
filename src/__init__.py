"""
EpicBot - Cloud Gaming Automation Bot.

Микросервисная архитектура:
    - core: Базовые компоненты (config, logger, db, security, exceptions)
    - vision: Компьютерное зрение (detection, state, capture, templates, yolo)
    - browser: Браузерная автоматизация (manager, input)
    - bot: Логика бота (logic, runner)
    - ipc: IPC сервер для Electron (server)
"""

__version__ = "3.0.0"
__author__ = "EpicBot Team"

# === Core модуль ===
from .core import (
    # Логирование
    get_logger,
    setup_logging,
    LogContext,
    # Конфигурация
    ROOT_DIR,
    TIMEOUTS,
    VISION,
    RL,
    BROWSER,
    ASSETS,
    DEFAULT_SETTINGS,
    # База данных
    init_db,
    fetch_accounts,
    fetch_proxies,
    get_settings,
    set_settings,
    # Безопасность
    encrypt_password,
    decrypt_password,
    # Исключения
    EpicBotError,
    BadCredentialsError,
    BrowserClosedError,
    TemplateNotFoundError,
    NavigationError,
)

# === Vision модуль ===
from .vision import (
    # Состояния экрана
    ScreenState,
    detect_screen_state,
    wait_for_screen_state,
    # Детекция
    find_template,
    find_template_multi,
    wait_for_template,
    detect_button,
    smart_find_element,
    # Захват
    capture_screen,
    capture_page_bgr,
    # Шаблоны
    load_template,
    # YOLO
    yolo_load_model,
    yolo_detect,
)

# === Browser модуль ===
from .browser import (
    BrowserManager,
    create_browser,
    # Ввод
    press_key,
    press_action,
    click_at,
    type_text,
    click_canvas,
    focus_canvas,
)

# === Bot модуль ===
from .bot import (
    BotLogic,
    BotRunner,
    run_bot,
)

# === IPC модуль ===
from .ipc import (
    main as ipc_main,
    handle_command,
)

__all__ = [
    # Core - Логирование
    "get_logger",
    "setup_logging",
    "LogContext",
    # Core - Конфигурация
    "ROOT_DIR",
    "TIMEOUTS",
    "VISION",
    "RL",
    "BROWSER",
    "ASSETS",
    "DEFAULT_SETTINGS",
    # Core - База данных
    "init_db",
    "fetch_accounts",
    "fetch_proxies",
    "get_settings",
    "set_settings",
    # Core - Безопасность
    "encrypt_password",
    "decrypt_password",
    # Core - Исключения
    "EpicBotError",
    "BadCredentialsError",
    "BrowserClosedError",
    "TemplateNotFoundError",
    "NavigationError",
    # Vision - Состояния
    "ScreenState",
    "detect_screen_state",
    "wait_for_screen_state",
    # Vision - Детекция
    "find_template",
    "find_template_multi",
    "wait_for_template",
    "detect_button",
    "smart_find_element",
    # Vision - Захват
    "capture_screen",
    "capture_page_bgr",
    # Vision - Шаблоны
    "load_template",
    # Vision - YOLO
    "yolo_load_model",
    "yolo_detect",
    # Browser
    "BrowserManager",
    "create_browser",
    "press_key",
    "press_action",
    "click_at",
    "type_text",
    "click_canvas",
    "focus_canvas",
    # Bot
    "BotLogic",
    "BotRunner",
    "run_bot",
    # IPC
    "ipc_main",
    "handle_command",
    # Версия
    "__version__",
]