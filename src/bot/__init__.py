"""
Bot модуль - бизнес-логика бота.

Содержит:
- logic: Основная логика работы бота
- runner: Запуск и управление ботами
- auth: Авторизация Microsoft
- xbox: Навигация Xbox Cloud Gaming
- island: Поиск и запуск островов
- ingame: Внутриигровые действия
- canvas: Умная навигация в canvas-стриме
- parallel: Параллельный запуск нескольких ботов
"""

from .logic import BotLogic
from .runner import run_bot, BotRunner
from .auth import microsoft_login, try_login_flow
from .xbox import (
    open_browser,
    navigate_to_xbox,
    click_play_button,
    click_play_with_retries,
    wait_for_stream_connected,
    keep_stream_open,
)
from .island import (
    wait_for_lobby_ui,
    search_and_launch_island_unified,
    search_and_launch_island_canvas,
    open_search_panel,
)
from .ingame import (
    do_active_ingame_actions,
    lock_mouse_into_stream,
    ensure_stream_focus,
    perform_afk_prevention,
)
from .canvas import (
    CanvasNavigator,
    ScreenState,
    GamepadButton,
    NavigationDirection,
    create_navigator,
    quick_search_island,
)
from .parallel import (
    BotWorkerPool,
    BotPriority,
    BotStatus,
    BotTask,
    BotResult,
    PoolStats,
    AsyncBotPool,
    run_bots_parallel,
    run_bots_async,
)

__all__ = [
    # Core
    'BotLogic',
    'run_bot',
    'BotRunner',
    # Auth
    'microsoft_login',
    'try_login_flow',
    # Xbox
    'open_browser',
    'navigate_to_xbox',
    'click_play_button',
    'click_play_with_retries',
    'wait_for_stream_connected',
    'keep_stream_open',
    # Island
    'wait_for_lobby_ui',
    'search_and_launch_island_unified',
    'search_and_launch_island_canvas',
    'open_search_panel',
    # Ingame
    'do_active_ingame_actions',
    'lock_mouse_into_stream',
    'ensure_stream_focus',
    'perform_afk_prevention',
    # Canvas Navigation
    'CanvasNavigator',
    'ScreenState',
    'GamepadButton',
    'NavigationDirection',
    'create_navigator',
    'quick_search_island',
    # Parallel
    'BotWorkerPool',
    'BotPriority',
    'BotStatus',
    'BotTask',
    'BotResult',
    'PoolStats',
    'AsyncBotPool',
    'run_bots_parallel',
    'run_bots_async',
]
