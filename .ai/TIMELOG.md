# AI Development Timelog — EpicBot

## 2026-03-29

### Session 1 — Emulator Module (OOP)

Створено повний модуль `src/emulator/` (7 файлів, 87 тестів):

- `config.py` — 7 dataclass-ів (EmulatorConfig, LDPlayerConfig, VPNConfig, APKConfig, MacroConfig, SessionConfig, InstanceConfig)
- `ldplayer.py` — LDPlayerManager, EmulatorInstance, IMEIGenerator
- `vpn.py` — VPNManager, VPNSession
- `apk.py` — APKManager, APKInfo
- `accounts.py` — AccountData, AccountStorage (SQLite), EmulatorAccountManager
- `macros.py` — MacroAction, MacroStep, MacroSequence, MacroComposer, MacroPlayer, MacroFactory
- `session.py` — SessionState FSM, GameSession, SessionOrchestrator
- `exceptions.py` — 13 класів виключень

Всі 87 тестів пройшли.

### Session 2 — Vision/AI Removal

Видалено всі залишки vision/AI коду:
- `src/vision/` (8 файлів) — OpenCV, YOLO, template matching
- `src/bot/ai_agent.py`, `config/ai_agent/`, `config/yolo/`
- `requirements-ml.txt`, `yolo26n.pt`, `yolov8n.pt`
- Очищено core/config.py, core/exceptions.py, core/__init__.py

## 2026-03-30

### Session 3 — Архітектурний пивот: браузер → емулятор

**РІШЕННЯ:** Повністю відмовляємось від браузерного підходу. Працюємо ТІЛЬКИ через LDPlayer.

**Видалено:**
- `src/bot/` — 11 файлів (canvas.py, lobby_waiter.py, auth.py, xbox.py, island.py, ingame.py, runner.py, logic.py, parallel.py, gamepad.py, game_monitor.py)
- `src/browser/` — manager.py, input.py, extension/
- `src/services/` — base.py, input_service.py
- `src/ipc/` — server.py, __main__.py
- `desktop/` — Electron + Vite GUI
- `browser-profile/`, `browser-profiles/`, `chrome-profile/`, `chromedriver-win64/`
- `start_desktop.py`, `tsconfig.json`
- `tools/train_yolo.py`, `tools/train_ai_agent.py`
- `tests/test_browser.py`, `tests/test_canvas.py`, `tests/test_ipc_server.py`, `tests/test_parallel.py`

**Оновлено:**
- `src/__init__.py` — тільки core + emulator (v4.0)
- `src/main.py` — SessionOrchestrator entry point
- `src/run.py` — CLI з argparse (--island, --max-instances, --ldplayer)
- `src/core/config.py` — BrowserConfig → EmulatorDefaults, очищено DEFAULT_SETTINGS
- `src/core/exceptions.py` — видалено BrowserClosedError, IslandNavigationError
- `src/core/__init__.py` — BROWSER → EMULATOR, оновлено exports
- `src/emulator/__init__.py` — додано MacroFactory в exports
- `tests/test_config.py` — BROWSER → EMULATOR, виправлено syntax error
- `requirements.txt` — видалено playwright, camoufox, opencv, vgamepad, pyautogui, etc.
- `.ai/RULES.md` — нова архітектура (core/ + emulator/)
- `.ai/SCOPE.md` — повний перепис під емуляторний підхід

**Результат:** 108 тестів, всі проходять (7 config + 14 db + 87 emulator).

### Session 4 — IPC Server + Desktop GUI (React/Electron)

**Створено `src/ipc/` — JSON-RPC 2.0 сервер (3 файли):**
- `__init__.py` — exports: IPCServer, main, handle_command
- `__main__.py` — `python -m src.ipc` entry point
- `server.py` — IPCServer клас з 20 командами:
  - Global: ping, get_version, get_status
  - Accounts: get_accounts, add_account, delete_account, import_accounts
  - Instances: list_instances, setup_instance, clone_instance, remove_instance
  - Farm: start_farm, stop_farm, stop_all, shutdown_all
  - Settings: get_settings, set_settings, get_emulator_config, set_emulator_config
  - Logs: get_recent_logs

**Створено `desktop/` — Electron + React + Vite + Tailwind GUI:**
- **Electron shell (2 файли):**
  - `electron/main.ts` — spawns Python IPC, JSON-RPC bridge, window management
  - `electron/preload.ts` — contextBridge з IPC API для renderer process
- **React UI (13 файлів):**
  - `src/main.tsx` — ReactDOM entry
  - `src/App.tsx` — layout + BrowserRouter з 5 маршрутами
  - `src/pages/Dashboard.tsx` — огляд статусу (stats, active instances, events)
  - `src/pages/Instances.tsx` — CRUD LDPlayer інстансів (create, clone, start/stop, remove)
  - `src/pages/Accounts.tsx` — CRUD акаунтів (add, delete, bulk import)
  - `src/pages/Settings.tsx` — налаштування (island code, time, VPN, theme)
  - `src/pages/Logs.tsx` — real-time лог viewer з фільтром та auto-scroll
  - `src/components/ui.tsx` — PageHeader, Card, StatCard, Button, Badge, Spinner, EmptyState
  - `src/lib/types.ts` — TypeScript типи (Account, Instance, BotStatus, Settings, etc.)
  - `src/lib/ipc.ts` — typed IPC client wrapper (20 методів)
  - `src/lib/hooks.ts` — React hooks: useIPC, useStatus, useEvents, useConnection
  - `src/index.css` — Tailwind + custom styles (dark theme, scrollbar, animations)
  - `src/vite-env.d.ts` — Vite env types
- **Config (5 файлів):**
  - `package.json`, `tsconfig.json`, `tsconfig.electron.json`
  - `vite.config.ts`, `tailwind.config.js`, `postcss.config.js`

**Тести `tests/test_ipc.py` — 45 тестів:**
- TestJSONRPCHelpers (7) — _ok, _error, _notification
- TestIPCServerHandle (8) — routing, ping, version, params, errors
- TestAccountCommands (7) — get, add, delete, import (bulk, comments, pipe separator)
- TestSettingsCommands (2) — get, set
- TestInstanceCommands (6) — list, list with session, setup, clone, remove found/not found
- TestFarmCommands (5) — start, start not found, stop, stop_all, shutdown
- TestEmulatorConfigCommands (2) — get, set
- TestLogCommands (3) — no file, read, default count
- TestErrorHandling (3) — EmulatorError, RuntimeError, handle_command standalone
- TestMethodRegistry (2) — all 20 methods registered, count check

**Оновлено:**
- `src/__init__.py` — v4.0 + IPC exports (IPCServer, handle_command)
- Видалено застарілі тести: `test_ipc_server.py` (стара browser-based IPC), `test_canvas.py` (canvas navigator)

**Результат:** 153 тести, всі проходять (7 config + 14 db + 87 emulator + 45 ipc).

### Session 5 — UI: перевод на русский + исправление ошибок

**Исправлены 3 ошибки:**

1. **`Objects are not valid as a React child`** — `StatCard` в `ui.tsx` рендерил `value` напрямую как `{value}`. Если IPC возвращал `null`/`{}` вместо числа, React бросал ошибку. FIX: добавлена проверка `typeof value` — если не string/number, показывается `«—»`.

2. **React Router Future Flag Warnings** (`v7_startTransition`, `v7_relativeSplatPath`) — `BrowserRouter` в `App.tsx` не передавал `future` флаги. FIX: добавлен prop `future={{ v7_startTransition: true, v7_relativeSplatPath: true }}`.

3. **Electron Security Warning (Insecure CSP)** — не был установлен Content-Security-Policy. FIX: в `electron/main.ts` добавлен `session.defaultSession.webRequest.onHeadersReceived()` с правильной CSP-политикой (запрет `unsafe-eval`, разрешение `'self'`, `ws://localhost`, `http://localhost`).

**Перевод на русский язык (13 файлов):**
- `src/App.tsx` — навигация: Панель, Инстансы, Аккаунты, Настройки, Логи; статус подключения
- `src/components/ui.tsx` — комментарии, вынесены цветовые карты в константы (DRY), экспорт типов, добавлен `SettingField`
- `src/pages/Dashboard.tsx` — весь текст на русском, вынесена функция `instanceDotColor()`
- `src/pages/Instances.tsx` — весь текст на русском, вынесена `farmBadgeColor()`, импорт `BadgeColor`
- `src/pages/Accounts.tsx` — весь текст на русском
- `src/pages/Settings.tsx` — весь текст на русском, `SettingField` импортируется из `ui.tsx` (DRY)
- `src/pages/Logs.tsx` — весь текст на русском, вынесена `getLogColor()` из компонента
- `src/lib/ipc.ts` — комментарии на русском
- `src/lib/hooks.ts` — комментарии на русском
- `src/lib/types.ts` — комментарии на русском
- `electron/main.ts` — комментарии на русском + CSP fix
- `electron/preload.ts` — комментарии на русском

**Рефакторинг по OOP/SOLID:**
- **DRY:** цветовые карты (`STAT_COLOR_MAP`, `BADGE_COLOR_MAP`, `BUTTON_VARIANT_MAP`, `BUTTON_SIZE_MAP`) вынесены в модульные константы вместо inline-объектов
- **DRY:** `SettingField` вынесен из `Settings.tsx` в `ui.tsx` — переиспользуемый компонент
- **SRP:** `instanceDotColor()`, `farmBadgeColor()`, `getLogColor()` — вынесены из компонентов в отдельные функции
- **Open/Closed:** экспортированы типы `StatColor`, `BadgeColor`, `ButtonVariant`, `ButtonSize` для типобезопасного расширения

**Сборка:** Vite build ✓, TypeScript (electron) ✓ — без ошибок.

### Session 6 — Централизованная тема: убран весь хардкод Tailwind из компонентов

**Проблема:** Все 7 UI-файлов содержали хардкод Tailwind-строки прямо в JSX (`className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm..."` и т.д.). Стили были размазаны по компонентам — невозможно менять тему в одном месте.

**Решение:** Создана централизованная тема `src/lib/theme.ts` — единственный источник правды для всех стилей. Все компоненты переписаны на импорт из неё.

---

**Создан `src/lib/theme.ts` (~220 строк) — тема-конфиг:**

7 цветовых палитр (`as const`):
- `statColors` — brand / green / yellow / red (StatCard иконки)
- `badgeColors` — gray / green / yellow / red / blue (Badge)
- `buttonVariants` — primary / secondary / danger / ghost (Button)
- `buttonSizes` — sm / md / lg (Button)
- `dotColors` — active / warning / idle (индикаторы инстансов)
- `logColors` — error / warning / debug / success / default (строки логов)
- `connectionColors` — online / offline (статус бэкенда)

4 экспортируемых типа:
- `StatColor`, `BadgeColor`, `ButtonVariant`, `ButtonSize`

Объект `theme` с 16 секциями:
- `layout` — root, page, pageFullHeight, center, grid2, grid3, grid4, main
- `sidebar` — root, logo, logoTitle, logoSubtitle, nav, navItem, navItemActive, navItemInactive, footer, footerContent
- `card` — base, hover, noPadding, title
- `header` — root, title, subtitle, actions
- `button` — base
- `badge` — base
- `stat` — wrapper, iconBox, value, label
- `input` — base, full, textarea
- `empty` — root, icon, title, description, action
- `setting` — root, info, label, description, control
- `instance` — row, name, state, dot, meta, actions, cardHeader, cardHeaderLeft
- `account` — row, email, password, divider
- `logs` — container, line, emptyRoot
- `event` — list, item
- `spinner` — root
- `text` — muted, brand, white, red, mono

5 хелпер-функций:
- `getInstanceDotColor(farmState, status)` → Tailwind-класс точки
- `getFarmBadgeColor(state)` → `BadgeColor`
- `getLogLineColor(line)` → Tailwind-класс цвета строки
- `getNavLinkClass(isActive)` → полный className для NavLink
- `safeDisplayValue(value)` → защита от объектов в React children

---

**Переписан `src/components/ui.tsx`:**
- Удалены локальные константы: `STAT_COLOR_MAP`, `BADGE_COLOR_MAP`, `BUTTON_VARIANT_MAP`, `BUTTON_SIZE_MAP`
- Все компоненты используют `theme.*` и палитры из `theme.ts`
- Типы реэкспортируются из `theme.ts` для обратной совместимости
- `StatCard` → `safeDisplayValue()` из theme.ts
- `Button` → `buttonVariants[variant]` + `buttonSizes[size]` из theme.ts
- `Badge` → `badgeColors[color]` из theme.ts
- Все `className` строки заменены на `theme.stat.*`, `theme.card.*`, `theme.header.*`, etc.

**Переписан `src/App.tsx`:**
- Импорт: `theme`, `connectionColors`, `getNavLinkClass`
- Sidebar: `theme.sidebar.root`, `theme.sidebar.logo`, `theme.sidebar.logoTitle`, `theme.sidebar.logoSubtitle`
- Навигация: `getNavLinkClass(isActive)` вместо inline тернарника
- Статус: `connectionColors.online` / `connectionColors.offline`
- Layout: `theme.layout.root`, `theme.layout.main`

**Переписан `src/pages/Dashboard.tsx`:**
- Удалена локальная `instanceDotColor()` → `getInstanceDotColor()` из theme.ts
- `theme.layout.center` (спиннер), `theme.layout.page`, `theme.layout.grid4`, `theme.layout.grid2`
- `theme.card.title`, `theme.instance.row`, `theme.instance.dot`, `theme.instance.name`, `theme.instance.state`
- `theme.instance.cardHeaderLeft` (flex row для точки + имени)
- `theme.event.list`, `theme.event.item` (список событий)
- `theme.text.muted` (заглушки при пустых данных)

**Переписан `src/pages/Instances.tsx`:**
- Удалена локальная `farmBadgeColor()` → `getFarmBadgeColor()` из theme.ts
- Удалён импорт `BadgeColor` из ui.tsx (больше не нужен — тип в theme.ts)
- `theme.layout.center`, `theme.layout.page`, `theme.layout.grid3`
- `theme.header.actions`, `theme.input.base` (поле ввода имени)
- `theme.card.hover`, `theme.text.brand`, `theme.text.white`, `theme.text.red`
- `theme.instance.meta`, `theme.instance.actions`, `theme.instance.cardHeader`, `theme.instance.cardHeaderLeft`

**Переписан `src/pages/Accounts.tsx`:**
- `theme.layout.center`, `theme.layout.page`, `theme.header.actions`
- `theme.input.full` (email + password), `theme.input.textarea` (массовый импорт)
- `theme.setting.label` (заголовок импорта)
- `theme.account.divider`, `theme.account.row`, `theme.account.email`, `theme.account.password`
- `theme.text.red` (иконка удаления)

**Переписан `src/pages/Settings.tsx`:**
- `theme.layout.center`, `theme.layout.page`, `theme.layout.grid2`
- `theme.header.actions`, `theme.card.title`
- Все `className="input-field"` заменены на `theme.input.base`
- Числовые поля: `${theme.input.base} w-24`

**Переписан `src/pages/Logs.tsx`:**
- Удалена локальная `getLogColor()` → `getLogLineColor()` из theme.ts
- `theme.layout.center`, `theme.layout.pageFullHeight`
- `theme.header.actions`, `theme.input.base` (фильтр)
- `theme.card.noPadding`, `theme.logs.container`, `theme.logs.line`, `theme.logs.emptyRoot`

---

**Итого изменений:** 8 файлов (1 создан + 7 переписаны)

| Файл | Удалено инлайн-классов | Заменено на |
|---|---|---|
| `src/lib/theme.ts` | — | СОЗДАН: 220 строк, 16 секций, 7 палитр, 5 хелперов |
| `src/components/ui.tsx` | 4 локальные карты + все className | `theme.*` + палитры из theme.ts |
| `src/App.tsx` | 8 className строк | `theme.sidebar.*`, `theme.layout.*`, `getNavLinkClass()` |
| `src/pages/Dashboard.tsx` | `instanceDotColor()` + 9 className | `getInstanceDotColor()` + `theme.*` |
| `src/pages/Instances.tsx` | `farmBadgeColor()` + 8 className | `getFarmBadgeColor()` + `theme.*` |
| `src/pages/Accounts.tsx` | 7 className строк | `theme.input.*`, `theme.account.*` |
| `src/pages/Settings.tsx` | 6 `input-field` + 3 card title | `theme.input.base`, `theme.card.title` |
| `src/pages/Logs.tsx` | `getLogColor()` + 6 className | `getLogLineColor()` + `theme.logs.*` |

**Принципы OOP/SOLID:**
- **Single Responsibility:** `theme.ts` = стили, компоненты = структура + логика
- **Open/Closed:** добавить новый цвет/секцию в theme.ts — не трогая компоненты
- **DRY:** ни одна Tailwind-строка не повторяется в компонентах — всё из одного файла

**Сборка:** Vite build ✓, TypeScript (electron) ✓ — 0 ошибок.
