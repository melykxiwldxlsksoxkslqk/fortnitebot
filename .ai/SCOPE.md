# Project Scope — EpicBot v4.0

## LDPlayer Emulator Bot — Fortnite XP Farm via Xbox Cloud Gaming

Бот автоматизує фарм XP в Fortnite через LDPlayer Android-емулятор.
Запускає Chrome у емуляторі → Xbox Cloud Gaming → Fortnite.
Використовує ADB макроси для автоматизації, JumpJumpVPN для зміни IP.
Multi-account з паралельним запуском кількох інстансів.

**Підхід: емулятор + макроси. Без браузерної автоматизації на хості.**

---

## DONE

### Core / Інфраструктура
- [x] Структура проєкту: `core/`, `emulator/`
- [x] Конфігурація через dataclasses (`TIMEOUTS`, `EMULATOR`, `EmulatorConfig`)
- [x] SQLite база даних (акаунти, проксі, налаштування)
- [x] Шифрування паролів (Fernet AES, `core/security.py`)
- [x] Централізоване логування (`core/logger.py`)
- [x] Кастомні виключення (`core/exceptions.py`)
- [x] CLI точка входу (`run.py` + `main.py`)

### Емулятор / LDPlayer (OOP)
- [x] `EmulatorConfig` — 7 dataclass-ів (LDPlayer, VPN, APK, Macros, Session, Instance, Emulator)
- [x] `LDPlayerManager` — управління інстансами (створення, клонування, batch, запуск/зупинка)
- [x] `EmulatorInstance` — стан інстансу (ім'я, індекс, is_running, ADB порт)
- [x] `IMEIGenerator` — генерація унікальних IMEI
- [x] ADB команди: `adb_tap`, `adb_swipe`, `adb_key_event`, `adb_text`, `adb_shell`
- [x] Управління додатками: `launch_app`, `stop_app`, `install_apk`

### VPN
- [x] `VPNManager` — JumpJumpVPN через ADB
- [x] `VPNSession` — стан VPN (підключено, регіон, таймер)
- [x] Операції: connect, disconnect, change_region, restart

### APK
- [x] `APKManager` — встановлення та патчинг
- [x] `APKInfo` — метадані APK (пакет, версія, шлях)
- [x] Lucky Patcher модифікація (вирізання реклами з VPN)

### Акаунти
- [x] `AccountData` — дані акаунту (MS email/password, Epic linking status)
- [x] `AccountStorage` — SQLite сховище з шифруванням
- [x] `EmulatorAccountManager` — реєстрація, вибір, оновлення статистики

### Макроси
- [x] `MacroAction` enum (TAP, LONG_TAP, SWIPE, KEY_EVENT, TEXT_INPUT, WAIT, LAUNCH_APP, STOP_APP, SCREENSHOT)
- [x] `MacroStep` dataclass (дія + координати + затримки)
- [x] `MacroSequence` — послідовність кроків з repeat_count, randomize_timing/position
- [x] `MacroComposer` — fluent API для композиції макросів
- [x] `MacroPlayer` — виконання через ADB команди LDPlayer
- [x] `MacroFactory` — готові макроси:
  - `launch_fortnite` — Chrome → Xbox → Fortnite
  - `enter_island_code` — введення коду острова
  - `gameplay` (×45 повторів) — AFK рухи в грі
  - `exit_game` — вихід з Fortnite
  - `toggle_vpn` — увімк/вимк JumpJumpVPN
  - `create_full_session_macro` — повний цикл

### Сесія / Оркестрація
- [x] `SessionState` FSM (IDLE → VPN_CONNECTING → LAUNCHING_GAME → IN_GAME → EXITING → RESTARTING)
- [x] `GameSession` — один цикл VPN→Chrome→макрос
- [x] `GameSessionStats` — статистика сесії
- [x] `SessionOrchestrator` — верхній рівень:
  - `setup_instance()` — створення + APK + патчинг
  - `clone_and_setup()` — клонування інстансу
  - `batch_setup()` — масове створення
  - `start_farming()` — запуск фарму (background thread)
  - `full_setup_and_farm()` — повний потік
  - `stop_all()`, `shutdown_everything()` — graceful shutdown

### Тести
- [x] Тести конфігурації (`tests/test_config.py`) — 7 тестів
- [x] Тести БД (`tests/test_db.py`) — 14 тестів
- [x] Тести емулятора (`tests/test_emulator.py`) — 87 тестів
- [x] **Всього: 108 тестів, всі проходять**

---

## REMOVED (v3.0 → v4.0)

Повністю видалено браузерний підхід:
- ~~`src/browser/`~~ — Playwright/Camoufox
- ~~`src/bot/`~~ — canvas.py, lobby_waiter.py, game_monitor.py, gamepad.py, etc.
- ~~`src/services/`~~ — input_service.py
- ~~`src/ipc/`~~ — JSON-RPC для Electron
- ~~`src/vision/`~~ — OpenCV, YOLO, template matching
- ~~`desktop/`~~ — Electron + Vite GUI
- ~~`requirements-ml.txt`~~ — ML залежності
- ~~`tools/`~~ — train_yolo.py, train_ai_agent.py

---

## TODO

- [ ] Інтеграційні тести (повний цикл з мокнутим LDPlayer)
- [ ] Калібрування координат для різних роздільностей
- [ ] MacroRecorder — запис дій в реальному часі через ADB events
- [ ] Моніторинг / алерти (Telegram-повідомлення)
- [ ] Proxy rotation та автоматична валідація проксі
- [ ] Web GUI для моніторингу (Flask/FastAPI) — опціонально
- [ ] Підтримка кількох ігор (не тільки Fortnite)
