# EpicBot - Cloud Gaming Bot v4.0

🎮 Бот для автоматизации действий в играх через Xbox Cloud Gaming с использованием компьютерного зрения и машинного обучения.

## ✨ Возможности

### Основные
- **🦊 Camoufox** — Анти-детект браузер на основе Firefox (обход защит)
- **Браузерная автоматизация** — Playwright + Camoufox для авторизации
- **Компьютерное зрение** — OpenCV + YOLO для распознавания элементов
- **Desktop GUI** — Современный Electron-интерфейс
- **Мульти-аккаунт** — Поддержка нескольких аккаунтов с прокси
- **🔒 Шифрование** — Безопасное хранение паролей (Fernet AES)

### Новое в v4.0
- **🎯 Canvas Navigation** — Умная навигация в canvas-стриме Xbox Cloud Gaming
- **🔧 Parallel Execution** — Параллельный запуск нескольких ботов с пулом воркеров
- **📊 Enhanced YOLO** — Улучшенный YOLO детектор с кэшированием и отслеживанием
- **📡 Extended IPC** — Расширенный API для Desktop приложения
- **📈 Metrics & Monitoring** — Метрики и мониторинг работы ботов

## 📁 Структура проекта

```
fortnitebot/
├── src/                    # Исходный код Python
│   ├── core/               # 🔧 Базовые компоненты
│   │   ├── config.py       # Конфигурация и константы
│   │   ├── db.py           # Работа с SQLite
│   │   ├── logger.py       # Централизованное логирование
│   │   ├── security.py     # Шифрование паролей
│   │   └── exceptions.py   # Исключения
│   ├── vision/             # 👁️ Компьютерное зрение
│   │   ├── capture.py      # Захват экрана/страницы
│   │   ├── detection.py    # Поиск шаблонов
│   │   ├── state.py        # ScreenState enum
│   │   ├── templates.py    # Кэширование шаблонов
│   │   └── yolo_detector.py # YOLO детекция с кэшированием
│   ├── browser/            # 🌐 Браузерная автоматизация
│   │   ├── manager.py      # BrowserManager (Camoufox/Playwright)
│   │   └── input.py        # Эмуляция ввода
│   ├── bot/                # 🤖 Логика бота
│   │   ├── logic.py        # BotLogic класс
│   │   ├── runner.py       # BotRunner, run_bot()
│   │   ├── canvas.py       # 🆕 Canvas навигация
│   │   ├── parallel.py     # 🆕 Параллельный запуск
│   │   ├── auth.py         # Microsoft авторизация
│   │   ├── xbox.py         # Xbox Cloud Gaming
│   │   ├── island.py       # Поиск островов
│   │   └── ingame.py       # Внутриигровые действия
│   ├── ipc/                # 📡 IPC для Electron
│   │   └── server.py       # JSON-RPC сервер (расширенный)
│   ├── main.py             # Точка входа (legacy)
│   └── run.py              # Новая точка входа
├── desktop/                # Electron GUI
├── tests/                  # Unit-тесты (159+ тестов)
├── config/                 # Конфигурация (БД, настройки)
├── assets/                 # Шаблоны изображений
├── logs/                   # Логи (автоматически)
├── requirements.txt        # Основные зависимости
└── requirements-ml.txt     # ML зависимости (PyTorch, Gym)
```

## 🎯 Canvas Navigation Module

Новый модуль для умной навигации в canvas-элементе Xbox Cloud Gaming стрима.

### Возможности

- **Автоматическое определение состояния экрана** — Lobby, Loading, InGame, Menu
- **Эмуляция геймпада** — Все кнопки Xbox контроллера
- **Виртуальный стик** — Точное управление движением
- **Сохранение снапшотов** — Для отладки и обучения YOLO
- **AFK Prevention** — Автоматическое предотвращение отключения

### Использование

```python
from src.bot.canvas import CanvasNavigator, ScreenState, GamepadButton

# Создание навигатора
navigator = CanvasNavigator(page)

# Проверка и фокус на canvas
await navigator.ensure_focus()

# Определение состояния экрана
state = await navigator.detect_screen_state()
if state == ScreenState.LOBBY:
    print("Мы в лобби!")

# Навигация и ввод
await navigator.press_button(GamepadButton.A)
await navigator.navigate(NavigationDirection.DOWN, count=3)

# Поиск и запуск острова
success = await navigator.search_and_launch_island("1234-5678-9012")

# AFK Prevention
await navigator.run_afk_prevention(duration_minutes=15)
```

### Quick Start функции

```python
from src.bot.canvas import create_navigator, quick_search_island

# Быстрое создание навигатора
navigator = await create_navigator(page)

# Быстрый поиск острова
success = await quick_search_island(page, "1234-5678-9012")
```

## 🔧 Parallel Execution

Модуль для параллельного запуска нескольких ботов.

### Возможности

- **Пул воркеров** — Ограничение одновременных ботов
- **Очередь с приоритетами** — HIGH, NORMAL, LOW
- **Автоматические рестарты** — При ошибках
- **Метрики и статистика** — Uptime, success rate, etc.
- **Graceful shutdown** — Корректное завершение

### Использование

```python
from src.bot.parallel import (
    BotWorkerPool, BotPriority, BotStatus,
    run_bots_parallel, run_bots_async
)

# Вариант 1: Пул воркеров
pool = BotWorkerPool(
    max_workers=5,
    settings={'island_code': '1234-5678-9012'},
    status_callback=lambda login, msg: print(f"[{login}] {msg}")
)

pool.start()

# Добавление ботов
pool.submit(
    account={'login': 'user1', 'password': 'pass1'},
    proxy={'host': '127.0.0.1', 'port': 8080},
    priority=BotPriority.HIGH,
)

# Добавление нескольких
pool.submit_many(accounts, proxies)

# Ожидание завершения
pool.wait_for_completion(timeout=3600)

# Статистика
stats = pool.get_stats()
print(f"Completed: {stats.completed_tasks}/{stats.total_tasks}")

pool.stop()

# Вариант 2: Простая функция
results = run_bots_parallel(
    accounts=[{'login': 'u1', 'password': 'p1'}, ...],
    proxies=[...],
    island_code='1234-5678-9012',
    max_workers=3,
)

# Вариант 3: Асинхронный запуск
results = await run_bots_async(accounts, max_concurrent=5)
```

## 📡 IPC API

Расширенный JSON-RPC API для Desktop приложения.

### Основные команды

| Метод | Описание |
|-------|----------|
| `start` | Запуск всех ботов |
| `stop` | Остановка всех ботов |
| `get_status` | Текущий статус |
| `get_settings` / `save_settings` | Настройки |
| `get_accounts` / `save_accounts` | Аккаунты |
| `get_proxies` / `save_proxies` | Прокси |

### Управление отдельным ботом

| Метод | Описание |
|-------|----------|
| `start_one` | Запуск конкретного бота |
| `stop_one` | Остановка конкретного бота |
| `restart_bot` | Перезапуск бота |
| `get_bot_state` | Детальное состояние |

### Параллельный запуск

| Метод | Описание |
|-------|----------|
| `start_parallel` | Запуск через пул воркеров |
| `stop_parallel` | Остановка пула |
| `get_parallel_stats` | Статистика пула |
| `set_parallel_workers` | Изменение max_workers |

### Vision / YOLO

| Метод | Описание |
|-------|----------|
| `detect_screen_state` | Определение состояния экрана |
| `run_yolo_detection` | YOLO детекция |

### Canvas

| Метод | Описание |
|-------|----------|
| `canvas_press_button` | Нажатие кнопки геймпада |
| `canvas_navigate` | Навигация в меню |
| `canvas_get_screen_state` | Состояние через canvas |

### Метрики

| Метод | Описание |
|-------|----------|
| `get_metrics` | Метрики всех ботов |
| `reset_metrics` | Сброс метрик |

### Система

| Метод | Описание |
|-------|----------|
| `ping` | Проверка связи |
| `get_server_info` | Информация о сервере |
| `get_logs` | Последние логи |

### Пример запроса

```json
{"id": 1, "method": "start_parallel", "params": {"max_workers": 3}}
```

### Пример ответа

```json
{"id": 1, "result": {"ok": true, "started": 5, "max_workers": 3}}
```

## 🔍 YOLO Detector

Улучшенный YOLO детектор с дополнительными возможностями.

### Возможности

- **Кэширование детекций** — Ускорение повторных запросов
- **Адаптивная уверенность** — Автоматическая подстройка threshold
- **Temporal tracking** — Отслеживание объектов между кадрами
- **Fortnite UI классы** — Предопределённые классы для UI элементов
- **Training helpers** — Подготовка данных для обучения

### Предопределённые классы

```python
FORTNITE_UI_CLASSES = {
    'play_button', 'search_icon', 'island_code_input',
    'lobby_background', 'creative_button', 'start_button',
    'loading_spinner', 'menu_button', 'inventory_slot',
    'health_bar', 'shield_bar', 'minimap', ...
}
```

### Использование

```python
from src.vision.yolo_detector import (
    yolo_detect, yolo_detect_ui_elements,
    yolo_detect_game_state, yolo_detect_with_tracking,
)

# Базовая детекция
detections = yolo_detect(image, classes=['play_button'])

# UI элементы
ui_elements = yolo_detect_ui_elements(image)

# Определение состояния игры
state = yolo_detect_game_state(image)  # 'lobby', 'loading', 'ingame', 'menu'

# С отслеживанием
detections = yolo_detect_with_tracking(image, track_id="main")
```

## 🚀 Установка

### 1. Клонирование

```bash
git clone https://github.com/your-repo/fortnitebot.git
cd fortnitebot
```

### 2. Виртуальное окружение

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Зависимости

**Базовая установка:**
```bash
pip install -r requirements.txt
```

**Установка Camoufox (анти-детект браузер):**
```bash
camoufox fetch
```

**С ML (PyTorch, ultralytics):**
```bash
pip install -r requirements-ml.txt
```

### 4. Ассеты

Замените файлы-заглушки в `assets/` на скриншоты:
- `creative_mode_button.png`
- `island_code_button.png`
- `island_code_input_field.png`
- `launch_island_button.png`

## ⚙️ Использование

### GUI (Electron)
```bash
python start_desktop.py
```

### CLI
```bash
python -m src.run --cli
```

### IPC Server (для Electron)
```bash
python -m src.run
```

### Тесты
```bash
# Все тесты
pytest tests/ -v

# Конкретный модуль
pytest tests/test_canvas.py -v
pytest tests/test_parallel.py -v
pytest tests/test_ipc_server.py -v
```

## 🔒 Безопасность

Пароли автоматически шифруются при сохранении.

**Никогда не коммитьте:**
- `config/accounts.txt`
- `config/settings.json`
- `config/epicbot.db`
- `config/.secret_key`

## 🦊 Camoufox — Анти-детект браузер

Проект использует [Camoufox](https://github.com/nickolaj-jepsen/camoufox-python) — форк Firefox с защитой от обнаружения:

- ✅ Обход Cloudflare, DataDome и других защит
- ✅ Эмуляция реальных отпечатков браузера
- ✅ Интеграция с Playwright API

**Использование:**
```python
from src.browser import create_browser, BrowserManager

# Вариант 1: Ручное управление
browser, context, page = create_browser(
    proxy={'server': 'http://host:port'},
    headless=False
)
# ... работа с page ...
browser.close()

# Вариант 2: Контекстный менеджер (рекомендуется)
async with BrowserManager(proxy="...") as bm:
    page = bm.page
    await page.goto("https://example.com")
```

## 📊 Конфигурация

Редактируйте `src/core/config.py`:

```python
TIMEOUTS.page_load = 30
VISION.default_confidence = 0.8
BROWSER.headless = False
```

## 📝 Логирование

Логи: `logs/epicbot_YYYY-MM-DD.log`

```python
from src.core import get_logger
logger = get_logger(__name__)
logger.info("Сообщение")
```

## 📈 Статус тестов

```
159+ тестов
├── test_config.py       # 15 тестов - конфигурация
├── test_db.py           # 33 теста - база данных
├── test_canvas.py       # 36 тестов - canvas навигация
├── test_parallel.py     # 33 теста - параллельный запуск
└── test_ipc_server.py   # 42 теста - IPC сервер
```

## ⚠️ Отказ от ответственности

Проект в образовательных целях. Использование может нарушать ToS игровых сервисов.

## 📄 Лицензия

MIT License