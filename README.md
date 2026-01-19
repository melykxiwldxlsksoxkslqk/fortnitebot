# EpicBot - Cloud Gaming Bot v2.1

🎮 Бот для автоматизации действий в играх через Xbox Cloud Gaming с использованием компьютерного зрения и машинного обучения.

## ✨ Возможности

- **🦊 Camoufox** — Анти-детект браузер на основе Firefox (обход защит)
- **Браузерная автоматизация** — Playwright + Camoufox для авторизации
- **Компьютерное зрение** — OpenCV + YOLO для распознавания элементов
- **RL-агент** — Обучение с подкреплением через Stable-Baselines3 (опционально)
- **Desktop GUI** — Современный Electron-интерфейс
- **Мульти-аккаунт** — Поддержка нескольких аккаунтов с прокси
- **🔒 Шифрование** — Безопасное хранение паролей (AES-128)

## 📁 Структура проекта

```
fortnitebot/
├── src/                    # Исходный код Python
│   ├── __init__.py         # Экспорт модулей
│   ├── bot_logic.py        # Логика бота и RL-среда
│   ├── browser.py          # Playwright + Camoufox
│   ├── config.py           # Конфигурация и константы
│   ├── db.py               # Работа с SQLite
│   ├── logger.py           # Централизованное логирование
│   ├── main.py             # Точка входа
│   ├── security.py         # Шифрование паролей
│   ├── stream_input.py     # Управление вводом
│   └── vision.py           # Компьютерное зрение
├── desktop/                # Electron GUI
├── tests/                  # Unit-тесты
├── config/                 # Конфигурационные файлы
├── assets/                 # Шаблоны изображений
├── logs/                   # Логи (автоматически)
├── requirements.txt        # Основные зависимости
└── requirements-ml.txt     # ML зависимости (PyTorch, Gym)
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

**С ML (PyTorch, Gym):**
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

### GUI
```bash
python start_desktop.py
```

### CLI
```bash
python -m src.main
```

### Тесты
```bash
pytest
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
browser, page = create_browser(
    proxy="http://user:pass@host:port",  # опционально
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

Редактируйте `src/config.py`:

```python
TIMEOUTS.page_load = 30
VISION.default_confidence = 0.8
RL.attack_with_target_reward = 0.5
BROWSER.headless = False
```

## 📝 Логирование

Логи: `logs/epicbot_YYYY-MM-DD.log`

```python
from src.logger import get_logger
logger = get_logger(__name__)
logger.info("Сообщение")
```

## ⚠️ Отказ от ответственности

Проект в образовательных целях. Использование может нарушать ToS игровых сервисов.