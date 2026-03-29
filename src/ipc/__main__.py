"""
python -m src.ipc — запуск IPC сервера.

УВАГА: При прямому виклику `python -m src.ipc`, Python першим завантажує
src/__init__.py який імпортує get_logger → setup_logging(log_to_console=True).
Це пише лог-повідомлення у stdout ДО нашого коду.

Для IPC серверу використовуйте `python ipc_entry.py` (обгортка що захищає stdout).
"""
from .server import main

if __name__ == "__main__":
    main()
