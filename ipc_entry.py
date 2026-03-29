#!/usr/bin/env python3
"""
Точка входу IPC серверу для Electron.

!! КРИТИЧНО: Цей скрипт виконується ДО імпорту будь-яких модулів src.
Python при `python -m src.ipc` завантажує src/__init__.py який імпортує
get_logger → setup_logging(log_to_console=True) → пише лог у stdout.
А stdout — це IPC канал (JSON-RPC).

Рішення:
    1) Клонуємо fd1 (stdout) та fd0 (stdin) через os.dup()
    2) Перенаправляємо fd1 у devnull (захищаємо від логів під час імпорту)
    3) Будуємо IPC потоки на клонованих fd
    4) Імпортуємо src (логи підуть у devnull — безпечно)
    5) Передаємо IPC потоки серверу та запускаємо
"""
import sys
import os
import io

# =========================================================================
# 1) Клонуємо справжній stdout/stdin
# =========================================================================
_real_stdout_fd = os.dup(1)
_real_stdin_fd = os.dup(0)

# =========================================================================
# 2) Перенаправляємо fd1 у devnull + заміняємо sys.stdout
# =========================================================================
_devnull_fd = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull_fd, 1)
os.close(_devnull_fd)

# Заміняємо sys.stdout на devnull-обгортку (щоб logging не отримав OSError при flush)
sys.stdout = open(os.devnull, 'w', encoding='utf-8')

# =========================================================================
# 3) Будуємо IPC потоки на збережених fd
# =========================================================================
_IPC_STDOUT = io.TextIOWrapper(
    io.BufferedWriter(io.FileIO(_real_stdout_fd, 'w', closefd=False)),
    encoding='utf-8', errors='replace', line_buffering=True,
)
_IPC_STDIN = io.TextIOWrapper(
    io.BufferedReader(io.FileIO(_real_stdin_fd, 'r', closefd=False)),
    encoding='utf-8', errors='replace',
)

# =========================================================================
# 4) Тепер безпечно імпортуємо — всі логи підуть у devnull
# =========================================================================
from src.ipc import server as _server_mod
from src.ipc.server import main

# =========================================================================
# 5) Передаємо IPC потоки серверу та запускаємо
# =========================================================================
_server_mod._IPC_STDOUT = _IPC_STDOUT
_server_mod._IPC_STDIN = _IPC_STDIN

if __name__ == "__main__":
    main()
