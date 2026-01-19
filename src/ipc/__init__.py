"""
IPC модуль - межпроцессное взаимодействие.

Содержит:
- server: JSON-RPC сервер для общения с Electron
"""

from .server import main, handle_command

__all__ = [
    'main',
    'handle_command',
]
