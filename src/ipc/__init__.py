"""
IPC модуль — JSON-RPC сервер для Desktop GUI.

Забезпечує зв'язок між Electron (React UI) та Python (SessionOrchestrator).
Протокол: JSON-RPC 2.0 через stdin/stdout.
"""

from .server import IPCServer, main, handle_command

__all__ = [
    'IPCServer',
    'main',
    'handle_command',
]
