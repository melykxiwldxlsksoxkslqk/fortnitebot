"""
Централизованный модуль логирования для EpicBot.

Предоставляет единую систему логирования для всех модулей проекта
с поддержкой файлов, консоли и структурированного вывода.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler

from .config import LOGS_DIR

# Константы
LOG_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5

# Уровни логирования по имени
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

# Глобальная конфигурация
_initialized = False
_root_logger: Optional[logging.Logger] = None


def _ensure_log_dir() -> str:
    """Создаёт директорию для логов, если не существует."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    return LOGS_DIR


def setup_logging(
    level: str = 'INFO',
    log_to_file: bool = True,
    log_to_console: bool = True,
    filename: Optional[str] = None
) -> logging.Logger:
    """
    Настраивает глобальную систему логирования.
    
    Args:
        level: Уровень логирования ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_to_file: Логировать в файл
        log_to_console: Логировать в консоль
        filename: Имя файла лога (по умолчанию: epicbot_YYYY-MM-DD.log)
    
    Returns:
        Корневой логгер приложения
    """
    global _initialized, _root_logger
    
    if _initialized and _root_logger:
        return _root_logger
    
    # Создаём корневой логгер для приложения
    logger = logging.getLogger('epicbot')
    logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    logger.handlers.clear()
    
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # Консольный вывод
    if log_to_console:
        # На Windows нужно использовать UTF-8 для поддержки всех символов
        try:
            import io
            console_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception:
            console_stream = sys.stdout
        
        console_handler = logging.StreamHandler(console_stream)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
        logger.addHandler(console_handler)
    
    # Файловый вывод
    if log_to_file:
        _ensure_log_dir()
        if not filename:
            filename = f"epicbot_{datetime.now().strftime('%Y-%m-%d')}.log"
        log_path = os.path.join(LOGS_DIR, filename)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUP_COUNT,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)  # Файл получает все сообщения
        logger.addHandler(file_handler)
    
    _initialized = True
    _root_logger = logger
    
    logger.info("Система логирования инициализирована")
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Возвращает логгер для модуля.
    
    Args:
        name: Имя модуля (обычно __name__)
    
    Returns:
        Логгер для модуля
    """
    global _root_logger
    
    if not _initialized:
        setup_logging()
    
    if name:
        # Создаём дочерний логгер
        if name.startswith('src.'):
            name = name[4:]  # Убираем префикс 'src.'
        return logging.getLogger(f'epicbot.{name}')
    
    return _root_logger or logging.getLogger('epicbot')


def set_log_level(level: str) -> None:
    """Изменяет уровень логирования."""
    global _root_logger
    
    if _root_logger:
        _root_logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
        for handler in _root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                handler.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))


class LogContext:
    """Контекстный менеджер для логирования с дополнительным контекстом."""
    
    def __init__(self, logger: logging.Logger, context: str):
        self.logger = logger
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"[{self.context}] Начало")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(f"[{self.context}] Ошибка за {duration:.2f}с: {exc_val}")
        else:
            self.logger.debug(f"[{self.context}] Завершено за {duration:.2f}с")
        return False
