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

# Константы
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
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
    os.makedirs(LOG_DIR, exist_ok=True)
    return LOG_DIR


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
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
        logger.addHandler(console_handler)
    
    # Файловый вывод
    if log_to_file:
        _ensure_log_dir()
        if not filename:
            filename = f"epicbot_{datetime.now().strftime('%Y-%m-%d')}.log"
        log_path = os.path.join(LOG_DIR, filename)
        
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


def get_logger(name: str) -> logging.Logger:
    """
    Получает логгер для указанного модуля.
    
    Args:
        name: Имя модуля (обычно __name__)
    
    Returns:
        Логгер, дочерний от корневого epicbot
    
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Сообщение")
    """
    global _initialized
    
    # Автоматическая инициализация при первом вызове
    if not _initialized:
        setup_logging()
    
    # Убираем префикс src. если есть
    if name.startswith('src.'):
        name = name[4:]
    
    return logging.getLogger(f'epicbot.{name}')


class LoggerMixin:
    """
    Миксин для добавления логирования в классы.
    
    Example:
        >>> class MyClass(LoggerMixin):
        ...     def do_something(self):
        ...         self.logger.info("Делаю что-то")
    """
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_exception(logger: logging.Logger, message: str = "Произошла ошибка") -> None:
    """
    Логирует текущее исключение с полным traceback.
    
    Использовать внутри блока except:
        >>> try:
        ...     risky_operation()
        ... except Exception:
        ...     log_exception(logger, "Ошибка в risky_operation")
    """
    logger.exception(message)


# Упрощённые функции для быстрого использования
def debug(message: str, logger_name: str = 'main') -> None:
    """Быстрое логирование DEBUG."""
    get_logger(logger_name).debug(message)


def info(message: str, logger_name: str = 'main') -> None:
    """Быстрое логирование INFO."""
    get_logger(logger_name).info(message)


def warning(message: str, logger_name: str = 'main') -> None:
    """Быстрое логирование WARNING."""
    get_logger(logger_name).warning(message)


def error(message: str, logger_name: str = 'main') -> None:
    """Быстрое логирование ERROR."""
    get_logger(logger_name).error(message)


def critical(message: str, logger_name: str = 'main') -> None:
    """Быстрое логирование CRITICAL."""
    get_logger(logger_name).critical(message)
