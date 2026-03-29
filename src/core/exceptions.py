"""
Пользовательские исключения для EpicBot.

Централизованное место для всех исключений приложения.
"""


class EpicBotError(Exception):
    """Базовое исключение для всех ошибок EpicBot."""
    pass


class BadCredentialsError(EpicBotError):
    """Выбрасывается при неверных логине/пароле Microsoft."""
    pass


class CodeRequiredError(EpicBotError):
    """Требуется вход по коду (не поддерживается выбранной стратегией)."""
    pass


class TimeoutError(EpicBotError):
    """Превышено время ожидания операции."""
    pass


class LoginError(EpicBotError):
    """Ошибка при входе в аккаунт."""
    pass


class GameLaunchError(EpicBotError):
    """Ошибка при запуске игры."""
    pass


class NavigationError(EpicBotError):
    """Общая ошибка навигации."""
    pass


class ProxyError(EpicBotError):
    """Ошибка прокси-сервера."""
    pass


class DatabaseError(EpicBotError):
    """Ошибка базы данных."""
    pass
