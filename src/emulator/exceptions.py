"""
Виключення для модуля емулятора.

Ієрархія виключень побудована аналогічно core.exceptions.
"""

from ..core.exceptions import EpicBotError


class EmulatorError(EpicBotError):
    """Базове виключення для всіх помилок емулятора."""
    pass


class LDPlayerError(EmulatorError):
    """Помилка при роботі з LDPlayer."""
    pass


class LDPlayerNotFoundError(LDPlayerError):
    """LDPlayer не знайдено за вказаним шляхом."""
    pass


class InstanceNotFoundError(LDPlayerError):
    """Інстанс емулятора не знайдено."""
    pass


class InstanceAlreadyRunningError(LDPlayerError):
    """Інстанс вже запущений."""
    pass


class VPNError(EmulatorError):
    """Помилка при роботі з VPN."""
    pass


class VPNConnectionError(VPNError):
    """Не вдалося підключитися до VPN."""
    pass


class VPNTimeoutError(VPNError):
    """Таймаут підключення VPN."""
    pass


class MacroError(EmulatorError):
    """Помилка при роботі з макросами."""
    pass


class MacroRecordError(MacroError):
    """Помилка запису макросу."""
    pass


class MacroPlaybackError(MacroError):
    """Помилка відтворення макросу."""
    pass


class MacroNotFoundError(MacroError):
    """Макрос не знайдено."""
    pass


class APKError(EmulatorError):
    """Помилка при роботі з APK."""
    pass


class APKInstallError(APKError):
    """Помилка встановлення APK."""
    pass


class APKPatchError(APKError):
    """Помилка модифікації APK через Lucky Patcher."""
    pass


class AccountCreationError(EmulatorError):
    """Помилка при створенні акаунту."""
    pass


class MicrosoftAccountError(AccountCreationError):
    """Помилка при створенні акаунту Microsoft."""
    pass


class EpicAccountError(AccountCreationError):
    """Помилка при створенні акаунту Epic Games."""
    pass


class XboxLinkError(AccountCreationError):
    """Помилка при прив'язці акаунту до Xbox."""
    pass


class SessionError(EmulatorError):
    """Помилка ігрового сеансу."""
    pass


class SessionTimeoutError(SessionError):
    """Таймаут ігрового сеансу."""
    pass


class SessionInterruptedError(SessionError):
    """Сеанс було перервано (опитування, помилка макросу тощо)."""
    pass
