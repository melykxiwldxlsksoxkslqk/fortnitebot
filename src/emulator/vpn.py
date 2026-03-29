"""
Управління JumpJumpVPN всередині емулятора LDPlayer.

Класи:
- VPNRegion: Доступні регіони
- VPNStatus: Статус підключення
- VPNManager: Повне управління VPN (запуск, підключення, перезапуск, таймер)

VPN управляється через ADB команди (tap, swipe, launch/stop app).
Безкоштовна версія JumpJumpVPN дає ~2.5 години підключення.
"""

import time
from enum import Enum
from typing import Optional, Callable
from dataclasses import dataclass, field

from ..core.logger import get_logger
from .config import VPNConfig
from .ldplayer import LDPlayerManager, EmulatorInstance
from .exceptions import VPNError, VPNConnectionError, VPNTimeoutError

logger = get_logger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class VPNStatus(str, Enum):
    """Статус VPN підключення."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    ERROR = "error"
    UNKNOWN = "unknown"


class VPNRegion(str, Enum):
    """Доступні регіони VPN (основні для Xbox Cloud Gaming)."""
    UNITED_STATES = "United States"
    UNITED_KINGDOM = "United Kingdom"
    CANADA = "Canada"
    GERMANY = "Germany"
    FRANCE = "France"
    AUSTRALIA = "Australia"
    JAPAN = "Japan"
    BRAZIL = "Brazil"
    SOUTH_KOREA = "South Korea"
    NETHERLANDS = "Netherlands"


# ============================================================================
# VPN SESSION
# ============================================================================


@dataclass
class VPNSession:
    """Інформація про поточну сесію VPN."""
    started_at: float = 0.0
    region: str = ""
    status: VPNStatus = VPNStatus.DISCONNECTED
    reconnect_count: int = 0

    @property
    def elapsed_minutes(self) -> float:
        """Скільки хвилин минуло з початку сесії."""
        if self.started_at == 0:
            return 0.0
        return (time.time() - self.started_at) / 60.0

    @property
    def is_active(self) -> bool:
        """Чи активна сесія."""
        return self.status == VPNStatus.CONNECTED

    @property
    def remaining_minutes(self) -> float:
        """Скільки хвилин залишилось (з 2.5 годин)."""
        return max(0, 150 - self.elapsed_minutes)


# ============================================================================
# VPN MANAGER
# ============================================================================


class VPNManager:
    """
    Управління JumpJumpVPN в емуляторі LDPlayer.
    
    Відповідальності:
    - Запуск/зупинка VPN додатку
    - Підключення до обраного регіону
    - Відключення та перепідключення
    - Моніторинг часу сесії (2.5 години ліміт)
    - Перезапуск VPN коли час закінчується
    
    Взаємодія з VPN відбувається через ADB (tap на координати UI).
    Координати кнопок налаштовуються через VPNConfig.
    
    Використання:
        vpn = VPNManager(ldplayer_mgr, instance, config)
        vpn.connect("United States")
        # ... гра ...
        if vpn.session.remaining_minutes < 5:
            vpn.reconnect()
    """

    def __init__(
        self,
        ldplayer: LDPlayerManager,
        instance: EmulatorInstance,
        config: Optional[VPNConfig] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self._ldplayer = ldplayer
        self._instance = instance
        self._config = config or VPNConfig()
        self._status_callback = status_callback
        self._session = VPNSession()

        logger.info(f"VPNManager створено для '{instance.name}'")

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def session(self) -> VPNSession:
        """Поточна сесія VPN."""
        return self._session

    @property
    def status(self) -> VPNStatus:
        """Поточний статус VPN."""
        return self._session.status

    @property
    def is_connected(self) -> bool:
        """Чи підключений VPN."""
        return self._session.is_active

    @property
    def config(self) -> VPNConfig:
        """Конфігурація VPN."""
        return self._config

    # ========================================================================
    # ВНУТРІШНІ МЕТОДИ
    # ========================================================================

    def _emit(self, message: str) -> None:
        """Відправляє статус-повідомлення."""
        logger.info(f"[VPN:{self._instance.name}] {message}")
        if self._status_callback:
            try:
                self._status_callback(message)
            except Exception:
                pass

    def _tap(self, x: int, y: int, delay: float = 0.5) -> None:
        """Натискає на координати та чекає."""
        self._ldplayer.adb_tap(self._instance, x, y)
        time.sleep(delay)

    def _is_vpn_running(self) -> bool:
        """Перевіряє, чи запущений VPN додаток."""
        return self._ldplayer.is_app_running(self._instance, self._config.package_name)

    # ========================================================================
    # ЗАПУСК / ЗУПИНКА ДОДАТКУ
    # ========================================================================

    def launch_app(self) -> None:
        """Запускає VPN додаток."""
        self._emit("Запуск VPN додатку...")
        self._ldplayer.launch_app(
            self._instance,
            self._config.package_name,
            self._config.activity_name,
        )
        time.sleep(3)  # Чекаємо завантаження UI

        if not self._is_vpn_running():
            raise VPNError("VPN додаток не запустився")

        self._emit("VPN додаток запущено")

    def stop_app(self) -> None:
        """Зупиняє VPN додаток."""
        self._emit("Зупинка VPN додатку...")
        self._ldplayer.stop_app(self._instance, self._config.package_name)
        self._session.status = VPNStatus.DISCONNECTED
        time.sleep(1)

    # ========================================================================
    # ПІДКЛЮЧЕННЯ / ВІДКЛЮЧЕННЯ
    # ========================================================================

    def connect(self, region: Optional[str] = None) -> None:
        """
        Підключається до VPN серверу.
        
        Args:
            region: Регіон (або default з конфігурації)
            
        Raises:
            VPNConnectionError: Якщо не вдалося підключитися
            VPNTimeoutError: Якщо перевищено таймаут
        """
        region = region or self._config.default_region
        self._emit(f"Підключення VPN: {region}")
        self._session.status = VPNStatus.CONNECTING

        # 1. Запускаємо додаток якщо не запущений
        if not self._is_vpn_running():
            self.launch_app()

        # 2. Вибираємо регіон (натискаємо на кнопку регіону)
        self._select_region(region)

        # 3. Натискаємо кнопку Connect
        self._tap(
            self._config.connect_button_x,
            self._config.connect_button_y,
            delay=2.0,
        )

        # 4. Чекаємо підключення
        self._wait_for_connection()

        # 5. Оновлюємо сесію
        self._session.started_at = time.time()
        self._session.region = region
        self._session.status = VPNStatus.CONNECTED
        self._emit(f"VPN підключено: {region}")

    def _select_region(self, region: str) -> None:
        """Вибирає регіон у VPN додатку."""
        # Натискаємо на кнопку вибору регіону
        self._tap(
            self._config.region_button_x,
            self._config.region_button_y,
            delay=1.0,
        )
        # Тут можна додати скрол та пошук потрібного регіону
        # Для спрощення — натискаємо перший доступний (US зазвичай перший)
        time.sleep(1)
        # Натискаємо на регіон у списку
        self._tap(
            self._config.region_button_x,
            self._config.region_button_y + 100,  # Зсув вниз до першого елемента
            delay=1.0,
        )

    def _wait_for_connection(self) -> None:
        """Очікує встановлення VPN з'єднання."""
        timeout = self._config.connection_timeout
        deadline = time.time() + timeout
        check_interval = 2

        while time.time() < deadline:
            # Перевіряємо мережеве з'єднання через VPN
            result = self._ldplayer.adb_shell(
                self._instance,
                'ip route show | grep -c tun',
            )
            if result.strip() and result.strip() != '0':
                return

            time.sleep(check_interval)

        raise VPNTimeoutError(
            f"Таймаут підключення VPN ({timeout}с). "
            f"Перевірте VPN додаток вручну."
        )

    def disconnect(self) -> None:
        """Відключає VPN."""
        self._emit("Відключення VPN...")
        self._session.status = VPNStatus.DISCONNECTING

        # Натискаємо кнопку Disconnect (та ж кнопка Connect)
        if self._is_vpn_running():
            self._tap(
                self._config.connect_button_x,
                self._config.connect_button_y,
                delay=2.0,
            )

        self._session.status = VPNStatus.DISCONNECTED
        self._emit("VPN відключено")

    def reconnect(self, region: Optional[str] = None) -> None:
        """
        Перезапускає VPN (disconnect → wait → connect).
        
        Використовується коли:
        - Закінчився ліміт часу (2.5 години)
        - Пропало з'єднання
        - Потрібен новий IP
        """
        region = region or self._session.region or self._config.default_region
        self._emit("Перезапуск VPN...")

        # Відключаємо
        self.disconnect()
        time.sleep(self._config.reconnect_delay)

        # Зупиняємо та запускаємо додаток заново
        self.stop_app()
        time.sleep(2)

        # Підключаємо знову
        self.connect(region)
        self._session.reconnect_count += 1
        self._emit(f"VPN перезапущено (спроба #{self._session.reconnect_count})")

    # ========================================================================
    # МОНІТОРИНГ
    # ========================================================================

    def check_and_reconnect_if_needed(self) -> bool:
        """
        Перевіряє стан VPN та перезапускає якщо потрібно.
        
        Повертає True якщо було перезапущено.
        """
        # Перевіряємо ліміт часу (2.5 години безкоштовно)
        if self._session.is_active and self._session.remaining_minutes < 5:
            self._emit(f"VPN ліміт часу: залишилось {self._session.remaining_minutes:.0f} хв")
            self.reconnect()
            return True

        # Перевіряємо чи ще підключено
        if self._session.status == VPNStatus.CONNECTED:
            result = self._ldplayer.adb_shell(
                self._instance,
                'ip route show | grep -c tun',
            )
            if not result.strip() or result.strip() == '0':
                self._emit("VPN з'єднання втрачено, перепідключення...")
                self.reconnect()
                return True

        return False

    def get_session_info(self) -> dict:
        """Повертає інформацію про поточну сесію."""
        return {
            'status': self._session.status.value,
            'region': self._session.region,
            'elapsed_minutes': round(self._session.elapsed_minutes, 1),
            'remaining_minutes': round(self._session.remaining_minutes, 1),
            'reconnect_count': self._session.reconnect_count,
            'is_active': self._session.is_active,
        }
