"""
Оркестрація повного ігрового сеансу фарму Fortnite.

Класи:
- SessionState: Стан сеансу (FSM)
- GameSession: Один ігровий сеанс (VPN → гра → вихід)
- SessionOrchestrator: Управління безкінечним циклом фарму

Повний потік:
1. Перший запуск (ручний): відкрити VPN → Connect → відкрити Chrome (потрібна сторінка)
2. Запуск макросу: [ввід коду + геймплей + вихід] × 2 + перезапуск VPN
3. Безкінечне повторення
4. Моніторинг: опитування, помилки макросу, відключення VPN
"""

import time
import threading
from enum import Enum
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field

from ..core.logger import get_logger
from .config import (
    EmulatorConfig,
    SessionConfig,
    MacroConfig,
)
from .ldplayer import LDPlayerManager, EmulatorInstance, InstanceStatus
from .vpn import VPNManager, VPNStatus
from .apk import APKManager
from .accounts import AccountData, EmulatorAccountManager, AccountStorage
from .macros import MacroFactory, MacroComposer, MacroPlayer
from .exceptions import (
    SessionError,
    SessionTimeoutError,
    SessionInterruptedError,
    VPNError,
    MacroError,
)

logger = get_logger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class SessionState(str, Enum):
    """Стан ігрового сеансу (State Machine)."""
    IDLE = "idle"                        # Не запущено
    INITIALIZING = "initializing"        # Ініціалізація
    VPN_CONNECTING = "vpn_connecting"    # Підключення VPN
    CHROME_OPENING = "chrome_opening"    # Відкриття Chrome
    GAME_LAUNCHING = "game_launching"    # Запуск Fortnite
    MACRO_RUNNING = "macro_running"      # Виконання макросу
    VPN_RECONNECTING = "vpn_reconnecting"  # Перезапуск VPN
    GAME_EXITING = "game_exiting"        # Вихід з гри
    COOLDOWN = "cooldown"                # Пауза між сесіями
    ERROR = "error"                      # Помилка
    STOPPED = "stopped"                  # Зупинено


# ============================================================================
# GAME SESSION
# ============================================================================


@dataclass
class GameSessionStats:
    """Статистика одного ігрового сеансу."""
    started_at: float = 0.0
    finished_at: float = 0.0
    games_played: int = 0
    vpn_reconnects: int = 0
    errors: int = 0
    total_macro_steps: int = 0

    @property
    def duration_minutes(self) -> float:
        """Тривалість сесії в хвилинах."""
        if self.started_at == 0:
            return 0.0
        end = self.finished_at or time.time()
        return (end - self.started_at) / 60.0


class GameSession:
    """
    Один ігровий сеанс фарму.
    
    Об'єднує VPN + Chrome + макроси в єдиний потік:
    1. Підключити VPN
    2. Відкрити Chrome → Xbox → Fortnite
    3. Запустити макрос (гра)
    4. Вихід
    
    Використання:
        session = GameSession(ldplayer, instance, account, config)
        session.start()  # Блокуючий виклик
    """

    def __init__(
        self,
        ldplayer: LDPlayerManager,
        instance: EmulatorInstance,
        account: AccountData,
        config: Optional[EmulatorConfig] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None,
    ):
        self._ldplayer = ldplayer
        self._instance = instance
        self._account = account
        self._config = config or EmulatorConfig()
        self._status_callback = status_callback
        self._stop_check = stop_check or (lambda: False)

        self._state = SessionState.IDLE
        self._stats = GameSessionStats()

        # Підмодулі
        self._vpn = VPNManager(
            ldplayer, instance, self._config.vpn, status_callback,
        )
        self._macro_player = MacroPlayer(
            ldplayer, instance, self._config.macros,
            status_callback, stop_check,
        )
        self._macro_factory = MacroFactory(
            resolution_width=instance.config.resolution_width,
            resolution_height=instance.config.resolution_height,
            island_code=self._config.session.island_code,
        )

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def state(self) -> SessionState:
        """Поточний стан."""
        return self._state

    @property
    def stats(self) -> GameSessionStats:
        """Статистика сесії."""
        return self._stats

    @property
    def vpn(self) -> VPNManager:
        """VPN менеджер."""
        return self._vpn

    # ========================================================================
    # ВНУТРІШНІ
    # ========================================================================

    def _emit(self, message: str) -> None:
        """Відправляє статус."""
        logger.info(f"[Session:{self._instance.name}] {message}")
        if self._status_callback:
            try:
                self._status_callback(message)
            except Exception:
                pass

    def _should_stop(self) -> bool:
        try:
            return self._stop_check()
        except Exception:
            return False

    def _set_state(self, state: SessionState) -> None:
        """Змінює стан."""
        old = self._state
        self._state = state
        logger.debug(f"Session state: {old.value} → {state.value}")

    # ========================================================================
    # КРОКИ СЕСІЇ
    # ========================================================================

    def _step_connect_vpn(self) -> None:
        """Крок 1: Підключення VPN."""
        self._set_state(SessionState.VPN_CONNECTING)
        self._emit("Крок 1: Підключення VPN...")

        region = self._config.vpn.default_region
        self._vpn.connect(region)

        self._emit(f"VPN підключено: {region}")

    def _step_open_chrome(self) -> None:
        """Крок 2: Відкриття Chrome з Xbox Cloud Gaming."""
        self._set_state(SessionState.CHROME_OPENING)
        self._emit("Крок 2: Відкриття Chrome...")

        url = self._config.session.xbox_play_url
        self._ldplayer.adb_shell(
            self._instance,
            f'am start -a android.intent.action.VIEW -d "{url}" com.android.chrome',
        )
        time.sleep(5)

        self._emit("Chrome відкрито")

    def _step_run_macro(self) -> None:
        """Крок 3: Запуск головного макросу."""
        self._set_state(SessionState.MACRO_RUNNING)
        self._emit("Крок 3: Запуск макросу...")

        # Створюємо повний макрос сесії
        composer = self._macro_factory.create_full_session_macro(self._config.macros)

        # Відтворюємо
        success = self._macro_player.play_composed(composer)

        if not success and not self._should_stop():
            self._stats.errors += 1
            raise SessionInterruptedError("Макрос завершився з помилкою")

    def _step_reconnect_vpn(self) -> None:
        """Перезапуск VPN між ігровими сесіями."""
        self._set_state(SessionState.VPN_RECONNECTING)
        self._emit("Перезапуск VPN...")

        self._vpn.reconnect()
        self._stats.vpn_reconnects += 1

        delay = self._config.session.vpn_restart_delay_seconds
        time.sleep(delay)

    # ========================================================================
    # ЗАПУСК СЕСІЇ
    # ========================================================================

    def start(self) -> GameSessionStats:
        """
        Запускає повний ігровий сеанс.
        
        Потік:
        1. VPN connect
        2. Chrome → Xbox → Fortnite
        3. Macro loop (enter code → gameplay × 45 → exit) × 2 → toggle VPN → repeat ∞
        
        Returns:
            Статистика сесії
        """
        self._set_state(SessionState.INITIALIZING)
        self._stats.started_at = time.time()

        self._emit(f"=== Сесія для {self._account.ms_email} ===")

        try:
            # 1. VPN
            self._step_connect_vpn()

            if self._should_stop():
                return self._finalize()

            # 2. Chrome
            self._step_open_chrome()

            if self._should_stop():
                return self._finalize()

            # 3. Макрос (головний цикл)
            self._step_run_macro()

        except (VPNError, MacroError, SessionError) as e:
            self._set_state(SessionState.ERROR)
            self._emit(f"❌ Помилка сесії: {e}")
            self._stats.errors += 1

        except Exception as e:
            self._set_state(SessionState.ERROR)
            self._emit(f"❌ Неочікувана помилка: {e}")
            logger.exception(f"Session error: {e}")
            self._stats.errors += 1

        return self._finalize()

    def _finalize(self) -> GameSessionStats:
        """Завершує сесію та повертає статистику."""
        self._stats.finished_at = time.time()
        self._set_state(SessionState.STOPPED)

        self._emit(
            f"Сесія завершена: {self._stats.duration_minutes:.1f} хв, "
            f"помилок: {self._stats.errors}, "
            f"VPN reconnects: {self._stats.vpn_reconnects}"
        )
        return self._stats

    def stop(self) -> None:
        """Примусово зупиняє сесію."""
        self._emit("Зупинка сесії...")
        self._macro_player.stop()


# ============================================================================
# SESSION ORCHESTRATOR
# ============================================================================


class SessionOrchestrator:
    """
    Оркестратор фарму — верхній рівень управління.
    
    Об'єднує все:
    - LDPlayer інстанси
    - VPN
    - Акаунти
    - Макроси
    - Ігрові сесії
    
    Підтримує:
    - Безкінечний цикл фарму
    - Кілька інстансів паралельно
    - Автоматичні рестарти при помилках
    - Моніторинг та статистику
    
    Використання:
        orchestrator = SessionOrchestrator(config)
        orchestrator.setup_instance("FarmBot-1")  # Створити + налаштувати
        orchestrator.start_farming("FarmBot-1", account)  # Запустити фарм
        
        # Або повний потік:
        orchestrator.full_setup_and_farm()
    """

    def __init__(
        self,
        config: Optional[EmulatorConfig] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self._config = config or EmulatorConfig.load()
        self._status_callback = status_callback
        self._stop_event = threading.Event()

        # Менеджери
        self._ldplayer = LDPlayerManager(self._config.ldplayer)
        self._account_storage = AccountStorage()
        self._active_sessions: Dict[str, GameSession] = {}
        self._session_threads: Dict[str, threading.Thread] = {}

        logger.info("SessionOrchestrator ініціалізовано")

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def config(self) -> EmulatorConfig:
        return self._config

    @property
    def ldplayer(self) -> LDPlayerManager:
        return self._ldplayer

    @property
    def account_storage(self) -> AccountStorage:
        return self._account_storage

    # ========================================================================
    # SETUP
    # ========================================================================

    def setup_instance(
        self,
        name: str,
        install_apks: bool = True,
        patch_vpn: bool = True,
    ) -> EmulatorInstance:
        """
        Повне налаштування нового інстансу.
        
        1. Створює інстанс LDPlayer з мінімальними налаштуваннями
        2. Запускає його
        3. Встановлює APK (Chrome, VPN, Lucky Patcher)
        4. Модифікує VPN (вирізає рекламу)
        5. Зупиняє інстанс
        
        Args:
            name: Ім'я інстансу
            install_apks: Встановити APK
            patch_vpn: Модифікувати VPN
            
        Returns:
            Налаштований EmulatorInstance
        """
        self._emit(f"=== Налаштування інстансу: {name} ===")

        # 1. Створюємо інстанс
        instance = self._ldplayer.create_instance(name)

        if install_apks:
            # 2. Запускаємо
            self._ldplayer.launch(instance)

            # 3. Встановлюємо APK
            apk_mgr = APKManager(self._ldplayer, instance, self._config.apk)
            apk_mgr.prepare_instance(patch_vpn=patch_vpn)

            # 4. Зупиняємо
            self._ldplayer.shutdown(instance)

        self._emit(f"Інстанс '{name}' готовий до роботи")
        return instance

    def clone_and_setup(
        self,
        source_name: str,
        new_name: str,
    ) -> EmulatorInstance:
        """
        Клонує існуючий інстанс (з усіма APK).
        
        Швидший спосіб — не потрібно заново встановлювати APK.
        """
        source = self._ldplayer.get_instance(source_name)
        if not source:
            raise SessionError(f"Інстанс не знайдено: {source_name}")

        clone = self._ldplayer.clone_instance(source, new_name, randomize_device=True)
        self._emit(f"Інстанс клоновано: {source_name} → {new_name}")
        return clone

    def batch_setup(
        self,
        base_name: str,
        count: int,
        template_name: Optional[str] = None,
    ) -> List[EmulatorInstance]:
        """
        Масове створення інстансів.
        
        Якщо є template_name — клонує його (швидше).
        Інакше створює з нуля.
        """
        self._emit(f"Масове створення {count} інстансів: {base_name}-*")

        template = None
        if template_name:
            template = self._ldplayer.get_instance(template_name)

        return self._ldplayer.batch_create(base_name, count, template)

    # ========================================================================
    # ФАРМ
    # ========================================================================

    def start_farming(
        self,
        instance_name: str,
        account: AccountData,
        in_background: bool = True,
    ) -> Optional[GameSession]:
        """
        Запускає фарм на конкретному інстансі.
        
        Args:
            instance_name: Ім'я інстансу
            account: Акаунт для фарму
            in_background: Запустити в окремому потоці
            
        Returns:
            GameSession або None
        """
        instance = self._ldplayer.get_instance(instance_name)
        if not instance:
            raise SessionError(f"Інстанс не знайдено: {instance_name}")

        # Запускаємо інстанс якщо не працює
        if not instance.is_running:
            self._ldplayer.launch(instance)

        # Створюємо сесію
        session = GameSession(
            ldplayer=self._ldplayer,
            instance=instance,
            account=account,
            config=self._config,
            status_callback=self._status_callback,
            stop_check=self._stop_event.is_set,
        )

        self._active_sessions[instance_name] = session

        if in_background:
            thread = threading.Thread(
                target=self._run_session_loop,
                args=(session, account),
                name=f"farm-{instance_name}",
                daemon=True,
            )
            self._session_threads[instance_name] = thread
            thread.start()
            self._emit(f"Фарм запущено у фоні: {instance_name}")
        else:
            self._run_session_loop(session, account)

        return session

    def _run_session_loop(self, session: GameSession, account: AccountData) -> None:
        """Цикл фарму з автоматичними рестартами."""
        session_config = self._config.session
        session_num = 0

        while not self._stop_event.is_set():
            session_num += 1
            self._emit(f"--- Сесія #{session_num} ---")

            try:
                stats = session.start()

                # Оновлюємо статистику акаунту
                acc_mgr = EmulatorAccountManager(
                    self._ldplayer,
                    session._instance,
                    self._account_storage,
                )
                acc_mgr.update_session_stats(account, stats.duration_minutes)

            except Exception as e:
                self._emit(f"Помилка сесії #{session_num}: {e}")
                logger.exception(f"Session loop error: {e}")

            if self._stop_event.is_set():
                break

            if not session_config.loop_forever:
                if session_config.max_sessions > 0 and session_num >= session_config.max_sessions:
                    self._emit(f"Досягнуто ліміт сесій: {session_config.max_sessions}")
                    break

            # Пауза між сесіями
            cooldown = session_config.session_cooldown_seconds
            self._emit(f"Пауза {cooldown}с перед наступною сесією...")
            self._stop_event.wait(cooldown)

        self._emit("Фарм завершено")

    # ========================================================================
    # УПРАВЛІННЯ
    # ========================================================================

    def stop_all(self) -> None:
        """Зупиняє всі сесії фарму."""
        self._emit("Зупинка всіх сесій...")
        self._stop_event.set()

        # Зупиняємо всі активні сесії
        for name, session in self._active_sessions.items():
            try:
                session.stop()
            except Exception as e:
                logger.error(f"Error stopping session {name}: {e}")

        # Чекаємо завершення потоків
        for name, thread in self._session_threads.items():
            try:
                thread.join(timeout=30)
            except Exception:
                pass

        self._active_sessions.clear()
        self._session_threads.clear()
        self._emit("Всі сесії зупинено")

    def stop_instance(self, instance_name: str) -> None:
        """Зупиняє фарм на конкретному інстансі."""
        session = self._active_sessions.get(instance_name)
        if session:
            session.stop()
            self._active_sessions.pop(instance_name, None)

    def shutdown_everything(self) -> None:
        """Повна зупинка: сесії + емулятори."""
        self.stop_all()
        self._ldplayer.shutdown_all()
        self._emit("Все зупинено")

    # ========================================================================
    # ПОВНИЙ ПОТІК
    # ========================================================================

    def full_setup_and_farm(
        self,
        instance_name: str = "FarmBot-1",
        account: Optional[AccountData] = None,
    ) -> None:
        """
        Повний потік: створення інстансу → налаштування → фарм.
        
        Якщо акаунт не вказано — використовує перший з бази.
        
        Args:
            instance_name: Ім'я інстансу
            account: Акаунт (або None для автовибору)
        """
        self._emit("=== ПОВНИЙ ПОТІК ФАРМУ ===")

        # 1. Знаходимо або створюємо інстанс
        instance = self._ldplayer.get_instance(instance_name)
        if not instance:
            self._emit(f"Створення інстансу '{instance_name}'...")
            instance = self.setup_instance(instance_name)
        else:
            self._emit(f"Інстанс '{instance_name}' вже існує")

        # 2. Знаходимо акаунт
        if not account:
            ready = self._account_storage.get_ready_accounts()
            if ready:
                account = ready[0]
                self._emit(f"Використовуємо акаунт: {account.ms_email}")
            else:
                raise SessionError(
                    "Немає готових акаунтів. "
                    "Створіть акаунт через EmulatorAccountManager."
                )

        # 3. Запускаємо фарм
        self.start_farming(instance_name, account, in_background=False)

    # ========================================================================
    # СТАТУС
    # ========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Повертає статус всіх сесій."""
        instances = self._ldplayer.list_instances()
        return {
            'instances': {
                i.name: i.to_dict() for i in instances
            },
            'active_sessions': {
                name: session.state.value
                for name, session in self._active_sessions.items()
            },
            'total_instances': len(instances),
            'running_instances': len([i for i in instances if i.is_running]),
            'active_farms': len(self._active_sessions),
            'accounts': self._account_storage.get_account_count(),
        }

    def _emit(self, message: str) -> None:
        """Відправляє статус."""
        logger.info(f"[Orchestrator] {message}")
        if self._status_callback:
            try:
                self._status_callback(message)
            except Exception:
                pass
