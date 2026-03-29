"""
Управління акаунтами Microsoft та Epic Games для фарму у Fortnite.

Класи:
- AccountType: Тип акаунту (Microsoft, Epic, Xbox)
- AccountData: Повні дані акаунту (email, пароль, зв'язані акаунти)
- EmulatorAccountManager: Створення, зберігання, прив'язка акаунтів

Процес створення акаунту:
1. Увімкнути VPN (регіон US)
2. Через Chrome в емуляторі створити Microsoft акаунт
3. Зайти в пошту Microsoft
4. Створити Epic Games акаунт
5. Зайти на xbox.com/play → запустити Fortnite
6. Прив'язати Epic акаунт через epicgames.com/activate
"""

import os
import json
import time
from enum import Enum
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from ..core.logger import get_logger
from ..core.security import encrypt_password, decrypt_password
from .config import ACCOUNTS_DATA_DIR, APKConfig
from .ldplayer import LDPlayerManager, EmulatorInstance
from .exceptions import (
    AccountCreationError,
    MicrosoftAccountError,
    EpicAccountError,
    XboxLinkError,
)

logger = get_logger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class AccountType(str, Enum):
    """Тип акаунту."""
    MICROSOFT = "microsoft"
    EPIC = "epic"
    XBOX = "xbox"


class AccountStatus(str, Enum):
    """Статус акаунту."""
    CREATED = "created"            # Щойно створений
    VERIFIED = "verified"          # Email підтверджено
    LINKED = "linked"              # Прив'язано до Epic/Xbox
    ACTIVE = "active"              # Готовий до використання
    BANNED = "banned"              # Заблокований
    ERROR = "error"                # Помилка


# ============================================================================
# ACCOUNT DATA
# ============================================================================


@dataclass
class AccountData:
    """
    Повні дані акаунту для фарму.
    
    Об'єднує дані Microsoft, Epic та Xbox акаунтів.
    Паролі зберігаються зашифрованими.
    """
    # Microsoft акаунт
    ms_email: str = ""
    ms_password: str = ""
    ms_created_at: float = 0.0
    
    # Epic Games акаунт
    epic_email: str = ""
    epic_display_name: str = ""
    epic_password: str = ""
    epic_created_at: float = 0.0
    
    # Xbox / Fortnite
    xbox_gamertag: str = ""
    fortnite_linked: bool = False
    activation_code: str = ""
    
    # Метадані
    status: AccountStatus = AccountStatus.CREATED
    vpn_region: str = "United States"
    emulator_instance: str = ""    # Ім'я інстансу де створювався
    notes: str = ""
    
    # Статистика
    total_play_time_minutes: float = 0.0
    last_session_at: float = 0.0
    session_count: int = 0

    @property
    def login(self) -> str:
        """Основний логін (Microsoft email)."""
        return self.ms_email

    @property
    def is_ready(self) -> bool:
        """Чи готовий акаунт до використання."""
        return (
            bool(self.ms_email)
            and bool(self.ms_password)
            and self.fortnite_linked
            and self.status == AccountStatus.ACTIVE
        )

    def to_dict(self, encrypt: bool = True) -> Dict[str, Any]:
        """Серіалізує в словник. Паролі можуть бути зашифровані."""
        data = {
            'ms_email': self.ms_email,
            'ms_password': encrypt_password(self.ms_password) if encrypt else self.ms_password,
            'ms_created_at': self.ms_created_at,
            'epic_email': self.epic_email,
            'epic_display_name': self.epic_display_name,
            'epic_password': encrypt_password(self.epic_password) if encrypt else self.epic_password,
            'epic_created_at': self.epic_created_at,
            'xbox_gamertag': self.xbox_gamertag,
            'fortnite_linked': self.fortnite_linked,
            'activation_code': self.activation_code,
            'status': self.status.value,
            'vpn_region': self.vpn_region,
            'emulator_instance': self.emulator_instance,
            'notes': self.notes,
            'total_play_time_minutes': self.total_play_time_minutes,
            'last_session_at': self.last_session_at,
            'session_count': self.session_count,
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AccountData':
        """Десеріалізує зі словника. Паролі розшифровуються."""
        account = cls()
        account.ms_email = data.get('ms_email', '')
        account.ms_password = decrypt_password(data.get('ms_password', ''))
        account.ms_created_at = data.get('ms_created_at', 0.0)
        account.epic_email = data.get('epic_email', '')
        account.epic_display_name = data.get('epic_display_name', '')
        account.epic_password = decrypt_password(data.get('epic_password', ''))
        account.epic_created_at = data.get('epic_created_at', 0.0)
        account.xbox_gamertag = data.get('xbox_gamertag', '')
        account.fortnite_linked = data.get('fortnite_linked', False)
        account.activation_code = data.get('activation_code', '')
        account.vpn_region = data.get('vpn_region', 'United States')
        account.emulator_instance = data.get('emulator_instance', '')
        account.notes = data.get('notes', '')
        account.total_play_time_minutes = data.get('total_play_time_minutes', 0.0)
        account.last_session_at = data.get('last_session_at', 0.0)
        account.session_count = data.get('session_count', 0)

        status_str = data.get('status', 'created')
        try:
            account.status = AccountStatus(status_str)
        except ValueError:
            account.status = AccountStatus.CREATED

        return account


# ============================================================================
# ACCOUNT STORAGE
# ============================================================================


class AccountStorage:
    """
    Зберігання акаунтів у JSON файлах.
    
    Кожен акаунт зберігається як окремий JSON файл в директорії accounts.
    Паролі зашифровані через Fernet (core.security).
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self._storage_dir = storage_dir or ACCOUNTS_DATA_DIR
        os.makedirs(self._storage_dir, exist_ok=True)

    def _account_path(self, email: str) -> str:
        """Шлях до файлу акаунту."""
        safe_name = email.replace('@', '_at_').replace('.', '_')
        return os.path.join(self._storage_dir, f"{safe_name}.json")

    def save(self, account: AccountData) -> bool:
        """Зберігає акаунт у файл."""
        if not account.ms_email:
            return False
        try:
            path = self._account_path(account.ms_email)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(account.to_dict(encrypt=True), f, indent=2, ensure_ascii=False)
            logger.debug(f"Акаунт збережено: {account.ms_email}")
            return True
        except Exception as e:
            logger.error(f"Помилка збереження акаунту: {e}")
            return False

    def load(self, email: str) -> Optional[AccountData]:
        """Завантажує акаунт з файлу."""
        try:
            path = self._account_path(email)
            if not os.path.exists(path):
                return None
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return AccountData.from_dict(data)
        except Exception as e:
            logger.error(f"Помилка завантаження акаунту {email}: {e}")
            return None

    def load_all(self) -> List[AccountData]:
        """Завантажує всі акаунти."""
        accounts = []
        if not os.path.isdir(self._storage_dir):
            return accounts
        for filename in os.listdir(self._storage_dir):
            if filename.endswith('.json'):
                try:
                    path = os.path.join(self._storage_dir, filename)
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    accounts.append(AccountData.from_dict(data))
                except Exception as e:
                    logger.error(f"Помилка читання {filename}: {e}")
        return accounts

    def delete(self, email: str) -> bool:
        """Видаляє акаунт."""
        try:
            path = self._account_path(email)
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Акаунт видалено: {email}")
                return True
            return False
        except Exception as e:
            logger.error(f"Помилка видалення акаунту: {e}")
            return False

    def get_ready_accounts(self) -> List[AccountData]:
        """Повертає акаунти готові до використання."""
        return [acc for acc in self.load_all() if acc.is_ready]

    def get_account_count(self) -> int:
        """Кількість збережених акаунтів."""
        return len(self.load_all())


# ============================================================================
# EMULATOR ACCOUNT MANAGER
# ============================================================================


class EmulatorAccountManager:
    """
    Управління акаунтами в контексті емулятора.
    
    Автоматизує процес створення акаунтів:
    1. Microsoft акаунт (через Chrome в емуляторі + VPN)
    2. Epic Games акаунт
    3. Прив'язка до Xbox / Fortnite
    
    Використання:
        acc_mgr = EmulatorAccountManager(ldplayer, instance)
        account = acc_mgr.create_microsoft_account()
        acc_mgr.create_epic_account(account)
        acc_mgr.link_fortnite(account)
        acc_mgr.save_account(account)
    """

    # URLs для роботи з акаунтами
    MS_SIGNUP_URL = "https://signup.live.com/"
    EPIC_SIGNUP_URL = "https://www.epicgames.com/id/register"
    XBOX_PLAY_URL = "https://www.xbox.com/en-GB/play/games/fortnite/BT5P2X999VH2"
    EPIC_ACTIVATE_URL = "http://www.epicgames.com/activate"
    MS_OUTLOOK_URL = "https://outlook.live.com/"

    def __init__(
        self,
        ldplayer: LDPlayerManager,
        instance: EmulatorInstance,
        storage: Optional[AccountStorage] = None,
        chrome_package: str = "com.android.chrome",
    ):
        self._ldplayer = ldplayer
        self._instance = instance
        self._storage = storage or AccountStorage()
        self._chrome_package = chrome_package

        logger.info(f"EmulatorAccountManager для '{instance.name}'")

    @property
    def storage(self) -> AccountStorage:
        """Сховище акаунтів."""
        return self._storage

    # ========================================================================
    # ХЕЛПЕРИ
    # ========================================================================

    def _open_url_in_chrome(self, url: str) -> None:
        """Відкриває URL в Chrome через ADB."""
        self._ldplayer.adb_shell(
            self._instance,
            f'am start -a android.intent.action.VIEW -d "{url}" {self._chrome_package}',
        )
        time.sleep(3)

    def _press_back(self) -> None:
        """Натискає кнопку Back."""
        self._ldplayer.adb_key_event(self._instance, 4)  # KEYCODE_BACK

    def _press_home(self) -> None:
        """Натискає кнопку Home."""
        self._ldplayer.adb_key_event(self._instance, 3)  # KEYCODE_HOME

    # ========================================================================
    # СТВОРЕННЯ АКАУНТІВ
    # ========================================================================

    def create_microsoft_account(
        self,
        email_prefix: Optional[str] = None,
        password: Optional[str] = None,
    ) -> AccountData:
        """
        Відкриває сторінку створення Microsoft акаунту в Chrome.
        
        УВАГА: Цей метод відкриває сторінку реєстрації.
        Фактичне заповнення форми потребує ручної взаємодії
        або інтеграції з макро-системою.
        
        Args:
            email_prefix: Бажаний префікс email (опціонально)
            password: Бажаний пароль (опціонально)
            
        Returns:
            AccountData з попередньо заповненими даними
        """
        logger.info("Створення Microsoft акаунту...")

        account = AccountData(
            ms_created_at=time.time(),
            emulator_instance=self._instance.name,
        )

        # Відкриваємо сторінку реєстрації
        self._open_url_in_chrome(self.MS_SIGNUP_URL)

        logger.info(
            "Сторінку реєстрації Microsoft відкрито. "
            "Заповніть форму вручну або використовуйте макрос."
        )

        return account

    def register_microsoft_data(
        self,
        account: AccountData,
        email: str,
        password: str,
    ) -> AccountData:
        """
        Записує дані створеного Microsoft акаунту.
        
        Викликається після того, як акаунт створено
        (вручну або через макрос).
        
        Args:
            account: AccountData для оновлення
            email: Email акаунту Microsoft
            password: Пароль
            
        Returns:
            Оновлений AccountData
        """
        account.ms_email = email.strip().lower()
        account.ms_password = password
        account.status = AccountStatus.CREATED
        self._storage.save(account)
        logger.info(f"Microsoft акаунт зареєстровано: {email}")
        return account

    def create_epic_account(self, account: AccountData) -> AccountData:
        """
        Відкриває сторінку створення Epic Games акаунту.
        
        Args:
            account: AccountData з даними Microsoft
            
        Returns:
            Оновлений AccountData
        """
        if not account.ms_email:
            raise EpicAccountError("Спочатку створіть Microsoft акаунт")

        logger.info(f"Створення Epic акаунту для {account.ms_email}...")
        self._open_url_in_chrome(self.EPIC_SIGNUP_URL)

        logger.info(
            "Сторінку реєстрації Epic Games відкрито. "
            "Заповніть форму вручну або використовуйте макрос."
        )

        return account

    def register_epic_data(
        self,
        account: AccountData,
        display_name: str,
        epic_email: Optional[str] = None,
        epic_password: Optional[str] = None,
    ) -> AccountData:
        """
        Записує дані створеного Epic акаунту.
        
        Args:
            account: AccountData для оновлення
            display_name: Нікнейм у Epic
            epic_email: Email Epic (або той самий MS email)
            epic_password: Пароль Epic (або той самий MS пароль)
        """
        account.epic_email = epic_email or account.ms_email
        account.epic_display_name = display_name
        account.epic_password = epic_password or account.ms_password
        account.epic_created_at = time.time()
        account.status = AccountStatus.VERIFIED
        self._storage.save(account)
        logger.info(f"Epic акаунт зареєстровано: {display_name}")
        return account

    # ========================================================================
    # ПРИВ'ЯЗКА FORTNITE
    # ========================================================================

    def open_xbox_fortnite(self, account: AccountData) -> None:
        """
        Відкриває Xbox Cloud Gaming → Fortnite.
        
        Крок 1: Зайти на xbox.com/play → запустити Fortnite
        """
        logger.info("Відкриваємо Xbox Cloud Gaming → Fortnite...")
        self._open_url_in_chrome(self.XBOX_PLAY_URL)
        time.sleep(5)
        logger.info(
            "Xbox Cloud Gaming відкрито. "
            "Залогіньтесь в Microsoft акаунт та запустіть Fortnite."
        )

    def open_epic_activate(self, account: AccountData) -> None:
        """
        Відкриває сторінку прив'язки Epic акаунту.
        
        Крок 2: epicgames.com/activate → ввести код → прив'язати акаунт
        """
        logger.info("Відкриваємо Epic Games Activate...")
        self._open_url_in_chrome(self.EPIC_ACTIVATE_URL)
        time.sleep(3)
        logger.info(
            "Сторінку активації Epic Games відкрито. "
            "Введіть код з Fortnite для прив'язки акаунту."
        )

    def link_fortnite(
        self,
        account: AccountData,
        activation_code: str = "",
    ) -> AccountData:
        """
        Відмічає акаунт як прив'язаний до Fortnite.
        
        Args:
            account: AccountData
            activation_code: Код активації з epicgames.com/activate
        """
        account.fortnite_linked = True
        account.activation_code = activation_code
        account.status = AccountStatus.ACTIVE
        self._storage.save(account)
        logger.info(f"Fortnite прив'язано: {account.ms_email}")
        return account

    # ========================================================================
    # ПОВНИЙ ПОТІК СТВОРЕННЯ
    # ========================================================================

    def full_account_setup_flow(self) -> AccountData:
        """
        Повний потік створення акаунту (напів-автоматичний).
        
        Відкриває всі необхідні сторінки по черзі.
        Користувач заповнює форми вручну.
        
        Returns:
            AccountData (потрібно доповнити даними вручну)
        """
        logger.info("=== Повний потік створення акаунту ===")

        # Крок 1: Microsoft
        account = self.create_microsoft_account()
        logger.info("Крок 1/4: Створіть Microsoft акаунт та запишіть дані")

        # Крок 2: Epic Games
        # (виконується після ручного заповнення Microsoft)
        logger.info("Крок 2/4: Створіть Epic Games акаунт")

        # Крок 3: Xbox Fortnite
        logger.info("Крок 3/4: Зайдіть в Xbox Cloud → Fortnite")

        # Крок 4: Epic Activate
        logger.info("Крок 4/4: Прив'яжіть акаунт через epicgames.com/activate")

        return account

    # ========================================================================
    # МАСОВІ ОПЕРАЦІЇ
    # ========================================================================

    def get_all_accounts(self) -> List[AccountData]:
        """Повертає всі збережені акаунти."""
        return self._storage.load_all()

    def get_ready_accounts(self) -> List[AccountData]:
        """Повертає акаунти готові до фарму."""
        return self._storage.get_ready_accounts()

    def save_account(self, account: AccountData) -> bool:
        """Зберігає акаунт."""
        return self._storage.save(account)

    def update_session_stats(
        self,
        account: AccountData,
        play_time_minutes: float,
    ) -> None:
        """Оновлює статистику після сесії."""
        account.total_play_time_minutes += play_time_minutes
        account.last_session_at = time.time()
        account.session_count += 1
        self._storage.save(account)

    def get_account_summary(self) -> Dict[str, Any]:
        """Повертає зведення по всіх акаунтах."""
        accounts = self._storage.load_all()
        return {
            'total': len(accounts),
            'ready': len([a for a in accounts if a.is_ready]),
            'active': len([a for a in accounts if a.status == AccountStatus.ACTIVE]),
            'banned': len([a for a in accounts if a.status == AccountStatus.BANNED]),
            'total_play_time_hours': sum(a.total_play_time_minutes for a in accounts) / 60,
        }
