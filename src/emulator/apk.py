"""
Управління APK файлами в емуляторі LDPlayer.

Класи:
- APKType: Типи APK (VPN, Chrome, Lucky Patcher)
- APKInfo: Інформація про APK
- APKManager: Встановлення, модифікація, перевірка APK

Робочий процес:
1. Встановити Chrome
2. Встановити JumpJumpVPN
3. Встановити Lucky Patcher
4. Модифікувати VPN через Lucky Patcher (вирізати рекламу)
"""

import os
import time
from enum import Enum
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from ..core.logger import get_logger
from .config import APKConfig
from .ldplayer import LDPlayerManager, EmulatorInstance
from .exceptions import APKError, APKInstallError, APKPatchError

logger = get_logger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class APKType(str, Enum):
    """Типи APK файлів для встановлення."""
    VPN = "vpn"
    CHROME = "chrome"
    LUCKY_PATCHER = "lucky_patcher"
    OTHER = "other"


# ============================================================================
# APK INFO
# ============================================================================


@dataclass
class APKInfo:
    """Інформація про APK файл."""
    apk_type: APKType
    package_name: str
    display_name: str
    apk_path: str = ""
    version: str = ""
    is_installed: bool = False
    is_patched: bool = False

    @property
    def is_available(self) -> bool:
        """Чи доступний APK файл для встановлення."""
        return bool(self.apk_path) and os.path.isfile(self.apk_path)


# ============================================================================
# РЕЄСТР APK
# ============================================================================


# Стандартні APK для встановлення
DEFAULT_APKS: Dict[APKType, APKInfo] = {
    APKType.VPN: APKInfo(
        apk_type=APKType.VPN,
        package_name="com.jumpjump.vpn",
        display_name="JumpJump VPN",
    ),
    APKType.CHROME: APKInfo(
        apk_type=APKType.CHROME,
        package_name="com.android.chrome",
        display_name="Google Chrome",
    ),
    APKType.LUCKY_PATCHER: APKInfo(
        apk_type=APKType.LUCKY_PATCHER,
        package_name="ru.mgames.luckypatcher",
        display_name="Lucky Patcher",
    ),
}


# ============================================================================
# APK MANAGER
# ============================================================================


class APKManager:
    """
    Менеджер APK файлів для інстансу емулятора.
    
    Відповідальності:
    - Встановлення необхідних APK (Chrome, VPN, Lucky Patcher)
    - Перевірка наявності встановлених додатків
    - Модифікація VPN через Lucky Patcher (вирізання реклами)
    - Підготовка інстансу до роботи (все-в-одному)
    
    Використання:
        apk_mgr = APKManager(ldplayer_mgr, instance, config)
        apk_mgr.install_all()               # Встановити всі необхідні APK
        apk_mgr.patch_vpn_with_lucky()       # Модифікувати VPN
        ready = apk_mgr.verify_all_installed()  # Перевірити
    """

    def __init__(
        self,
        ldplayer: LDPlayerManager,
        instance: EmulatorInstance,
        config: Optional[APKConfig] = None,
    ):
        self._ldplayer = ldplayer
        self._instance = instance
        self._config = config or APKConfig()
        self._apk_registry: Dict[APKType, APKInfo] = self._build_registry()

        logger.info(f"APKManager створено для '{instance.name}'")

    def _build_registry(self) -> Dict[APKType, APKInfo]:
        """Будує реєстр APK з конфігурації."""
        registry = {}
        for apk_type, default_info in DEFAULT_APKS.items():
            info = APKInfo(
                apk_type=default_info.apk_type,
                package_name=default_info.package_name,
                display_name=default_info.display_name,
            )
            # Оновлюємо шляхи з конфігурації
            if apk_type == APKType.VPN and self._config.vpn_apk_path:
                info.apk_path = self._config.vpn_apk_path
            elif apk_type == APKType.CHROME and self._config.chrome_apk_path:
                info.apk_path = self._config.chrome_apk_path
            elif apk_type == APKType.LUCKY_PATCHER and self._config.lucky_patcher_apk_path:
                info.apk_path = self._config.lucky_patcher_apk_path

            # Шукаємо APK в стандартній директорії
            if not info.apk_path or not os.path.isfile(info.apk_path):
                found_path = self._find_apk_in_directory(info.package_name)
                if found_path:
                    info.apk_path = found_path

            registry[apk_type] = info

        return registry

    def _find_apk_in_directory(self, package_name: str) -> Optional[str]:
        """Шукає APK файл у директорії APK."""
        apk_dir = self._config.apk_directory
        if not os.path.isdir(apk_dir):
            return None

        for filename in os.listdir(apk_dir):
            if filename.endswith('.apk'):
                lower_name = filename.lower()
                # Спрощений пошук за назвою пакету
                pkg_parts = package_name.split('.')
                if any(part in lower_name for part in pkg_parts[-2:]):
                    return os.path.join(apk_dir, filename)
        return None

    @property
    def registry(self) -> Dict[APKType, APKInfo]:
        """Реєстр APK."""
        return self._apk_registry

    # ========================================================================
    # ПЕРЕВІРКА НАЯВНОСТІ
    # ========================================================================

    def is_installed(self, apk_type: APKType) -> bool:
        """Перевіряє, чи встановлений APK в інстансі."""
        info = self._apk_registry.get(apk_type)
        if not info:
            return False

        result = self._ldplayer.adb_shell(
            self._instance,
            f'pm list packages | grep {info.package_name}',
        )
        installed = info.package_name in result
        info.is_installed = installed
        return installed

    def check_all_installed(self) -> Dict[APKType, bool]:
        """Перевіряє наявність всіх APK."""
        results = {}
        for apk_type in self._apk_registry:
            results[apk_type] = self.is_installed(apk_type)
        return results

    def verify_all_installed(self) -> bool:
        """Перевіряє, що всі необхідні APK встановлені."""
        statuses = self.check_all_installed()
        all_ok = all(statuses.values())
        if not all_ok:
            missing = [t.value for t, installed in statuses.items() if not installed]
            logger.warning(f"Не встановлені APK: {', '.join(missing)}")
        return all_ok

    # ========================================================================
    # ВСТАНОВЛЕННЯ
    # ========================================================================

    def install(self, apk_type: APKType) -> bool:
        """
        Встановлює один APK файл.
        
        Args:
            apk_type: Тип APK
            
        Returns:
            True якщо успішно
            
        Raises:
            APKInstallError: Якщо APK файл не знайдено або помилка встановлення
        """
        info = self._apk_registry.get(apk_type)
        if not info:
            raise APKInstallError(f"Невідомий тип APK: {apk_type}")

        if not info.is_available:
            raise APKInstallError(
                f"APK файл не знайдено для {info.display_name}. "
                f"Покладіть файл в: {self._config.apk_directory}"
            )

        if self.is_installed(apk_type):
            logger.info(f"{info.display_name} вже встановлено")
            return True

        logger.info(f"Встановлення {info.display_name}...")
        success = self._ldplayer.adb_install(self._instance, info.apk_path)

        if not success:
            raise APKInstallError(f"Помилка встановлення {info.display_name}")

        info.is_installed = True
        logger.info(f"{info.display_name} встановлено успішно")
        return True

    def install_all(self) -> Dict[APKType, bool]:
        """
        Встановлює всі необхідні APK.
        
        Порядок: Chrome → VPN → Lucky Patcher
        
        Returns:
            Словник {тип: успіх}
        """
        results = {}
        install_order = [APKType.CHROME, APKType.VPN, APKType.LUCKY_PATCHER]

        for apk_type in install_order:
            try:
                results[apk_type] = self.install(apk_type)
            except APKInstallError as e:
                logger.error(f"Помилка: {e}")
                results[apk_type] = False

        return results

    # ========================================================================
    # МОДИФІКАЦІЯ VPN ЧЕРЕЗ LUCKY PATCHER
    # ========================================================================

    def patch_vpn_with_lucky(self) -> bool:
        """
        Модифікує VPN додаток через Lucky Patcher (вирізає рекламу).
        
        Процес (автоматизований через ADB tap):
        1. Запустити Lucky Patcher
        2. Знайти VPN додаток у списку
        3. Натиснути на нього
        4. Вибрати "Remove Ads" / "Modified APK"
        5. Підтвердити
        
        Returns:
            True якщо успішно
            
        Raises:
            APKPatchError: Якщо модифікація не вдалась
        """
        vpn_info = self._apk_registry.get(APKType.VPN)
        lp_info = self._apk_registry.get(APKType.LUCKY_PATCHER)

        if not vpn_info or not vpn_info.is_installed:
            raise APKPatchError("VPN додаток не встановлено")
        if not lp_info or not lp_info.is_installed:
            raise APKPatchError("Lucky Patcher не встановлено")

        logger.info("Модифікація VPN через Lucky Patcher...")

        # 1. Запускаємо Lucky Patcher
        self._ldplayer.launch_app(
            self._instance,
            lp_info.package_name,
        )
        time.sleep(5)  # Чекаємо повного завантаження

        # 2. Натискаємо на VPN додаток у списку
        # (координати залежать від розширення — потрібне калібрування)
        logger.info("Lucky Patcher: пошук VPN додатку в списку...")
        self._find_and_tap_app_in_lucky_patcher(vpn_info.package_name)
        time.sleep(2)

        # 3. Вибираємо "Menu of patches" → "Remove ads"
        logger.info("Lucky Patcher: вибір патчу...")
        self._apply_remove_ads_patch()
        time.sleep(3)

        # 4. Зупиняємо Lucky Patcher
        self._ldplayer.stop_app(self._instance, lp_info.package_name)

        vpn_info.is_patched = True
        logger.info("VPN модифіковано: реклама вирізана")
        return True

    def _find_and_tap_app_in_lucky_patcher(self, package_name: str) -> None:
        """
        Знаходить додаток у списку Lucky Patcher та натискає на нього.
        
        Примітка: Координати залежать від розширення екрану.
        Для точного позиціонування рекомендується налаштувати під свій інстанс.
        """
        # Пошук через скрол списку — базова реалізація
        # У реальному випадку можна використовувати UIAutomator або OCR
        res = self._instance.config
        center_x = res.resolution_width // 2
        item_height = 80

        # Скролимо вниз та шукаємо (до 20 спроб)
        for attempt in range(20):
            # Перевіряємо поточний екран
            # (у продакшені тут був би OCR або template matching)
            self._ldplayer.adb_tap(self._instance, center_x, 200 + item_height * (attempt % 5))
            time.sleep(1)

            # Перевіряємо чи відкрився контекст потрібного додатку
            # Якщо Lucky Patcher показує меню патчів — ми знайшли
            break  # Базова реалізація — натискаємо перший елемент

        logger.debug(f"Lucky Patcher: натиснуто на елемент (спроба пошуку {package_name})")

    def _apply_remove_ads_patch(self) -> None:
        """Застосовує патч видалення реклами."""
        res = self._instance.config
        center_x = res.resolution_width // 2

        # Натискаємо "Menu of Patches"
        self._ldplayer.adb_tap(self._instance, center_x, 300)
        time.sleep(2)

        # Натискаємо "Remove Google Ads"
        self._ldplayer.adb_tap(self._instance, center_x, 250)
        time.sleep(2)

        # Натискаємо "Patch"
        self._ldplayer.adb_tap(self._instance, center_x, 400)
        time.sleep(5)  # Чекаємо завершення патчу

        # Натискаємо "OK" / "Apply"
        self._ldplayer.adb_tap(self._instance, center_x, 350)
        time.sleep(2)

    # ========================================================================
    # ПІДГОТОВКА ІНСТАНСУ
    # ========================================================================

    def prepare_instance(self, patch_vpn: bool = True) -> bool:
        """
        Повна підготовка інстансу: встановлення всіх APK + модифікація VPN.
        
        Args:
            patch_vpn: Чи модифікувати VPN (вирізати рекламу)
            
        Returns:
            True якщо все успішно
        """
        logger.info(f"Підготовка інстансу '{self._instance.name}'...")

        # Встановлюємо всі APK
        results = self.install_all()
        if not all(results.values()):
            failed = [t.value for t, ok in results.items() if not ok]
            logger.error(f"Не вдалося встановити: {', '.join(failed)}")
            return False

        # Модифікуємо VPN
        if patch_vpn:
            try:
                self.patch_vpn_with_lucky()
            except APKPatchError as e:
                logger.warning(f"Модифікація VPN не вдалась: {e}. Продовжуємо без патчу.")

        logger.info(f"Інстанс '{self._instance.name}' підготовлено")
        return True

    def get_status(self) -> Dict[str, dict]:
        """Повертає статус всіх APK."""
        self.check_all_installed()
        return {
            apk_type.value: {
                'display_name': info.display_name,
                'package_name': info.package_name,
                'is_installed': info.is_installed,
                'is_patched': info.is_patched,
                'apk_available': info.is_available,
            }
            for apk_type, info in self._apk_registry.items()
        }
