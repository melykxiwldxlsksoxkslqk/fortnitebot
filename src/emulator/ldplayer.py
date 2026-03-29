"""
Управління інстансами LDPlayer.

Класи:
- EmulatorInstance: Один інстанс емулятора (стан, конфігурація, ADB)
- LDPlayerManager: Менеджер інстансів (створення, клонування, налаштування, запуск)

Взаємодія з LDPlayer відбувається через ldconsole.exe (CLI).
"""

import os
import re
import time
import random
import subprocess
from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from ..core.logger import get_logger
from .config import (
    LDPlayerConfig,
    InstanceConfig,
    DEVICE_MODELS,
    TABLET_RESOLUTIONS,
)
from .exceptions import (
    LDPlayerError,
    LDPlayerNotFoundError,
    InstanceNotFoundError,
    InstanceAlreadyRunningError,
)

logger = get_logger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class InstanceStatus(str, Enum):
    """Статус інстансу емулятора."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    UNKNOWN = "unknown"


# ============================================================================
# EMULATOR INSTANCE
# ============================================================================


@dataclass
class EmulatorInstance:
    """
    Представляє один інстанс LDPlayer.
    
    Зберігає стан, конфігурацію та надає методи для взаємодії через ADB.
    """
    index: int
    name: str
    status: InstanceStatus = InstanceStatus.STOPPED
    config: InstanceConfig = field(default_factory=InstanceConfig)
    adb_port: int = 0
    pid: int = 0
    
    # Прив'язаний акаунт
    account_email: str = ""
    
    # Метадані
    created_at: float = field(default_factory=time.time)
    last_launched: float = 0.0

    @property
    def is_running(self) -> bool:
        """Чи працює інстанс."""
        return self.status == InstanceStatus.RUNNING

    @property
    def adb_address(self) -> str:
        """ADB адреса інстансу (для adb connect)."""
        if self.adb_port:
            return f"127.0.0.1:{self.adb_port}"
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """Серіалізація в словник."""
        return {
            'index': self.index,
            'name': self.name,
            'status': self.status.value,
            'adb_port': self.adb_port,
            'account_email': self.account_email,
            'config': {
                'device_model': self.config.device_model,
                'manufacturer': self.config.manufacturer,
                'cpu_cores': self.config.cpu_cores,
                'ram_mb': self.config.ram_mb,
                'fps': self.config.fps,
                'resolution': f"{self.config.resolution_width}x{self.config.resolution_height}",
            }
        }


# ============================================================================
# ІМЕІ ГЕНЕРАТОР
# ============================================================================


class IMEIGenerator:
    """Генератор унікальних IMEI номерів (Luhn checksum)."""

    @staticmethod
    def generate() -> str:
        """Генерує валідний IMEI номер."""
        # TAC (Type Allocation Code) — 8 цифр
        tac_prefixes = [
            "35332407", "35391107", "35490506",
            "35467206", "35260506", "35838206",
            "86462003", "35566707", "35972307",
        ]
        tac = random.choice(tac_prefixes)
        # Serial number — 6 цифр
        serial = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        partial = tac + serial
        # Luhn checksum
        check_digit = IMEIGenerator._luhn_checksum(partial)
        return partial + str(check_digit)

    @staticmethod
    def _luhn_checksum(partial: str) -> int:
        """Обчислює контрольну цифру Luhn."""
        digits = [int(d) for d in partial]
        odd_sum = sum(digits[0::2])
        even_doubled = [d * 2 for d in digits[1::2]]
        even_sum = sum(d - 9 if d > 9 else d for d in even_doubled)
        total = odd_sum + even_sum
        return (10 - (total % 10)) % 10


# ============================================================================
# LDPLAYER MANAGER
# ============================================================================


class LDPlayerManager:
    """
    Менеджер інстансів LDPlayer.
    
    Забезпечує:
    - Створення нових інстансів з мінімальними налаштуваннями
    - Клонування існуючих інстансів
    - Налаштування кожного інстансу (CPU, RAM, FPS, модель, IMEI)
    - Запуск / зупинку / перезапуск інстансів
    - Список всіх інстансів та їх статусів
    - Встановлення APK через ADB
    - Виконання ADB команд
    
    Використання:
        mgr = LDPlayerManager(config)
        instance = mgr.create_instance("FarmBot-1")
        mgr.configure_instance(instance)
        mgr.launch(instance)
        mgr.adb_install(instance, "path/to/vpn.apk")
    """

    def __init__(self, config: Optional[LDPlayerConfig] = None):
        self._config = config or LDPlayerConfig()
        self._instances: Dict[int, EmulatorInstance] = {}
        self._validate_installation()
        logger.info(f"LDPlayerManager ініціалізований: {self._config.install_dir}")

    # ========================================================================
    # ВАЛІДАЦІЯ
    # ========================================================================

    def _validate_installation(self) -> None:
        """Перевіряє наявність LDPlayer."""
        if not os.path.isfile(self._config.ldconsole_path):
            raise LDPlayerNotFoundError(
                f"ldconsole.exe не знайдено: {self._config.ldconsole_path}. "
                f"Перевірте шлях до LDPlayer: {self._config.install_dir}"
            )

    @property
    def config(self) -> LDPlayerConfig:
        """Повертає конфігурацію менеджера."""
        return self._config

    # ========================================================================
    # ВИКОНАННЯ КОМАНД
    # ========================================================================

    def _run_ldconsole(
        self,
        *args: str,
        timeout: Optional[int] = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        """
        Виконує команду ldconsole.exe.
        
        Args:
            *args: Аргументи команди
            timeout: Таймаут виконання
            check: Чи перевіряти код повернення
            
        Returns:
            CompletedProcess з результатом
            
        Raises:
            LDPlayerError: Якщо команда завершилась з помилкою
        """
        cmd = [self._config.ldconsole_path] + list(args)
        timeout = timeout or self._config.adb_timeout

        try:
            logger.debug(f"ldconsole: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding='utf-8',
                errors='replace',
            )
            if check and result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                raise LDPlayerError(f"ldconsole помилка ({result.returncode}): {error_msg}")
            return result
        except subprocess.TimeoutExpired:
            raise LDPlayerError(f"Таймаут ldconsole ({timeout}с): {' '.join(args)}")
        except FileNotFoundError:
            raise LDPlayerNotFoundError(f"ldconsole.exe не знайдено: {self._config.ldconsole_path}")

    def _run_adb(
        self,
        instance: EmulatorInstance,
        *adb_args: str,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """
        Виконує ADB команду для конкретного інстансу.
        
        Args:
            instance: Інстанс емулятора
            *adb_args: Аргументи ADB команди
            timeout: Таймаут
        """
        args = ['adb', '--index', str(instance.index)] + list(adb_args)
        return self._run_ldconsole(*args, timeout=timeout, check=False)

    # ========================================================================
    # СПИСОК ІНСТАНСІВ
    # ========================================================================

    def list_instances(self) -> List[EmulatorInstance]:
        """
        Повертає список всіх інстансів LDPlayer.
        
        Returns:
            Список EmulatorInstance
        """
        result = self._run_ldconsole('list2', check=False)
        instances = []

        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        index = int(parts[0])
                        name = parts[1]
                        # Парсимо статус
                        status_map = {
                            '1': InstanceStatus.RUNNING,
                            '0': InstanceStatus.STOPPED,
                        }
                        status = status_map.get(parts[2].strip(), InstanceStatus.UNKNOWN)

                        instance = EmulatorInstance(
                            index=index,
                            name=name,
                            status=status,
                        )

                        # Оновлюємо кеш
                        self._instances[index] = instance
                        instances.append(instance)
                    except (ValueError, IndexError):
                        continue

        logger.debug(f"Знайдено {len(instances)} інстансів")
        return instances

    def get_instance(self, name_or_index) -> Optional[EmulatorInstance]:
        """
        Знаходить інстанс за іменем або індексом.
        
        Args:
            name_or_index: Ім'я (str) або індекс (int) інстансу
            
        Returns:
            EmulatorInstance або None
        """
        instances = self.list_instances()
        for inst in instances:
            if isinstance(name_or_index, int) and inst.index == name_or_index:
                return inst
            if isinstance(name_or_index, str) and inst.name == name_or_index:
                return inst
        return None

    def get_running_instances(self) -> List[EmulatorInstance]:
        """Повертає тільки запущені інстанси."""
        return [i for i in self.list_instances() if i.is_running]

    # ========================================================================
    # СТВОРЕННЯ ТА КЛОНУВАННЯ
    # ========================================================================

    def create_instance(
        self,
        name: str,
        config: Optional[InstanceConfig] = None,
    ) -> EmulatorInstance:
        """
        Створює новий інстанс LDPlayer.
        
        Args:
            name: Ім'я інстансу
            config: Конфігурація (або використовує default)
            
        Returns:
            Створений EmulatorInstance
        """
        config = config or InstanceConfig(name=name)

        logger.info(f"Створення інстансу: {name}")
        self._run_ldconsole('add', '--name', name)

        # Знаходимо створений інстанс
        instance = self.get_instance(name)
        if not instance:
            raise LDPlayerError(f"Не вдалося знайти створений інстанс: {name}")

        instance.config = config
        self.configure_instance(instance)

        logger.info(f"Інстанс створено: {name} (index={instance.index})")
        return instance

    def clone_instance(
        self,
        source: EmulatorInstance,
        new_name: str,
        randomize_device: bool = True,
    ) -> EmulatorInstance:
        """
        Клонує існуючий інстанс (з усіма APK та налаштуваннями).
        
        Args:
            source: Інстанс-джерело
            new_name: Ім'я нового інстансу
            randomize_device: Чи генерувати нову модель/IMEI
            
        Returns:
            Клонований EmulatorInstance
        """
        logger.info(f"Клонування '{source.name}' → '{new_name}'")
        self._run_ldconsole('copy', '--name', new_name, '--from', str(source.index))

        clone = self.get_instance(new_name)
        if not clone:
            raise LDPlayerError(f"Не вдалося знайти клонований інстанс: {new_name}")

        # Рандомізація пристрою для уникнення бану
        if randomize_device:
            new_config = InstanceConfig(name=new_name)
            self._randomize_device_config(new_config)
            clone.config = new_config
            self.configure_instance(clone)

        logger.info(f"Інстанс клоновано: {new_name} (index={clone.index})")
        return clone

    def _randomize_device_config(self, config: InstanceConfig) -> None:
        """Рандомізує модель пристрою, IMEI та Android ID."""
        device = random.choice(DEVICE_MODELS)
        config.device_model = device['model']
        config.manufacturer = device['manufacturer']
        config.imei = IMEIGenerator.generate()
        config.android_id = ''.join(
            random.choices('0123456789abcdef', k=16)
        )

        resolution = random.choice(TABLET_RESOLUTIONS)
        config.resolution_width = resolution['width']
        config.resolution_height = resolution['height']
        config.dpi = resolution['dpi']

    # ========================================================================
    # НАЛАШТУВАННЯ
    # ========================================================================

    def configure_instance(self, instance: EmulatorInstance) -> None:
        """
        Застосовує конфігурацію до інстансу.
        
        Встановлює CPU, RAM, FPS, розширення, модель, IMEI тощо.
        Інстанс повинен бути зупинений.
        
        Args:
            instance: Інстанс для налаштування
        """
        if instance.is_running:
            raise InstanceAlreadyRunningError(
                f"Інстанс '{instance.name}' має бути зупинений для налаштування"
            )

        cfg = instance.config
        logger.info(f"Налаштування інстансу '{instance.name}': "
                     f"{cfg.cpu_cores}CPU/{cfg.ram_mb}MB/{cfg.fps}FPS "
                     f"{cfg.resolution_width}x{cfg.resolution_height} "
                     f"model={cfg.device_model}")

        # Формуємо аргументи
        modify_args = ['modify', '--index', str(instance.index)]
        for key, value in cfg.to_ldplayer_args().items():
            modify_args.extend([key, value])

        self._run_ldconsole(*modify_args)

    def set_instance_property(
        self,
        instance: EmulatorInstance,
        prop: str,
        value: str,
    ) -> None:
        """Встановлює окрему властивість інстансу."""
        self._run_ldconsole(
            'modify', '--index', str(instance.index),
            f'--{prop}', value,
        )

    # ========================================================================
    # ЗАПУСК / ЗУПИНКА
    # ========================================================================

    def launch(self, instance: EmulatorInstance, wait: bool = True) -> None:
        """
        Запускає інстанс емулятора.
        
        Args:
            instance: Інстанс для запуску
            wait: Чекати завершення завантаження
        """
        if instance.is_running:
            logger.warning(f"Інстанс '{instance.name}' вже запущений")
            return

        logger.info(f"Запуск інстансу: {instance.name}")
        instance.status = InstanceStatus.STARTING

        self._run_ldconsole(
            'launch', '--index', str(instance.index),
            timeout=self._config.launch_timeout,
        )

        if wait:
            self._wait_for_boot(instance)

        instance.status = InstanceStatus.RUNNING
        instance.last_launched = time.time()
        logger.info(f"Інстанс запущено: {instance.name}")

    def _wait_for_boot(self, instance: EmulatorInstance, timeout: int = 120) -> None:
        """Очікує завершення завантаження Android."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self._run_adb(
                instance,
                'shell', 'getprop', 'sys.boot_completed',
            )
            if result.stdout.strip() == '1':
                logger.debug(f"Android завантажено: {instance.name}")
                return
            time.sleep(3)
        raise LDPlayerError(f"Таймаут завантаження інстансу: {instance.name}")

    def shutdown(self, instance: EmulatorInstance) -> None:
        """Зупиняє інстанс емулятора."""
        logger.info(f"Зупинка інстансу: {instance.name}")
        instance.status = InstanceStatus.STOPPING
        self._run_ldconsole(
            'quit', '--index', str(instance.index),
            timeout=self._config.shutdown_timeout,
        )
        instance.status = InstanceStatus.STOPPED
        logger.info(f"Інстанс зупинено: {instance.name}")

    def reboot(self, instance: EmulatorInstance) -> None:
        """Перезапускає інстанс."""
        logger.info(f"Перезапуск інстансу: {instance.name}")
        self._run_ldconsole(
            'reboot', '--index', str(instance.index),
            timeout=self._config.launch_timeout,
        )
        self._wait_for_boot(instance)
        instance.status = InstanceStatus.RUNNING

    def shutdown_all(self) -> None:
        """Зупиняє всі запущені інстанси."""
        logger.info("Зупинка всіх інстансів")
        self._run_ldconsole('quitall', check=False)

    # ========================================================================
    # ADB ОПЕРАЦІЇ
    # ========================================================================

    def adb_install(self, instance: EmulatorInstance, apk_path: str) -> bool:
        """
        Встановлює APK в інстанс через ADB.
        
        Args:
            instance: Цільовий інстанс
            apk_path: Шлях до APK файлу
            
        Returns:
            True якщо успішно
        """
        if not os.path.isfile(apk_path):
            raise LDPlayerError(f"APK файл не знайдено: {apk_path}")

        logger.info(f"Встановлення APK в '{instance.name}': {os.path.basename(apk_path)}")
        result = self._run_ldconsole(
            'installapp', '--index', str(instance.index),
            '--filename', apk_path,
            timeout=120,
            check=False,
        )
        success = result.returncode == 0
        if success:
            logger.info(f"APK встановлено: {os.path.basename(apk_path)}")
        else:
            logger.error(f"Помилка встановлення APK: {result.stderr}")
        return success

    def adb_uninstall(self, instance: EmulatorInstance, package: str) -> bool:
        """Видаляє додаток з інстансу."""
        result = self._run_ldconsole(
            'uninstallapp', '--index', str(instance.index),
            '--packagename', package,
            check=False,
        )
        return result.returncode == 0

    def adb_shell(
        self,
        instance: EmulatorInstance,
        command: str,
    ) -> str:
        """
        Виконує shell команду в інстансі через ADB.
        
        Args:
            instance: Цільовий інстанс
            command: Shell команда
            
        Returns:
            Вивід команди
        """
        result = self._run_adb(instance, 'shell', command)
        return result.stdout.strip()

    def adb_tap(self, instance: EmulatorInstance, x: int, y: int) -> None:
        """Натискає на координати в інстансі."""
        self.adb_shell(instance, f'input tap {x} {y}')

    def adb_swipe(
        self,
        instance: EmulatorInstance,
        x1: int, y1: int,
        x2: int, y2: int,
        duration_ms: int = 300,
    ) -> None:
        """Свайп від (x1,y1) до (x2,y2)."""
        self.adb_shell(instance, f'input swipe {x1} {y1} {x2} {y2} {duration_ms}')

    def adb_key_event(self, instance: EmulatorInstance, keycode: int) -> None:
        """Надсилає key event."""
        self.adb_shell(instance, f'input keyevent {keycode}')

    def adb_text(self, instance: EmulatorInstance, text: str) -> None:
        """Вводить текст."""
        # Екранування спеціальних символів для shell
        escaped = text.replace(' ', '%s').replace('&', '\\&').replace(';', '\\;')
        self.adb_shell(instance, f'input text "{escaped}"')

    def launch_app(self, instance: EmulatorInstance, package: str, activity: str = "") -> None:
        """
        Запускає додаток в інстансі.
        
        Args:
            instance: Цільовий інстанс
            package: Ім'я пакету
            activity: Ім'я Activity (опціонально)
        """
        if activity:
            self.adb_shell(instance, f'am start -n {package}/{activity}')
        else:
            self.adb_shell(instance, f'monkey -p {package} -c android.intent.category.LAUNCHER 1')

    def stop_app(self, instance: EmulatorInstance, package: str) -> None:
        """Зупиняє додаток."""
        self.adb_shell(instance, f'am force-stop {package}')

    def is_app_running(self, instance: EmulatorInstance, package: str) -> bool:
        """Перевіряє, чи працює додаток."""
        output = self.adb_shell(instance, f'pidof {package}')
        return bool(output.strip())

    def take_screenshot(self, instance: EmulatorInstance, local_path: str) -> bool:
        """
        Робить скріншот інстансу.
        
        Args:
            instance: Цільовий інстанс
            local_path: Шлях для збереження локально
        """
        remote_path = '/sdcard/screenshot.png'
        self.adb_shell(instance, f'screencap -p {remote_path}')
        result = self._run_ldconsole(
            'pull', '--index', str(instance.index),
            '--remote', remote_path,
            '--local', local_path,
            check=False,
        )
        return result.returncode == 0

    # ========================================================================
    # УТИЛІТИ
    # ========================================================================

    def get_instance_count(self) -> int:
        """Повертає кількість інстансів."""
        return len(self.list_instances())

    def remove_instance(self, instance: EmulatorInstance) -> None:
        """Видаляє інстанс."""
        if instance.is_running:
            self.shutdown(instance)
        logger.info(f"Видалення інстансу: {instance.name}")
        self._run_ldconsole('remove', '--index', str(instance.index))
        self._instances.pop(instance.index, None)

    def rename_instance(self, instance: EmulatorInstance, new_name: str) -> None:
        """Перейменовує інстанс."""
        self._run_ldconsole(
            'rename', '--index', str(instance.index),
            '--title', new_name,
        )
        instance.name = new_name

    def batch_create(
        self,
        base_name: str,
        count: int,
        template: Optional[EmulatorInstance] = None,
    ) -> List[EmulatorInstance]:
        """
        Масове створення інстансів.
        
        Якщо є template — клонує його, інакше створює нові.
        Кожен інстанс отримує унікальну модель/IMEI.
        
        Args:
            base_name: Базове ім'я (додається -1, -2, ...)
            count: Кількість інстансів
            template: Шаблон для клонування
            
        Returns:
            Список створених інстансів
        """
        if count > self._config.max_instances:
            raise LDPlayerError(
                f"Перевищено максимум інстансів: {count} > {self._config.max_instances}"
            )

        instances = []
        for i in range(1, count + 1):
            name = f"{base_name}-{i}"
            try:
                if template:
                    inst = self.clone_instance(template, name, randomize_device=True)
                else:
                    config = InstanceConfig(name=name)
                    self._randomize_device_config(config)
                    inst = self.create_instance(name, config)
                instances.append(inst)
                logger.info(f"Створено {i}/{count}: {name}")
            except LDPlayerError as e:
                logger.error(f"Помилка створення {name}: {e}")

        return instances
