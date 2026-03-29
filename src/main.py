"""
Головна точка входу EpicBot.

Оркеструє повний цикл:
    1. Ініціалізація БД та завантаження конфігурації
    2. Створення / клонування інстансів LDPlayer
    3. Встановлення APK (VPN, Chrome, Lucky Patcher)
    4. Запуск фарм-сесій (VPN → Chrome → Xbox Cloud Gaming → Fortnite → макрос)
    5. Нескінченний цикл з перепідключенням VPN кожні 45 хв
"""

import sys
import signal
import threading
from typing import Optional, Any

from .core.config import load_settings, load_accounts, load_island_code, EMULATOR
from .core.logger import get_logger, setup_logging
from .core.db import init_db

from .emulator import (
    SessionOrchestrator,
    EmulatorConfig,
    AccountData,
    AccountType,
)

logger = get_logger(__name__)

# Graceful shutdown
_shutdown_event = threading.Event()


def _handle_signal(sig: int, frame: Any) -> None:
    """Обробник сигналу для graceful shutdown."""
    logger.info(f"Отримано сигнал {sig}, зупиняю ботів...")
    _shutdown_event.set()


def _status_log(message: str) -> None:
    """Колбек статусу — логує все."""
    logger.info(f"[STATUS] {message}")


def main(
    island_code: Optional[str] = None,
    max_instances: Optional[int] = None,
    ldplayer_dir: Optional[str] = None,
) -> int:
    """
    Головна функція — запуск фарму через LDPlayer.

    Args:
        island_code: Код острова Fortnite (або з конфігу)
        max_instances: Максимум інстансів (або з конфігу)
        ldplayer_dir: Шлях до LDPlayer (або з конфігу)

    Returns:
        0 — успішно, 1 — помилка
    """
    setup_logging()
    logger.info("=" * 60)
    logger.info("EpicBot v4.0 — LDPlayer Emulator Mode")
    logger.info("=" * 60)

    init_db()

    # Завантаження конфігурації
    settings = load_settings()

    code = island_code or settings.get("island_code") or load_island_code()
    if not code or code == "1234-5678-9012":
        logger.error(
            "Код острова не встановлено. "
            "Вкажіть у config/island_code.txt або config/settings.json"
        )
        return 1

    instances_count = max_instances or settings.get("max_instances", 5)
    ld_dir = ldplayer_dir or EMULATOR.ldplayer_dir

    logger.info(f"Код острова: {code}")
    logger.info(f"Макс. інстансів: {instances_count}")
    logger.info(f"LDPlayer: {ld_dir}")

    # Завантаження акаунтів
    accounts = load_accounts()
    if not accounts:
        logger.error("Немає акаунтів. Додайте в config/accounts.txt (email:password)")
        return 1

    logger.info(f"Акаунтів завантажено: {len(accounts)}")

    # Реєстрація сигналів
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    # Конфігурація емулятора (з файлу або за замовчуванням + CLI overrides)
    emu_config = EmulatorConfig.load()
    emu_config.ldplayer.install_dir = ld_dir
    emu_config.session.island_code = code
    emu_config.session.loop_forever = True

    # Створення оркестратора
    orchestrator = SessionOrchestrator(
        config=emu_config,
        status_callback=_status_log,
    )

    try:
        actual_count = min(len(accounts), instances_count)
        logger.info(f"Запуск {actual_count} інстансів...")

        # Реєструємо акаунти
        for acct in accounts[:actual_count]:
            account_data = AccountData(
                ms_email=acct["email"],
                ms_password=acct["password"],
                account_type=AccountType.MICROSOFT,
            )
            orchestrator.account_storage.add_account(account_data)

        # Запускаємо по одному інстансу на акаунт
        ready_accounts = orchestrator.account_storage.get_ready_accounts()

        for i in range(actual_count):
            if _shutdown_event.is_set():
                break

            instance_name = f"FarmBot-{i + 1}"
            account = ready_accounts[i] if i < len(ready_accounts) else ready_accounts[0]

            try:
                orchestrator.start_farming(
                    instance_name=instance_name,
                    account=account,
                    in_background=True,
                )
            except Exception as e:
                logger.error(f"Помилка запуску {instance_name}: {e}")

        # Чекаємо на shutdown
        logger.info("Фарм запущено. Натисніть Ctrl+C для зупинки.")
        _shutdown_event.wait()

        orchestrator.shutdown_everything()
        logger.info("Фарм завершено")
        return 0

    except KeyboardInterrupt:
        logger.info("Перервано користувачем")
        _shutdown_event.set()
        orchestrator.shutdown_everything()
        return 0
    except Exception as e:
        logger.error(f"Критична помилка: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
