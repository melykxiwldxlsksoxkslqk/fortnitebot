"""
Главная точка входа для EpicBot.

Использование:
    python -m src.run           # Запуск IPC сервера (для Electron)
    python -m src.run --cli     # CLI режим
"""

import sys
import argparse


def main():
    """Точка входа."""
    parser = argparse.ArgumentParser(description="EpicBot - Fortnite Cloud Gaming Bot")
    parser.add_argument("--cli", action="store_true", help="Запуск в CLI режиме")
    parser.add_argument("--version", action="store_true", help="Показать версию")
    args = parser.parse_args()
    
    if args.version:
        from . import __version__
        print(f"EpicBot v{__version__}")
        return 0
    
    if args.cli:
        # CLI режим - запуск бота напрямую
        from .bot import run_bot
        from .core import init_db, fetch_accounts, fetch_proxies, get_settings
        
        init_db()
        accounts = fetch_accounts()
        proxies = fetch_proxies()
        settings = get_settings()
        
        if not accounts:
            print("Ошибка: Нет аккаунтов в базе данных")
            return 1
        
        # Запуск первого аккаунта
        account = accounts[0]
        proxy = proxies[0] if proxies else None
        island_code = settings.get('island_code', '')
        headless = bool(settings.get('headless', False))
        
        print(f"Запуск бота для {account.get('login')}")
        run_bot(account, island_code, headless=headless, proxy=proxy)
        return 0
    
    # По умолчанию - IPC сервер
    from .ipc import main as ipc_main
    ipc_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
