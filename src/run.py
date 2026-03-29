"""
CLI точка входу для EpicBot.

Використання:
    python -m src.run                          # Запуск фарму (всі акаунти)
    python -m src.run --island 1234-5678-9012  # Конкретний острів
    python -m src.run --max-instances 3        # Обмежити інстанси
    python -m src.run --ldplayer "D:\\LD9"     # Шлях до LDPlayer
    python -m src.run --version                # Версія
"""

import sys
import argparse


def main() -> int:
    """CLI точка входу."""
    parser = argparse.ArgumentParser(
        description="EpicBot — Fortnite XP Farm через LDPlayer",
    )
    parser.add_argument(
        "--version", action="store_true",
        help="Показати версію",
    )
    parser.add_argument(
        "--island", type=str, default=None,
        help="Код острова Fortnite (XXXX-XXXX-XXXX)",
    )
    parser.add_argument(
        "--max-instances", type=int, default=None,
        help="Максимальна кількість інстансів LDPlayer",
    )
    parser.add_argument(
        "--ldplayer", type=str, default=None,
        help="Шлях до директорії LDPlayer",
    )

    args = parser.parse_args()

    if args.version:
        from . import __version__
        print(f"EpicBot v{__version__}")
        return 0

    from .main import main as run_main
    return run_main(
        island_code=args.island,
        max_instances=args.max_instances,
        ldplayer_dir=args.ldplayer,
    )


if __name__ == "__main__":
    sys.exit(main())
