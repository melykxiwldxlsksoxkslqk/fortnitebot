# AI Development Rules — EpicBot v4.0

## Code Quality

1. **No spaghetti code.** Every function, class, and module must be clean, readable, and maintainable.
2. **No hardcoding.** All values that may change (URLs, timings, island codes, ADB coordinates, etc.) must come from dataclass configs or `config/settings.json`. Never inline magic numbers or strings.
3. **DRY (Don't Repeat Yourself).** If logic appears more than twice — extract it into a reusable function or method.
4. **Single Responsibility.** Each function does one thing. Each module handles one domain area.
5. **Meaningful names.** Variables, functions, and classes must have descriptive names. No `x`, `tmp`, `data2`.
6. **No dead code.** Remove unused imports, commented-out blocks, and unreachable code.

## Verification

1. **Always verify developer claims.** Before making changes, read the actual code — do not trust assumptions about what it does.
2. **Always verify AI output.** After writing code, re-read it to confirm correctness. Check for:
   - Logic errors
   - Missing edge cases
   - Broken imports or circular dependencies
   - Type mismatches
3. **Test after changes.** Run `pytest` from the project root. Tests are in `tests/`. If relevant tests don't exist — flag it.

## Logging

1. **Every session must be logged** in `.ai/TIMELOG.md` — what was done, when, and why.
2. **Every scope change must be reflected** in `.ai/SCOPE.md` — mark completed items, add new ones.

## Architecture

The project uses **emulator-based automation** (LDPlayer + ADB macros).
**There is NO browser-based automation** — no Playwright, no Camoufox, no canvas, no vision.

```
config/                  # Конфігурація (settings.json, accounts.txt, island_code.txt, emulator/)
src/
├── core/                # Базові компоненти: config.py, db.py (SQLite), logger.py, security.py, exceptions.py
├── emulator/            # Автоматизація LDPlayer (OOP):
│   ├── config.py        #   Dataclass конфігурації (LDPlayer, VPN, Macros, Session, APK)
│   ├── ldplayer.py      #   LDPlayerManager — управління інстансами через ldconsole + ADB
│   ├── vpn.py           #   VPNManager — JumpJumpVPN (увімкнення/регіон/таймер)
│   ├── apk.py           #   APKManager — встановлення та патчинг APK
│   ├── accounts.py      #   EmulatorAccountManager — акаунти MS/Epic (SQLite + Fernet)
│   ├── macros.py        #   MacroPlayer, MacroComposer, MacroFactory — запис/плей макросів
│   ├── session.py       #   GameSession, SessionOrchestrator — повний цикл фарму
│   └── exceptions.py    #   Виключення модуля
├── main.py              # Точка входу (SessionOrchestrator → нескінченний фарм)
└── run.py               # CLI launcher (argparse)
tests/                   # pytest тести
```

### Принципи слоїв

1. **`core/`** — утиліти, конфіг, БД, логування. Не імпортує з `emulator/`.
2. **`emulator/`** — вся бізнес-логіка: LDPlayer, VPN, макроси, сесії. Використовує `core/`.
3. **`main.py`** — оркестрація: створення інстансів → реєстрація акаунтів → запуск фарму.
4. **`run.py`** — CLI парсинг (argparse) → делегує в `main.py`.

### Важливі конвенції

- LDPlayer управляється через `ldconsole.exe` CLI + ADB команди.
- VPN (JumpJumpVPN) керується через ADB taps та launch/stop.
- Ігрові дії — виключно через ADB макроси (tap, swipe, key_event, text_input).
- Конфігурація завантажується з `config/emulator/emulator_settings.json` через `EmulatorConfig.load()`.
- Акаунти зберігаються в SQLite з шифруванням паролів (Fernet AES).
- Всі таймаути — через dataclass `TIMEOUTS` в `core/config.py`.
- Емулятор: планшетний режим, мінімальна роздільність, 2 ядра, 2 ГБ RAM, 10 FPS.
- Тести використовують `pytest` з моками (unittest.mock). Не залежать від реального LDPlayer.
