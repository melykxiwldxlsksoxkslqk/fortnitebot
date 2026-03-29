"""
Тести для модуля emulator.

Покриває:
- config: Конфігурація, серіалізація/десеріалізація
- ldplayer: EmulatorInstance, IMEIGenerator, LDPlayerManager (mock)
- vpn: VPNSession, VPNManager
- apk: APKInfo, APKManager
- accounts: AccountData, AccountStorage
- macros: MacroStep, MacroSequence, MacroComposer, MacroFactory
- session: GameSession, SessionOrchestrator
- exceptions: Ієрархія виключень
"""

import os
import sys
import json
import time
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import asdict

# Додаємо шлях до src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.emulator.config import (
    EmulatorConfig,
    InstanceConfig,
    LDPlayerConfig,
    VPNConfig,
    MacroConfig,
    SessionConfig,
    APKConfig,
    DEFAULT_EMULATOR_SETTINGS,
    DEVICE_MODELS,
    TABLET_RESOLUTIONS,
)
from src.emulator.ldplayer import (
    EmulatorInstance,
    InstanceStatus,
    IMEIGenerator,
)
from src.emulator.vpn import (
    VPNSession,
    VPNStatus,
    VPNRegion,
)
from src.emulator.apk import (
    APKInfo,
    APKType,
    DEFAULT_APKS,
)
from src.emulator.accounts import (
    AccountData,
    AccountType,
    AccountStatus,
    AccountStorage,
)
from src.emulator.macros import (
    MacroStep,
    MacroAction,
    MacroSequence,
    MacroComposer,
    MacroFactory,
)
from src.emulator.session import (
    SessionState,
    GameSessionStats,
)
from src.emulator.exceptions import (
    EmulatorError,
    LDPlayerError,
    LDPlayerNotFoundError,
    InstanceNotFoundError,
    VPNError,
    VPNConnectionError,
    VPNTimeoutError,
    MacroError,
    MacroPlaybackError,
    MacroNotFoundError,
    APKError,
    APKInstallError,
    APKPatchError,
    AccountCreationError,
    MicrosoftAccountError,
    EpicAccountError,
    SessionError,
    SessionTimeoutError,
    SessionInterruptedError,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_dir():
    """Створює тимчасову директорію для тестів."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def instance_config():
    """Конфігурація інстансу."""
    return InstanceConfig(
        name="TestBot",
        cpu_cores=2,
        ram_mb=2048,
        fps=10,
        resolution_width=960,
        resolution_height=540,
        device_model="SM-T510",
        manufacturer="samsung",
    )


@pytest.fixture
def emulator_instance(instance_config):
    """Інстанс емулятора."""
    return EmulatorInstance(
        index=0,
        name="TestBot",
        status=InstanceStatus.STOPPED,
        config=instance_config,
    )


@pytest.fixture
def account_data():
    """Тестові дані акаунту."""
    return AccountData(
        ms_email="test@outlook.com",
        ms_password="TestPass123",
        ms_created_at=time.time(),
        epic_email="test@outlook.com",
        epic_display_name="TestPlayer",
        epic_password="TestPass123",
        epic_created_at=time.time(),
        fortnite_linked=True,
        status=AccountStatus.ACTIVE,
        vpn_region="United States",
        emulator_instance="TestBot",
    )


# ============================================================================
# TEST: CONFIG
# ============================================================================


class TestInstanceConfig:
    """Тести конфігурації інстансу."""

    def test_default_values(self):
        config = InstanceConfig()
        assert config.cpu_cores == 2
        assert config.ram_mb == 2048
        assert config.fps == 10
        assert config.resolution_width == 960
        assert config.resolution_height == 540

    def test_to_ldplayer_args(self, instance_config):
        args = instance_config.to_ldplayer_args()
        assert args['--cpu'] == '2'
        assert args['--memory'] == '2048'
        assert args['--fps'] == '10'
        assert args['--width'] == '960'
        assert args['--height'] == '540'
        assert args['--model'] == 'SM-T510'

    def test_imei_not_in_args_when_empty(self):
        config = InstanceConfig(imei="")
        args = config.to_ldplayer_args()
        assert '--imei' not in args

    def test_imei_in_args_when_set(self):
        config = InstanceConfig(imei="123456789012345")
        args = config.to_ldplayer_args()
        assert args['--imei'] == '123456789012345'


class TestEmulatorConfig:
    """Тести головної конфігурації."""

    def test_default_creation(self):
        config = EmulatorConfig()
        assert config.ldplayer is not None
        assert config.vpn is not None
        assert config.macros is not None
        assert config.session is not None

    def test_to_dict_and_back(self):
        config = EmulatorConfig()
        config.vpn.default_region = "Canada"
        config.session.island_code = "1111-2222-3333"

        data = config.to_dict()
        restored = EmulatorConfig.from_dict(data)

        assert restored.vpn.default_region == "Canada"
        assert restored.session.island_code == "1111-2222-3333"

    def test_save_and_load(self, temp_dir):
        path = os.path.join(temp_dir, "test_config.json")
        config = EmulatorConfig()
        config.macros.gameplay_repeat_count = 99

        assert config.save(path) is True
        assert os.path.exists(path)

        loaded = EmulatorConfig.load(path)
        assert loaded.macros.gameplay_repeat_count == 99

    def test_load_nonexistent_returns_default(self, temp_dir):
        path = os.path.join(temp_dir, "nonexistent.json")
        config = EmulatorConfig.load(path)
        assert config is not None
        assert config.macros.gameplay_repeat_count == 45  # default

    def test_default_settings_dict(self):
        assert 'ldplayer_path' in DEFAULT_EMULATOR_SETTINGS
        assert 'vpn_region' in DEFAULT_EMULATOR_SETTINGS
        assert 'island_code' in DEFAULT_EMULATOR_SETTINGS


class TestVPNConfig:
    """Тести конфігурації VPN."""

    def test_default_region(self):
        config = VPNConfig()
        assert config.default_region == "United States"

    def test_session_duration(self):
        config = VPNConfig()
        assert config.session_duration_minutes == 150  # 2.5 години


class TestMacroConfig:
    """Тести конфігурації макросів."""

    def test_gameplay_defaults(self):
        config = MacroConfig()
        assert config.gameplay_repeat_count == 45
        assert config.gameplay_duration_seconds == 60
        assert config.xbox_session_minutes == 60
        assert config.games_per_vpn_session == 2

    def test_randomization_defaults(self):
        config = MacroConfig()
        assert config.randomize_timing is True
        assert config.timing_variance_ms == 200
        assert config.randomize_position is True
        assert config.position_variance_px == 5


class TestDeviceModels:
    """Тести списків моделей та розширень."""

    def test_device_models_not_empty(self):
        assert len(DEVICE_MODELS) > 0

    def test_device_model_fields(self):
        for model in DEVICE_MODELS:
            assert 'model' in model
            assert 'manufacturer' in model
            assert 'name' in model

    def test_tablet_resolutions_not_empty(self):
        assert len(TABLET_RESOLUTIONS) > 0

    def test_tablet_resolution_fields(self):
        for res in TABLET_RESOLUTIONS:
            assert 'width' in res
            assert 'height' in res
            assert 'dpi' in res
            assert res['width'] > 0
            assert res['height'] > 0


# ============================================================================
# TEST: LDPLAYER
# ============================================================================


class TestEmulatorInstance:
    """Тести інстансу емулятора."""

    def test_creation(self, emulator_instance):
        assert emulator_instance.index == 0
        assert emulator_instance.name == "TestBot"
        assert emulator_instance.status == InstanceStatus.STOPPED

    def test_is_running_false_when_stopped(self, emulator_instance):
        assert emulator_instance.is_running is False

    def test_is_running_true(self, emulator_instance):
        emulator_instance.status = InstanceStatus.RUNNING
        assert emulator_instance.is_running is True

    def test_adb_address_empty(self, emulator_instance):
        assert emulator_instance.adb_address == ""

    def test_adb_address_with_port(self, emulator_instance):
        emulator_instance.adb_port = 5555
        assert emulator_instance.adb_address == "127.0.0.1:5555"

    def test_to_dict(self, emulator_instance):
        d = emulator_instance.to_dict()
        assert d['index'] == 0
        assert d['name'] == "TestBot"
        assert d['status'] == "stopped"
        assert 'config' in d
        assert d['config']['device_model'] == "SM-T510"


class TestIMEIGenerator:
    """Тести генератора IMEI."""

    def test_generate_length(self):
        imei = IMEIGenerator.generate()
        assert len(imei) == 15

    def test_generate_digits_only(self):
        imei = IMEIGenerator.generate()
        assert imei.isdigit()

    def test_generate_unique(self):
        """Два згенерованих IMEI мають бути різними."""
        imei1 = IMEIGenerator.generate()
        imei2 = IMEIGenerator.generate()
        # Теоретично можуть співпасти, але ймовірність мізерна
        # Генеруємо 100 і перевіряємо унікальність
        imeis = {IMEIGenerator.generate() for _ in range(100)}
        assert len(imeis) > 90  # Допускаємо <10% колізій

    def test_luhn_checksum_valid(self):
        """Перевіряє що згенерований IMEI проходить Luhn."""
        for _ in range(10):
            imei = IMEIGenerator.generate()
            # Luhn validation
            digits = [int(d) for d in imei]
            total = 0
            for i, d in enumerate(digits):
                if i % 2 == 1:
                    d *= 2
                    if d > 9:
                        d -= 9
                total += d
            assert total % 10 == 0, f"IMEI {imei} не проходить Luhn"


class TestInstanceStatus:
    """Тести статусів."""

    def test_all_statuses(self):
        assert InstanceStatus.STOPPED == "stopped"
        assert InstanceStatus.RUNNING == "running"
        assert InstanceStatus.STARTING == "starting"
        assert InstanceStatus.ERROR == "error"


# ============================================================================
# TEST: VPN
# ============================================================================


class TestVPNSession:
    """Тести сесії VPN."""

    def test_initial_state(self):
        session = VPNSession()
        assert session.status == VPNStatus.DISCONNECTED
        assert session.elapsed_minutes == 0.0
        assert session.is_active is False

    def test_active_session(self):
        session = VPNSession(
            started_at=time.time() - 60,  # 1 хвилина тому
            status=VPNStatus.CONNECTED,
            region="United States",
        )
        assert session.is_active is True
        assert 0.9 < session.elapsed_minutes < 1.5
        assert session.remaining_minutes < 150

    def test_remaining_minutes(self):
        session = VPNSession(
            started_at=time.time() - 3600,  # 1 година тому
            status=VPNStatus.CONNECTED,
        )
        assert 89 < session.remaining_minutes < 91


class TestVPNRegion:
    """Тести регіонів VPN."""

    def test_us_region(self):
        assert VPNRegion.UNITED_STATES == "United States"

    def test_all_regions(self):
        regions = list(VPNRegion)
        assert len(regions) >= 5


# ============================================================================
# TEST: APK
# ============================================================================


class TestAPKInfo:
    """Тести інформації про APK."""

    def test_creation(self):
        info = APKInfo(
            apk_type=APKType.VPN,
            package_name="com.jumpjump.vpn",
            display_name="JumpJump VPN",
        )
        assert info.apk_type == APKType.VPN
        assert info.is_available is False

    def test_is_available_with_path(self, temp_dir):
        apk_path = os.path.join(temp_dir, "test.apk")
        with open(apk_path, 'w') as f:
            f.write("fake apk")

        info = APKInfo(
            apk_type=APKType.VPN,
            package_name="test",
            display_name="Test",
            apk_path=apk_path,
        )
        assert info.is_available is True

    def test_default_apks_registry(self):
        assert APKType.VPN in DEFAULT_APKS
        assert APKType.CHROME in DEFAULT_APKS
        assert APKType.LUCKY_PATCHER in DEFAULT_APKS


# ============================================================================
# TEST: ACCOUNTS
# ============================================================================


class TestAccountData:
    """Тести даних акаунту."""

    def test_creation(self, account_data):
        assert account_data.ms_email == "test@outlook.com"
        assert account_data.fortnite_linked is True

    def test_is_ready(self, account_data):
        assert account_data.is_ready is True

    def test_not_ready_without_password(self):
        account = AccountData(ms_email="test@test.com")
        assert account.is_ready is False

    def test_not_ready_without_fortnite(self):
        account = AccountData(
            ms_email="test@test.com",
            ms_password="pass",
            status=AccountStatus.ACTIVE,
        )
        assert account.is_ready is False

    def test_login_property(self, account_data):
        assert account_data.login == "test@outlook.com"

    def test_to_dict(self, account_data):
        d = account_data.to_dict(encrypt=False)
        assert d['ms_email'] == "test@outlook.com"
        assert d['fortnite_linked'] is True
        assert d['status'] == "active"

    def test_from_dict_roundtrip(self, account_data):
        d = account_data.to_dict(encrypt=False)
        restored = AccountData.from_dict(d)
        assert restored.ms_email == account_data.ms_email
        assert restored.epic_display_name == account_data.epic_display_name
        assert restored.fortnite_linked == account_data.fortnite_linked
        assert restored.status == account_data.status


class TestAccountStorage:
    """Тести зберігання акаунтів."""

    def test_save_and_load(self, temp_dir, account_data):
        storage = AccountStorage(temp_dir)
        assert storage.save(account_data) is True

        loaded = storage.load("test@outlook.com")
        assert loaded is not None
        assert loaded.ms_email == "test@outlook.com"
        assert loaded.epic_display_name == "TestPlayer"

    def test_load_nonexistent(self, temp_dir):
        storage = AccountStorage(temp_dir)
        assert storage.load("nonexistent@test.com") is None

    def test_load_all(self, temp_dir):
        storage = AccountStorage(temp_dir)
        acc1 = AccountData(ms_email="acc1@test.com", ms_password="pass1")
        acc2 = AccountData(ms_email="acc2@test.com", ms_password="pass2")
        storage.save(acc1)
        storage.save(acc2)

        all_accounts = storage.load_all()
        assert len(all_accounts) == 2
        emails = {a.ms_email for a in all_accounts}
        assert "acc1@test.com" in emails
        assert "acc2@test.com" in emails

    def test_delete(self, temp_dir, account_data):
        storage = AccountStorage(temp_dir)
        storage.save(account_data)
        assert storage.delete("test@outlook.com") is True
        assert storage.load("test@outlook.com") is None

    def test_delete_nonexistent(self, temp_dir):
        storage = AccountStorage(temp_dir)
        assert storage.delete("nonexistent@test.com") is False

    def test_get_ready_accounts(self, temp_dir, account_data):
        storage = AccountStorage(temp_dir)

        # Готовий акаунт
        storage.save(account_data)

        # Не готовий акаунт
        not_ready = AccountData(ms_email="notready@test.com")
        storage.save(not_ready)

        ready = storage.get_ready_accounts()
        assert len(ready) == 1
        assert ready[0].ms_email == "test@outlook.com"

    def test_get_account_count(self, temp_dir, account_data):
        storage = AccountStorage(temp_dir)
        assert storage.get_account_count() == 0
        storage.save(account_data)
        assert storage.get_account_count() == 1


class TestAccountStatus:
    """Тести статусів акаунту."""

    def test_all_statuses(self):
        assert AccountStatus.CREATED == "created"
        assert AccountStatus.ACTIVE == "active"
        assert AccountStatus.BANNED == "banned"


# ============================================================================
# TEST: MACROS
# ============================================================================


class TestMacroStep:
    """Тести кроку макросу."""

    def test_tap_creation(self):
        step = MacroStep(
            action=MacroAction.TAP,
            x=100, y=200,
            delay_ms=500,
            description="Натиснути кнопку",
        )
        assert step.action == MacroAction.TAP
        assert step.x == 100
        assert step.y == 200

    def test_to_dict(self):
        step = MacroStep(action=MacroAction.TAP, x=100, y=200, delay_ms=500)
        d = step.to_dict()
        assert d['action'] == 'tap'
        assert d['x'] == 100
        assert d['delay_ms'] == 500

    def test_from_dict_roundtrip(self):
        original = MacroStep(
            action=MacroAction.TEXT_INPUT,
            text="Hello World",
            delay_ms=300,
        )
        d = original.to_dict()
        restored = MacroStep.from_dict(d)
        assert restored.action == MacroAction.TEXT_INPUT
        assert restored.text == "Hello World"
        assert restored.delay_ms == 300

    def test_minimal_dict(self):
        """Мінімальний словник не містить нульових значень."""
        step = MacroStep(action=MacroAction.WAIT, delay_ms=5000)
        d = step.to_dict()
        assert 'x' not in d
        assert 'text' not in d
        assert 'key_code' not in d


class TestMacroSequence:
    """Тести послідовності макросу."""

    def test_creation(self):
        seq = MacroSequence(name="test")
        assert seq.name == "test"
        assert len(seq.steps) == 0

    def test_fluent_api(self):
        seq = MacroSequence(name="test") \
            .add_tap(100, 200) \
            .add_wait(1000) \
            .add_text("hello") \
            .add_key(66)
        assert len(seq.steps) == 4
        assert seq.steps[0].action == MacroAction.TAP
        assert seq.steps[1].action == MacroAction.WAIT
        assert seq.steps[2].action == MacroAction.TEXT_INPUT
        assert seq.steps[3].action == MacroAction.KEY_EVENT

    def test_total_steps_with_repeats(self):
        seq = MacroSequence(name="test", repeat_count=5)
        seq.add_tap(100, 200)
        seq.add_wait(1000)
        assert seq.total_steps == 10  # 2 * 5

    def test_estimated_duration(self):
        seq = MacroSequence(name="test", repeat_count=2)
        seq.add_wait(1000)
        seq.add_tap(100, 200, delay_ms=500)
        # Total delay per pass: 1000 + 500 = 1500 ms
        # Duration per pass: 0 + 0 = 0 ms (duration_ms on steps)
        # With 2 repeats: 3000 ms = 3.0 s
        assert seq.estimated_duration_seconds == 3.0

    def test_save_and_load(self, temp_dir):
        seq = MacroSequence(name="save_test")
        seq.add_tap(100, 200, description="test tap")
        seq.add_wait(500)

        path = seq.save(temp_dir)
        assert os.path.exists(path)

        loaded = MacroSequence.load("save_test", temp_dir)
        assert loaded.name == "save_test"
        assert len(loaded.steps) == 2
        assert loaded.steps[0].description == "test tap"

    def test_load_nonexistent(self, temp_dir):
        with pytest.raises(MacroNotFoundError):
            MacroSequence.load("nonexistent", temp_dir)

    def test_to_dict_and_back(self):
        seq = MacroSequence(
            name="roundtrip",
            repeat_count=10,
            randomize_timing=True,
            timing_variance_ms=100,
        )
        seq.add_tap(50, 60)
        seq.add_text("code")

        d = seq.to_dict()
        restored = MacroSequence.from_dict(d)
        assert restored.name == "roundtrip"
        assert restored.repeat_count == 10
        assert restored.randomize_timing is True
        assert len(restored.steps) == 2

    def test_add_swipe(self):
        seq = MacroSequence(name="test")
        seq.add_swipe(0, 0, 100, 100, duration_ms=500)
        assert seq.steps[0].action == MacroAction.SWIPE
        assert seq.steps[0].x2 == 100
        assert seq.steps[0].y2 == 100


class TestMacroComposer:
    """Тести композитора макросів."""

    def test_creation(self):
        composer = MacroComposer("main")
        assert len(composer.sequences) == 0

    def test_add_sequences(self):
        seq1 = MacroSequence(name="seq1")
        seq2 = MacroSequence(name="seq2")

        composer = MacroComposer("main")
        composer.add(seq1).add(seq2)
        assert len(composer.sequences) == 2

    def test_add_multiple(self):
        seq = MacroSequence(name="repeat_me")
        composer = MacroComposer("main")
        composer.add_multiple(seq, 3)
        assert len(composer.sequences) == 3

    def test_loop_forever(self):
        composer = MacroComposer("main")
        composer.set_loop_forever(True)
        assert composer.loop_forever is True

    def test_loop_count(self):
        composer = MacroComposer("main")
        composer.set_loop_count(5)
        assert composer.loop_count == 5
        assert composer.loop_forever is False

    def test_build(self):
        seq = MacroSequence(name="test")
        seq.add_wait(1000)

        composer = MacroComposer("main", "Test composition")
        composer.add(seq)
        composer.set_loop_forever(True)

        build = composer.build()
        assert build['name'] == "main"
        assert build['loop_forever'] is True
        assert build['total_sequences'] == 1

    def test_save_and_load(self, temp_dir):
        seq = MacroSequence(name="inner")
        seq.add_tap(50, 60)

        composer = MacroComposer("outer")
        composer.add(seq)
        composer.set_loop_forever(True)

        path = composer.save(temp_dir)
        assert os.path.exists(path)

        loaded = MacroComposer.load("outer", temp_dir)
        assert loaded.loop_forever is True
        assert len(loaded.sequences) == 1
        assert loaded.sequences[0].name == "inner"


class TestMacroFactory:
    """Тести фабрики стандартних макросів."""

    def test_create_launch_fortnite(self):
        factory = MacroFactory()
        macro = factory.create_launch_fortnite()
        assert macro.name == "launch_fortnite"
        assert len(macro.steps) > 0

    def test_create_enter_island_code(self):
        factory = MacroFactory(island_code="1234-5678-9012")
        macro = factory.create_enter_island_code()
        assert macro.name == "enter_island_code"
        # Має містити крок з введенням коду
        text_steps = [s for s in macro.steps if s.action == MacroAction.TEXT_INPUT]
        assert len(text_steps) > 0
        assert text_steps[0].text == "1234-5678-9012"

    def test_create_gameplay(self):
        config = MacroConfig(gameplay_repeat_count=10)
        factory = MacroFactory()
        macro = factory.create_gameplay(config)
        assert macro.name == "gameplay"
        assert macro.repeat_count == 10
        assert macro.randomize_timing is True

    def test_create_exit_game(self):
        factory = MacroFactory()
        macro = factory.create_exit_game()
        assert macro.name == "exit_game"
        assert len(macro.steps) > 0

    def test_create_toggle_vpn(self):
        factory = MacroFactory()
        macro = factory.create_toggle_vpn()
        assert macro.name == "toggle_vpn"
        # Має містити кроки launch_app та key_event
        app_steps = [s for s in macro.steps if s.action == MacroAction.LAUNCH_APP]
        assert len(app_steps) >= 1

    def test_create_full_session_macro(self):
        config = MacroConfig(games_per_vpn_session=2)
        factory = MacroFactory()
        composer = factory.create_full_session_macro(config)
        assert composer.loop_forever is True
        # Має бути: (enter + gameplay + exit) × 2 + toggle_vpn = 7 макросів
        assert len(composer.sequences) == 7

    def test_scaling(self):
        """Координати масштабуються під розширення."""
        factory = MacroFactory(resolution_width=1920, resolution_height=1080)
        macro = factory.create_launch_fortnite()
        # Перший tap має бути відмасштабований
        first_tap = next(s for s in macro.steps if s.action == MacroAction.TAP)
        assert first_tap.x == 960  # 480 * 1920/960


# ============================================================================
# TEST: SESSION
# ============================================================================


class TestSessionState:
    """Тести станів сесії."""

    def test_all_states(self):
        assert SessionState.IDLE == "idle"
        assert SessionState.MACRO_RUNNING == "macro_running"
        assert SessionState.VPN_CONNECTING == "vpn_connecting"
        assert SessionState.STOPPED == "stopped"
        assert SessionState.ERROR == "error"


class TestGameSessionStats:
    """Тести статистики сесії."""

    def test_initial(self):
        stats = GameSessionStats()
        assert stats.duration_minutes == 0.0
        assert stats.games_played == 0

    def test_duration(self):
        stats = GameSessionStats(
            started_at=time.time() - 120,  # 2 хвилини тому
            finished_at=time.time(),
        )
        assert 1.9 < stats.duration_minutes < 2.5

    def test_running_duration(self):
        stats = GameSessionStats(started_at=time.time() - 60)
        # Без finished_at — рахує від поточного часу
        assert 0.9 < stats.duration_minutes < 1.5


# ============================================================================
# TEST: EXCEPTIONS
# ============================================================================


class TestExceptions:
    """Тести ієрархії виключень."""

    def test_hierarchy(self):
        """Всі виключення наслідують від EpicBotError."""
        from src.core.exceptions import EpicBotError

        assert issubclass(EmulatorError, EpicBotError)
        assert issubclass(LDPlayerError, EmulatorError)
        assert issubclass(LDPlayerNotFoundError, LDPlayerError)
        assert issubclass(VPNError, EmulatorError)
        assert issubclass(VPNConnectionError, VPNError)
        assert issubclass(VPNTimeoutError, VPNError)
        assert issubclass(MacroError, EmulatorError)
        assert issubclass(MacroPlaybackError, MacroError)
        assert issubclass(APKError, EmulatorError)
        assert issubclass(APKInstallError, APKError)
        assert issubclass(AccountCreationError, EmulatorError)
        assert issubclass(SessionError, EmulatorError)

    def test_can_catch_parent(self):
        """Можна ловити виключення через батьківський клас."""
        with pytest.raises(EmulatorError):
            raise LDPlayerNotFoundError("test")

        with pytest.raises(VPNError):
            raise VPNTimeoutError("test")

    def test_exception_message(self):
        err = LDPlayerNotFoundError("LDPlayer not found at C:\\test")
        assert "LDPlayer not found" in str(err)


# ============================================================================
# TEST: INTEGRATION (з моками)
# ============================================================================


class TestMacroPlayerWithMock:
    """Тести MacroPlayer з замоканим LDPlayer."""

    def test_play_simple_macro(self):
        """Відтворення простого макросу."""
        # Мокаємо LDPlayer
        mock_ldplayer = MagicMock()
        mock_instance = EmulatorInstance(index=0, name="Mock")

        from src.emulator.macros import MacroPlayer
        player = MacroPlayer(mock_ldplayer, mock_instance)

        # Створюємо простий макрос
        seq = MacroSequence(name="test")
        seq.add_tap(100, 200, delay_ms=10)
        seq.add_wait(10)

        success = player.play(seq)
        assert success is True
        # Перевіряємо що adb_tap був викликаний
        mock_ldplayer.adb_tap.assert_called()

    def test_play_with_stop(self):
        """Зупинка макросу через stop_check."""
        mock_ldplayer = MagicMock()
        mock_instance = EmulatorInstance(index=0, name="Mock")

        from src.emulator.macros import MacroPlayer
        call_count = 0

        def stop_after_first():
            nonlocal call_count
            call_count += 1
            return call_count > 1

        player = MacroPlayer(
            mock_ldplayer, mock_instance,
            stop_check=stop_after_first,
        )

        seq = MacroSequence(name="test", repeat_count=100)
        seq.add_tap(100, 200, delay_ms=10)

        success = player.play(seq)
        assert success is False  # Було зупинено

    def test_get_status(self):
        mock_ldplayer = MagicMock()
        mock_instance = EmulatorInstance(index=0, name="Mock")

        from src.emulator.macros import MacroPlayer
        player = MacroPlayer(mock_ldplayer, mock_instance)

        status = player.get_status()
        assert status['is_playing'] is False
        assert status['is_paused'] is False


# ============================================================================
# MAIN
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
