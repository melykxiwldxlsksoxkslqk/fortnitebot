"""
Tests for Canvas Navigation module.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Optional

# Import canvas module components
from src.bot.canvas import (
    ScreenState,
    NavigationDirection,
    GamepadButton,
    CanvasElement,
    ScreenSnapshot,
    NavigationResult,
    CanvasNavigator,
    create_navigator,
    NAVIGATION_KEYS,
    TIMEOUTS,
)


class TestScreenState:
    """Test ScreenState enum."""
    
    def test_all_states_exist(self):
        """Verify all expected screen states are defined."""
        expected_states = [
            'UNKNOWN', 'LOADING', 'MAIN_MENU', 'LOBBY',
            'SEARCH_PANEL', 'ISLAND_PREVIEW', 'MATCHMAKING',
            'IN_GAME', 'PAUSE_MENU', 'ERROR_DIALOG'
        ]
        for state_name in expected_states:
            assert hasattr(ScreenState, state_name)
    
    def test_states_are_unique(self):
        """Ensure all states have unique values."""
        values = [s.value for s in ScreenState]
        assert len(values) == len(set(values))


class TestNavigationDirection:
    """Test NavigationDirection enum."""
    
    def test_all_directions_exist(self):
        """Verify all directions are defined."""
        assert NavigationDirection.UP.value == 'up'
        assert NavigationDirection.DOWN.value == 'down'
        assert NavigationDirection.LEFT.value == 'left'
        assert NavigationDirection.RIGHT.value == 'right'
    
    def test_navigation_keys_mapping(self):
        """Verify navigation keys are mapped correctly."""
        for direction in NavigationDirection:
            assert direction in NAVIGATION_KEYS
            assert isinstance(NAVIGATION_KEYS[direction], list)
            assert len(NAVIGATION_KEYS[direction]) >= 1


class TestGamepadButton:
    """Test GamepadButton enum."""
    
    def test_essential_buttons_exist(self):
        """Verify essential gamepad buttons are defined."""
        essential = ['A', 'B', 'X', 'Y', 'START', 'SELECT']
        for btn_name in essential:
            assert hasattr(GamepadButton, btn_name)
    
    def test_button_values_are_strings(self):
        """All button values should be keyboard keys (strings)."""
        for button in GamepadButton:
            assert isinstance(button.value, str)
    
    def test_a_button_is_enter(self):
        """A button should map to Enter for confirmation."""
        assert GamepadButton.A.value == 'Enter'
    
    def test_b_button_is_escape(self):
        """B button should map to Escape for cancel."""
        assert GamepadButton.B.value == 'Escape'


class TestCanvasElement:
    """Test CanvasElement dataclass."""
    
    def test_element_creation(self):
        """Test basic element creation."""
        elem = CanvasElement(
            name="test_button",
            x=100, y=200,
            width=50, height=30
        )
        assert elem.name == "test_button"
        assert elem.x == 100
        assert elem.y == 200
        assert elem.width == 50
        assert elem.height == 30
    
    def test_element_center(self):
        """Test center property calculation."""
        elem = CanvasElement(
            name="test",
            x=100, y=200,
            width=50, height=30
        )
        cx, cy = elem.center
        assert cx == 125  # 100 + 50/2
        assert cy == 215  # 200 + 30/2
    
    def test_element_bounds(self):
        """Test bounds property."""
        elem = CanvasElement(
            name="test",
            x=10, y=20,
            width=100, height=50
        )
        assert elem.bounds == (10, 20, 100, 50)
    
    def test_element_with_confidence(self):
        """Test element with confidence score."""
        elem = CanvasElement(
            name="test",
            x=0, y=0,
            width=10, height=10,
            confidence=0.85
        )
        assert elem.confidence == 0.85


class TestScreenSnapshot:
    """Test ScreenSnapshot dataclass."""
    
    def test_snapshot_creation(self):
        """Test basic snapshot creation."""
        snapshot = ScreenSnapshot(state=ScreenState.LOBBY)
        assert snapshot.state == ScreenState.LOBBY
        assert snapshot.elements == []
        assert snapshot.frame_hash == ""
    
    def test_has_element(self):
        """Test has_element method."""
        elem = CanvasElement(name="PlayButton", x=0, y=0, width=10, height=10)
        snapshot = ScreenSnapshot(
            state=ScreenState.LOBBY,
            elements=[elem]
        )
        assert snapshot.has_element("PlayButton")
        assert snapshot.has_element("playbutton")  # Case insensitive
        assert not snapshot.has_element("NonExistent")
    
    def test_get_element(self):
        """Test get_element method."""
        elem = CanvasElement(name="SearchIcon", x=50, y=50, width=20, height=20)
        snapshot = ScreenSnapshot(
            state=ScreenState.LOBBY,
            elements=[elem]
        )
        found = snapshot.get_element("SearchIcon")
        assert found is not None
        assert found.name == "SearchIcon"
        
        not_found = snapshot.get_element("Missing")
        assert not_found is None


class TestNavigationResult:
    """Test NavigationResult dataclass."""
    
    def test_successful_result(self):
        """Test successful navigation result."""
        result = NavigationResult(
            success=True,
            action="click_play",
            from_state=ScreenState.LOBBY,
            to_state=ScreenState.MATCHMAKING,
            duration_ms=1500
        )
        assert result.success is True
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed navigation result."""
        result = NavigationResult(
            success=False,
            action="open_search",
            from_state=ScreenState.LOBBY,
            to_state=ScreenState.LOBBY,
            duration_ms=5000,
            error="Search icon not found"
        )
        assert result.success is False
        assert result.error is not None


class TestTimeouts:
    """Test timeout constants."""
    
    def test_timeouts_exist(self):
        """Verify essential timeouts are defined."""
        assert 'screen_change' in TIMEOUTS
        assert 'element_appear' in TIMEOUTS
        assert 'action_cooldown' in TIMEOUTS
        assert 'loading_max' in TIMEOUTS
    
    def test_timeouts_are_positive(self):
        """All timeouts should be positive integers."""
        for name, value in TIMEOUTS.items():
            assert isinstance(value, int)
            assert value > 0


class TestCanvasNavigatorInit:
    """Test CanvasNavigator initialization."""
    
    def test_init_with_mock_page(self):
        """Test navigator creation with mock page."""
        mock_page = Mock()
        nav = CanvasNavigator(mock_page)
        assert nav.page is mock_page
        assert nav._last_snapshot is None
        assert nav._action_history == []
    
    def test_init_with_callback(self):
        """Test navigator with status callback."""
        mock_page = Mock()
        callback = Mock()
        nav = CanvasNavigator(mock_page, status_callback=callback)
        assert nav.status_callback is callback
    
    def test_create_navigator_function(self):
        """Test convenience function."""
        mock_page = Mock()
        nav = create_navigator(mock_page)
        assert isinstance(nav, CanvasNavigator)


class TestCanvasNavigatorMethods:
    """Test CanvasNavigator methods with mocks."""
    
    @pytest.fixture
    def navigator(self):
        """Create navigator with mock page."""
        mock_page = Mock()
        mock_page.viewport_size = {'width': 1280, 'height': 720}
        mock_page.keyboard = Mock()
        mock_page.mouse = Mock()
        mock_page.wait_for_timeout = Mock()
        mock_page.locator = Mock(return_value=Mock(
            first=Mock(
                is_visible=Mock(return_value=False),
                bounding_box=Mock(return_value=None)
            )
        ))
        return CanvasNavigator(mock_page)
    
    def test_get_canvas_bounds_fallback(self, navigator):
        """Test canvas bounds returns viewport as fallback."""
        bounds = navigator.get_canvas_bounds()
        assert bounds == (0, 0, 1280, 720)
    
    def test_press_button(self, navigator):
        """Test button press."""
        navigator.press_button(GamepadButton.A)
        navigator.page.keyboard.press.assert_called_with('Enter')
    
    def test_navigate_direction(self, navigator):
        """Test directional navigation."""
        navigator.navigate(NavigationDirection.UP, times=2)
        assert navigator.page.keyboard.press.call_count == 2
    
    def test_type_text(self, navigator):
        """Test text typing."""
        navigator.type_text("1234-5678-9012")
        navigator.page.keyboard.type.assert_called_once()
    
    def test_click_at(self, navigator):
        """Test coordinate click."""
        navigator.click_at(500, 300)
        navigator.page.mouse.click.assert_called_with(500, 300)
    
    def test_move(self, navigator):
        """Test movement."""
        navigator.move('forward', 500)
        navigator.page.keyboard.down.assert_called_with('w')
        navigator.page.keyboard.up.assert_called_with('w')
    
    def test_jump(self, navigator):
        """Test jump action."""
        navigator.jump()
        navigator.page.keyboard.press.assert_called_with('Space')
    
    def test_action_history(self, navigator):
        """Test action history tracking."""
        navigator.press_button(GamepadButton.A)
        navigator.jump()
        
        history = navigator.get_action_history()
        assert len(history) >= 2
    
    def test_clear_history(self, navigator):
        """Test history clearing."""
        navigator.press_button(GamepadButton.A)
        navigator.clear_history()
        assert navigator.get_action_history() == []


class TestCanvasNavigatorScreenDetection:
    """Test screen detection with mocked vision."""
    
    @pytest.fixture
    def navigator_with_vision(self):
        """Create navigator with mocked vision module."""
        mock_page = Mock()
        mock_page.viewport_size = {'width': 1280, 'height': 720}
        mock_page.screenshot = Mock(return_value=b'fake_screenshot_data')
        
        mock_vision = Mock()
        mock_vision.capture_page_bgr = Mock(return_value=None)
        mock_vision.find_template = Mock(return_value=None)
        
        nav = CanvasNavigator(mock_page, vision_module=mock_vision)
        return nav
    
    def test_detect_screen_state_unknown(self, navigator_with_vision):
        """Test detection returns UNKNOWN when no templates match."""
        snapshot = navigator_with_vision.detect_screen_state()
        assert snapshot.state == ScreenState.UNKNOWN
    
    def test_snapshot_stored(self, navigator_with_vision):
        """Test that last snapshot is stored when image is captured."""
        # Mock capture to return an image
        import numpy as np
        mock_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        navigator_with_vision._vision.capture_page_bgr = Mock(return_value=mock_img)
        
        navigator_with_vision.detect_screen_state()
        assert navigator_with_vision._last_snapshot is not None


class TestCanvasNavigatorIntegration:
    """Integration-style tests with more realistic mocks."""
    
    def test_search_and_launch_flow(self):
        """Test the full search and launch flow with mocks."""
        mock_page = Mock()
        mock_page.viewport_size = {'width': 1280, 'height': 720}
        mock_page.keyboard = Mock()
        mock_page.mouse = Mock()
        mock_page.wait_for_timeout = Mock()
        mock_page.locator = Mock(return_value=Mock(
            first=Mock(
                is_visible=Mock(return_value=True),
                bounding_box=Mock(return_value={
                    'x': 100, 'y': 100, 'width': 50, 'height': 50
                })
            )
        ))
        
        nav = CanvasNavigator(mock_page)
        
        # Mock internal methods for this test
        nav.ensure_focus = Mock(return_value=True)
        nav.open_search = Mock(return_value=True)
        nav.type_island_code = Mock(return_value=True)
        nav.submit_search = Mock(return_value=True)
        nav.select_island = Mock(return_value=True)
        nav.click_play = Mock(return_value=True)
        
        result = nav.search_and_launch_island("1234-5678-9012")
        
        assert result is True
        nav.ensure_focus.assert_called_once()
        nav.open_search.assert_called_once()
        nav.type_island_code.assert_called_once_with("1234-5678-9012")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_island_code(self):
        """Test handling of empty island code."""
        mock_page = Mock()
        mock_page.viewport_size = {'width': 1280, 'height': 720}
        
        nav = CanvasNavigator(mock_page)
        nav._emit = Mock()  # Suppress logging
        
        result = nav.search_and_launch_island("")
        assert result is False
    
    def test_canvas_not_found(self):
        """Test handling when canvas element not found."""
        mock_page = Mock()
        mock_page.viewport_size = None
        mock_page.locator = Mock(return_value=Mock(
            first=Mock(is_visible=Mock(return_value=False))
        ))
        
        nav = CanvasNavigator(mock_page)
        bounds = nav.get_canvas_bounds()
        assert bounds is None
