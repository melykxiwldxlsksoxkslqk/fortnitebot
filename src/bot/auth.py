"""
Authentication module for Xbox Cloud Gaming login.
Handles Microsoft account authentication with email/password flow.
"""

import re
import asyncio
from typing import TYPE_CHECKING, Optional, Callable

if TYPE_CHECKING:
    from playwright.sync_api import Page, BrowserContext

from ..core.exceptions import BadCredentialsError, CodeRequiredError
from ..core.logger import get_logger

logger = get_logger(__name__)


def _emit_status(message: str, status_callback: Optional[Callable] = None) -> None:
    """Emit status message if callback is provided."""
    logger.info(message)
    if status_callback:
        try:
            status_callback(message)
        except Exception:
            pass


def click_element_safe(page: "Page", selector: str, timeout: int = 5000) -> bool:
    """Safely click an element with timeout."""
    try:
        el = page.locator(selector).first
        if el and el.is_visible(timeout=timeout):
            el.click(timeout=timeout)
            return True
    except Exception:
        pass
    return False


def wait_for_any_selector(page: "Page", selectors: list, timeout: int = 30000) -> Optional[str]:
    """Wait for any of the given selectors to appear."""
    for selector in selectors:
        try:
            el = page.locator(selector).first
            if el and el.is_visible(timeout=timeout // len(selectors)):
                return selector
        except Exception:
            continue
    return None


def switch_to_password_mode(page: "Page") -> bool:
    """Switch from passwordless to password mode if needed."""
    try:
        # Try to click 'Use password instead' or similar
        password_mode_selectors = [
            'a:has-text("Use your password instead")',
            'a:has-text("Use a password instead")',
            'a:has-text("Sign in with a password instead")',
            '#idA_PWD_SwitchToPassword',
            'a[id*="SwitchToPassword"]',
        ]
        for sel in password_mode_selectors:
            if click_element_safe(page, sel, timeout=3000):
                page.wait_for_timeout(1000)
                return True
    except Exception:
        pass
    return False


def handle_kmsi(page: "Page") -> None:
    """Handle 'Keep Me Signed In' prompt."""
    try:
        # Try to find and click 'Yes' or 'No' button
        kmsi_selectors = [
            '#acceptButton',
            'input[value="Yes"]',
            'button:has-text("Yes")',
            '#idSIButton9',
        ]
        for sel in kmsi_selectors:
            if click_element_safe(page, sel, timeout=3000):
                logger.info("KMSI: Clicked Yes/Accept")
                page.wait_for_timeout(1000)
                return
    except Exception:
        pass


def microsoft_login(
    page: "Page",
    email: str,
    password: str,
    status_callback: Optional[Callable] = None,
    max_retries: int = 3
) -> bool:
    """
    Perform Microsoft account login with email and password.
    
    Args:
        page: Playwright page object
        email: Microsoft account email
        password: Account password
        status_callback: Optional callback for status updates
        max_retries: Maximum number of login attempts
        
    Returns:
        True if login successful, False otherwise
        
    Raises:
        BadCredentialsError: If credentials are invalid
        CodeRequiredError: If verification code is required
    """
    _emit_status("Начинаю вход в аккаунт Microsoft", status_callback)
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Login attempt {attempt + 1}/{max_retries}")
            
            # Wait for email input
            email_selectors = [
                'input[type="email"]',
                'input[name="loginfmt"]',
                '#i0116',
            ]
            
            email_input = None
            for sel in email_selectors:
                try:
                    el = page.locator(sel).first
                    if el and el.is_visible(timeout=5000):
                        email_input = el
                        break
                except Exception:
                    continue
                    
            if not email_input:
                _emit_status("Не найдено поле email", status_callback)
                continue
                
            # Enter email
            _emit_status("Ввожу email", status_callback)
            email_input.fill("")
            email_input.type(email, delay=50)
            page.wait_for_timeout(500)
            
            # Click Next
            next_selectors = [
                '#idSIButton9',
                'input[type="submit"]',
                'button:has-text("Next")',
                'button:has-text("Далее")',
            ]
            next_clicked = False
            for sel in next_selectors:
                if click_element_safe(page, sel, timeout=3000):
                    next_clicked = True
                    break
                    
            if not next_clicked:
                page.keyboard.press('Enter')
                
            page.wait_for_timeout(2000)
            
            # Check for email error
            try:
                error_el = page.locator('#usernameError, .alert-error, #usernameError').first
                if error_el and error_el.is_visible(timeout=1000):
                    error_text = error_el.inner_text()
                    logger.error(f"Email error: {error_text}")
                    raise BadCredentialsError(f"Invalid email: {error_text}")
            except BadCredentialsError:
                raise
            except Exception:
                pass
                
            # Switch to password mode if needed
            switch_to_password_mode(page)
            
            # Wait for password input
            _emit_status("Ввожу пароль", status_callback)
            password_selectors = [
                'input[type="password"]',
                'input[name="passwd"]',
                '#i0118',
            ]
            
            password_input = None
            for sel in password_selectors:
                try:
                    el = page.locator(sel).first
                    if el and el.is_visible(timeout=5000):
                        password_input = el
                        break
                except Exception:
                    continue
                    
            if not password_input:
                # Maybe we need to switch to password mode
                switch_to_password_mode(page)
                page.wait_for_timeout(1500)
                
                for sel in password_selectors:
                    try:
                        el = page.locator(sel).first
                        if el and el.is_visible(timeout=3000):
                            password_input = el
                            break
                    except Exception:
                        continue
                        
            if not password_input:
                _emit_status("Не найдено поле пароля", status_callback)
                continue
                
            # Enter password
            password_input.fill("")
            password_input.type(password, delay=30)
            page.wait_for_timeout(500)
            
            # Click Sign in
            signin_selectors = [
                '#idSIButton9',
                'input[type="submit"]',
                'button:has-text("Sign in")',
                'button:has-text("Войти")',
            ]
            signin_clicked = False
            for sel in signin_selectors:
                if click_element_safe(page, sel, timeout=3000):
                    signin_clicked = True
                    break
                    
            if not signin_clicked:
                page.keyboard.press('Enter')
                
            page.wait_for_timeout(3000)
            
            # Check for password error
            try:
                error_selectors = [
                    '#passwordError',
                    '.alert-error',
                    '#error_desc',
                    '#idTD_Error',
                ]
                for sel in error_selectors:
                    error_el = page.locator(sel).first
                    if error_el and error_el.is_visible(timeout=1000):
                        error_text = error_el.inner_text()
                        if 'incorrect' in error_text.lower() or 'wrong' in error_text.lower():
                            raise BadCredentialsError(f"Invalid password: {error_text}")
            except BadCredentialsError:
                raise
            except Exception:
                pass
                
            # Check for code required
            try:
                code_selectors = [
                    'input[name="otc"]',
                    '#idTxtBx_OTC_Password',
                    'input[aria-label*="code"]',
                    'input[placeholder*="code"]',
                ]
                for sel in code_selectors:
                    code_el = page.locator(sel).first
                    if code_el and code_el.is_visible(timeout=2000):
                        raise CodeRequiredError("Verification code required")
            except CodeRequiredError:
                raise
            except Exception:
                pass
                
            # Handle KMSI
            handle_kmsi(page)
            
            # Check if login successful (redirected to Xbox)
            page.wait_for_timeout(2000)
            current_url = page.url
            if 'xbox.com' in current_url or 'play/launch' in current_url:
                _emit_status("Вход выполнен успешно", status_callback)
                return True
                
            # Check for successful login by looking for profile elements
            try:
                profile_selectors = [
                    '[data-testid="user-menu"]',
                    '.gamertag',
                    '[aria-label*="Profile"]',
                ]
                for sel in profile_selectors:
                    el = page.locator(sel).first
                    if el and el.is_visible(timeout=3000):
                        _emit_status("Вход выполнен успешно", status_callback)
                        return True
            except Exception:
                pass
                
            # If we're still on login page after all steps, continue to next attempt
            if 'login' in page.url.lower() or 'signin' in page.url.lower():
                logger.warning(f"Still on login page after attempt {attempt + 1}")
                continue
                
            # Assume success if we got past login
            _emit_status("Вход предположительно выполнен", status_callback)
            return True
            
        except (BadCredentialsError, CodeRequiredError):
            raise
        except Exception as e:
            logger.error(f"Login attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                page.wait_for_timeout(2000)
                continue
                
    _emit_status("Не удалось войти после всех попыток", status_callback)
    return False


def try_login_flow(
    page: "Page",
    email: str,
    password: str,
    status_callback: Optional[Callable] = None
) -> bool:
    """
    Complete login flow with error handling.
    
    Args:
        page: Playwright page object
        email: Microsoft account email
        password: Account password
        status_callback: Optional callback for status updates
        
    Returns:
        True if login successful
    """
    try:
        return microsoft_login(page, email, password, status_callback)
    except BadCredentialsError as e:
        logger.error(f"Bad credentials: {e}")
        _emit_status(f"Ошибка авторизации: {e}", status_callback)
        raise
    except CodeRequiredError as e:
        logger.error(f"Code required: {e}")
        _emit_status(f"Требуется код подтверждения", status_callback)
        raise
    except Exception as e:
        logger.error(f"Login flow error: {e}")
        _emit_status(f"Ошибка входа: {e}", status_callback)
        return False
