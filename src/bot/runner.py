"""
Запуск и управление ботами.
"""

import time
import threading
from typing import Optional, Dict, Callable, Any

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

from ..core import get_logger, BadCredentialsError, BrowserClosedError
from ..core import init_db, get_settings, get_setting, fetch_accounts
from ..browser import create_browser, close_browser
from .. import vision

logger = get_logger(__name__)


class BotRunner:
    """Запуск бота для одного аккаунта."""
    
    def __init__(
        self,
        account: Dict[str, str],
        island_code: str,
        headless: bool = False,
        proxy: Optional[Dict[str, str]] = None,
        manual_lobby_event: Optional[threading.Event] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        stop_check: Optional[Callable[[], bool]] = None
    ):
        self.account = account
        self.island_code = island_code
        self.headless = headless
        self.proxy = proxy
        self.manual_lobby_event = manual_lobby_event
        self.status_callback = status_callback
        self.stop_check = stop_check or (lambda: False)
        
        self.browser = None
        self.context = None
        self.page = None
        
        self._login = account.get('login', 'unknown')
    
    def _log(self, message: str):
        """Логирует и отправляет статус."""
        logger.info(f"[{self._login}] {message}")
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass
    
    def _should_stop(self) -> bool:
        """Проверяет, запрошена ли остановка."""
        try:
            return self.stop_check()
        except Exception:
            return False
    
    def _close_browser(self):
        """Закрывает браузер."""
        try:
            if self.page:
                try:
                    self.page.close()
                except Exception:
                    pass
            if self.context:
                try:
                    self.context.close()
                except Exception:
                    pass
            if self.browser:
                try:
                    self.browser.close()
                except Exception:
                    pass
            self.page = None
            self.context = None
            self.browser = None
        except Exception as e:
            logger.debug(f"Error closing browser: {e}")
    
    def run(self) -> bool:
        """
        Запускает бота.
        
        Returns:
            True если успешно
        """
        try:
            if self._should_stop():
                self._log("Остановлен до запуска")
                return False
            
            self._log("Запуск браузера...")
            
            # Создаём уникальный профиль для аккаунта (для сохранения сессии)
            import os
            import re
            from ..core import ROOT_DIR
            safe_login = re.sub(r'[^\w\-.]', '_', self._login)  # Безопасное имя для папки
            profile_dir = os.path.join(ROOT_DIR, 'browser-profiles', safe_login)
            os.makedirs(profile_dir, exist_ok=True)
            
            self.browser, self.context, self.page = create_browser(
                headless=self.headless,
                proxy=self.proxy,
                profile_dir=profile_dir,
            )
            
            if self._should_stop():
                self._log("Остановлен")
                return False
            
            self._log("Переход на Xbox Cloud Gaming...")
            self.page.goto("https://www.xbox.com/play", wait_until="domcontentloaded", timeout=30000)
            
            if self._should_stop():
                self._log("Остановлен")
                return False
            
            # Логин
            if not self._handle_login():
                return False
            
            if self._should_stop():
                self._log("Остановлен")
                return False
            
            # Поиск и запуск Fortnite
            if not self._find_and_launch_fortnite():
                return False
            
            if self._should_stop():
                self._log("Остановлен")
                return False
            
            # Ожидание игры
            if not self._wait_for_game():
                return False
            
            if self._should_stop():
                self._log("Остановлен")
                return False
            
            # Навигация к острову
            if not self._navigate_to_island():
                return False
            
            if self._should_stop():
                self._log("Остановлен")
                return False
            
            # Время на острове - проверяем остановку каждую секунду
            time_on_island = int(get_setting('time_on_island_min', 15))
            self._log(f"На острове. Ожидание {time_on_island} мин...")
            for _ in range(time_on_island * 60):
                if self._should_stop():
                    self._log("Остановлен пользователем")
                    return False
                time.sleep(1)
            
            self._log("Успешно завершено!")
            return True
            
        except BadCredentialsError:
            self._log("Неверный логин/пароль")
            raise
        except BrowserClosedError:
            self._log("Браузер закрыт")
            raise
        except Exception as e:
            self._log(f"Ошибка: {e}")
            return False
        finally:
            self._cleanup()
    
    def _handle_controller_dialog(self) -> bool:
        """Закрывает диалог 'Подключите контроллер' если он появился."""
        try:
            # Ищем кнопку "PŘESTO POKRAČOVAT" (Всё равно продолжить) или аналоги
            continue_selectors = [
                # Чешский
                'button:has-text("PŘESTO POKRAČOVAT")',
                'button:has-text("Přesto pokračovat")',
                # Английский
                'button:has-text("Continue anyway")',
                'button:has-text("CONTINUE ANYWAY")',
                'button:has-text("Continue without")',
                # Русский
                'button:has-text("Продолжить")',
                'button:has-text("Всё равно продолжить")',
                'button:has-text("Продолжить без")',
                # Немецкий
                'button:has-text("Trotzdem fortfahren")',
                # Испанский
                'button:has-text("Continuar de todos modos")',
                # Французский
                'button:has-text("Continuer quand même")',
            ]
            
            for sel in continue_selectors:
                try:
                    btn = self.page.locator(sel).first
                    if btn.count() > 0 and btn.is_visible():
                        self._log("Закрытие диалога контроллера...")
                        btn.click(force=True)
                        time.sleep(1)
                        return True
                except Exception:
                    continue
            
            # Пробуем по роли
            continue_names = [
                "PŘESTO POKRAČOVAT",
                "Přesto pokračovat",
                "Continue anyway",
                "CONTINUE ANYWAY",
                "Продолжить",
            ]
            
            for name in continue_names:
                try:
                    btn = self.page.get_by_role("button", name=name)
                    if btn.count() > 0 and btn.first.is_visible():
                        self._log(f"Нажимаем '{name}'...")
                        btn.first.click(force=True)
                        time.sleep(1)
                        return True
                except Exception:
                    pass
            
            # Пробуем найти по тексту на странице
            try:
                btn = self.page.get_by_text("PŘESTO POKRAČOVAT", exact=False)
                if btn.count() > 0 and btn.first.is_visible():
                    self._log("Найден текст PŘESTO POKRAČOVAT, кликаем...")
                    btn.first.click(force=True)
                    time.sleep(1)
                    return True
            except Exception:
                pass
            
            # Крестик закрытия диалога
            close_selectors = [
                'button[aria-label="Close"]',
                'button[aria-label="Zavřít"]',
                'button[aria-label="Закрыть"]',
                '[class*="close"]:not([class*="closed"])',
                '[class*="Close"]:not([class*="Closed"])',
                'button:has-text("×")',
                'button:has-text("✕")',
            ]
            
            for sel in close_selectors:
                try:
                    btn = self.page.locator(sel).first
                    if btn.count() > 0 and btn.is_visible():
                        self._log("Закрытие диалога (крестик)...")
                        btn.click(force=True)
                        time.sleep(1)
                        return True
                except Exception:
                    continue
            
            return False
        except Exception as e:
            logger.debug(f"Ошибка обработки диалога контроллера: {e}")
            return False

    def _handle_fullscreen_popup(self) -> bool:
        """Закрывает popup про fullscreen режим Xbox Cloud Gaming."""
        try:
            # Ищем popup с текстом про fullscreen/celou obrazovku
            # "Nechcete, aby se hry spouštěly na celou obrazovku?"
            close_selectors = [
                # Крестик закрытия popup
                '[class*="dismissButton"]',
                '[class*="dismiss"]',
                '[class*="close-button"]',
                '[aria-label*="Dismiss"]',
                '[aria-label*="dismiss"]', 
                '[aria-label*="Close"]',
                '[aria-label*="Zavřít"]',
                # Кнопка X в popup
                'button:has-text("×")',
                'button:has-text("✕")',
                'button:has-text("X")',
            ]
            
            for sel in close_selectors:
                try:
                    btn = self.page.locator(sel).first
                    if btn.count() > 0 and btn.is_visible():
                        btn.click(force=True)
                        time.sleep(0.5)
                        return True
                except Exception:
                    continue
            
            # Пробуем закрыть по Escape
            # Это также может закрыть другие popup'ы
            return False
        except Exception:
            return False

    def _handle_title_screen(self) -> bool:
        """Обрабатывает титульный экран Fortnite - нажимает любую клавишу для продолжения."""
        try:
            # На титульном экране обычно написано "Press any key" или подобное
            # Просто нажимаем Enter или Space для продолжения
            self._log("Титульный экран - нажимаем для продолжения...")
            self.page.keyboard.press('Enter')
            time.sleep(2)
            return True
        except Exception:
            return False

    def _handle_login(self) -> bool:
        """Обрабатывает логин в Microsoft."""
        try:
            if self._should_stop():
                return False
            
            self._log("Проверка авторизации...")
            time.sleep(2)
            
            if self._should_stop():
                return False
            
            # Проверяем текущий URL - возможно мы уже на странице логина Microsoft
            try:
                current_url = self.page.url.lower()
                already_on_login = "login.live" in current_url or "login.microsoftonline" in current_url
            except Exception:
                already_on_login = False
            
            sign_in_clicked = already_on_login  # Если уже на login странице - не нужно кликать
            
            # Сначала проверяем есть ли кнопка входа на странице Xbox Cloud Gaming
            # Только если мы НЕ на странице логина Microsoft
            if not already_on_login:
                sign_in_selectors = [
                    'a[href*="login"]',
                    'button:has-text("Sign in")',
                    'a:has-text("Sign in")',
                    'button:has-text("PŘIHLÁSIT")',
                    'a:has-text("PŘIHLÁSIT")',
                    'button:has-text("Войти")',
                    'a:has-text("Войти")',
                    '[data-bi-id="sign-in"]',
                    '[class*="sign-in"]',
                    '[class*="SignIn"]',
                    '[aria-label*="Sign in"]',
                    '[aria-label*="sign in"]',
                ]
                
                for selector in sign_in_selectors:
                    if self._should_stop():
                        return False
                    try:
                        btn = self.page.locator(selector).first
                        if btn.count() > 0 and btn.is_visible():
                            self._log("Найдена кнопка входа, кликаем...")
                            btn.click()
                            sign_in_clicked = True
                            # Ждём навигации после клика
                            try:
                                self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                            except Exception:
                                pass
                            time.sleep(2)
                            break
                    except Exception:
                        continue
            
            if self._should_stop():
                return False
            
            # Если кликнули на вход - даём время на загрузку страницы логина
            if sign_in_clicked:
                time.sleep(2)
            
            if self._should_stop():
                return False
            
            # Ждём появления формы входа Microsoft
            max_wait = 30
            start = time.time()
            while time.time() - start < max_wait:
                if self._should_stop():
                    return False
                
                try:
                    current_url = self.page.url.lower()
                except Exception:
                    time.sleep(1)
                    continue
                
                # Страница логина Microsoft
                if "login.microsoftonline" in current_url or "login.live" in current_url:
                    self._log("Страница входа Microsoft...")
                    break
                    
                # Уже авторизован - проверяем наличие аватара пользователя
                try:
                    user_avatar = self.page.locator('[class*="avatar"], [class*="user-pic"], [class*="profile-pic"]')
                    if user_avatar.count() > 0:
                        self._log("Уже авторизован!")
                        return True
                except Exception:
                    pass
                
                time.sleep(1)
            
            if self._should_stop():
                return False
            
            # Проверяем есть ли экран "Выбор учетной записи" (аккаунт уже сохранён)
            login_email = self.account.get('login', '').strip().lower()
            account_picker_handled = False
            
            # Ищем сохранённый аккаунт по email
            for _ in range(5):
                if self._should_stop():
                    return False
                
                # Проверяем есть ли список аккаунтов
                try:
                    # Ищем элемент с нашим email
                    account_tile = self.page.locator(f'[data-test-id*="{login_email}"], [title*="{login_email}"]')
                    if account_tile.count() > 0 and account_tile.first.is_visible():
                        self._log(f"Найден сохранённый аккаунт: {login_email}")
                        account_tile.first.click()
                        account_picker_handled = True
                        time.sleep(3)
                        break
                except Exception:
                    pass
                
                # Пробуем найти по тексту email
                try:
                    account_by_text = self.page.get_by_text(login_email, exact=False)
                    if account_by_text.count() > 0 and account_by_text.first.is_visible():
                        self._log(f"Выбираем сохранённый аккаунт: {login_email}")
                        account_by_text.first.click()
                        account_picker_handled = True
                        time.sleep(3)
                        break
                except Exception:
                    pass
                
                # Также проверяем по частичному совпадению (без учёта регистра)
                try:
                    # Ищем все элементы списка аккаунтов
                    account_items = self.page.locator('[data-test-id], .table-cell, [role="option"], [role="listitem"]')
                    for i in range(account_items.count()):
                        if self._should_stop():
                            return False
                        try:
                            item = account_items.nth(i)
                            item_text = item.text_content() or ""
                            if login_email in item_text.lower():
                                self._log(f"Выбираем аккаунт из списка: {login_email}")
                                item.click()
                                account_picker_handled = True
                                time.sleep(3)
                                break
                        except Exception:
                            continue
                    if account_picker_handled:
                        break
                except Exception:
                    pass
                
                time.sleep(1)
            
            if self._should_stop():
                return False
            
            # Если выбрали сохранённый аккаунт - ждём загрузки страницы
            if account_picker_handled:
                self._log("Ожидание после выбора аккаунта...")
                
                # Ждём навигации/редиректа
                try:
                    self.page.wait_for_load_state("domcontentloaded", timeout=15000)
                except Exception:
                    pass
                
                time.sleep(3)
                
                if self._should_stop():
                    return False
                
                # Ждём пока URL станет xbox.com (редирект может занять время)
                for _ in range(15):
                    if self._should_stop():
                        return False
                    try:
                        current_url = self.page.url.lower()
                        if "xbox.com" in current_url and "login" not in current_url:
                            # Дополнительно ждём загрузки страницы
                            try:
                                self.page.wait_for_load_state("networkidle", timeout=10000)
                            except Exception:
                                pass
                            time.sleep(2)
                            self._log("Авторизация через сохранённый аккаунт завершена!")
                            return True
                    except Exception:
                        pass
                    time.sleep(1)
                
                # Если не перешли на xbox - возможно нужен пароль
                self._log("Редирект на Xbox не произошёл, продолжаем авторизацию...")
            
            if self._should_stop():
                return False
            
            # Теперь обрабатываем форму входа Microsoft
            # Email (если не выбрали сохранённый аккаунт)
            try:
                email_input = self.page.locator('input[type="email"], input[name="loginfmt"]')
                if email_input.count() > 0 and not account_picker_handled:
                    self._log("Ввод email...")
                    email_input.first.fill(self.account.get('login', ''))
                    
                    # Кнопка Next/Далее
                    next_btn = self.page.locator('input[type="submit"], button[type="submit"], #idSIButton9')
                    if next_btn.count() > 0:
                        next_btn.first.click()
                    else:
                        self.page.keyboard.press('Enter')
                    time.sleep(3)
            except Exception as e:
                # Если страница изменилась - это нормально
                self._log(f"Пропускаем ввод email: {e}")
            
            if self._should_stop():
                return False
            
            # Проверяем появился ли выбор "Отправить код" / "Использовать пароль"
            # Нужно нажать "Используйте свой пароль" / "Use password" / "Sign in with password"
            # Элемент может быть <a>, <span role="button">, <button> и т.д.
            use_password_clicked = False
            
            # Пробуем несколько раз кликнуть на ссылку "Использовать пароль"
            for attempt in range(5):
                if use_password_clicked or self._should_stop():
                    break
                
                # Проверяем не авторизованы ли мы уже
                try:
                    if "xbox.com" in self.page.url.lower() and "login" not in self.page.url.lower():
                        self._log("Уже авторизованы!")
                        return True
                except Exception:
                    pass
                    
                # Проверяем, не появилось ли уже поле пароля
                try:
                    pass_check = self.page.locator('input[type="password"], input[name="passwd"]')
                    if pass_check.count() > 0:
                        if pass_check.first.is_visible():
                            self._log("Поле пароля уже видно")
                            use_password_clicked = True
                            break
                except Exception:
                    pass
                
                # Ищем ссылку по тексту (getByText более надёжный)
                password_texts = [
                    "Используйте свой пароль",
                    "Use password", 
                    "Sign in with password",
                    "Use your password",
                ]
                
                for text in password_texts:
                    try:
                        link = self.page.get_by_text(text, exact=False)
                        if link.count() > 0 and link.first.is_visible():
                            self._log(f"Найдена ссылка: {text}")
                            # Пробуем обычный клик
                            try:
                                link.first.click(force=True)
                                use_password_clicked = True
                                time.sleep(2)
                                break
                            except Exception:
                                # Пробуем JavaScript клик
                                try:
                                    link.first.evaluate("el => el.click()")
                                    use_password_clicked = True
                                    time.sleep(2)
                                    break
                                except Exception:
                                    pass
                    except Exception:
                        continue
                
                if not use_password_clicked:
                    time.sleep(1)
            
            if self._should_stop():
                return False
            
            # Ожидаем появления поля пароля (до 15 секунд)
            pass_input = None
            for _ in range(15):
                if self._should_stop():
                    return False
                pass_locator = self.page.locator('input[type="password"], input[name="passwd"]')
                if pass_locator.count() > 0:
                    try:
                        if pass_locator.first.is_visible():
                            pass_input = pass_locator.first
                            break
                    except Exception:
                        pass
                time.sleep(1)
            
            if self._should_stop():
                return False
            
            # Ввод пароля
            if pass_input:
                self._log("Ввод пароля...")
                pass_input.fill(self.account.get('password', ''))
                time.sleep(0.5)
                
                # Кнопка Sign in
                sign_in_btn = self.page.locator('input[type="submit"], button[type="submit"], #idSIButton9')
                if sign_in_btn.count() > 0:
                    sign_in_btn.first.click()
                else:
                    self.page.keyboard.press('Enter')
                time.sleep(3)
            else:
                self._log("Поле пароля не найдено!")
            
            if self._should_stop():
                return False
            
            # "Stay signed in?" / "Не выходить из системы?" - ждём и нажимаем Yes/Да
            for _ in range(15):
                if self._should_stop():
                    return False
                
                # Ищем кнопку "Да" или "Yes"
                yes_btn = None
                
                # Пробуем разные селекторы
                yes_selectors = [
                    'button:has-text("Да")',
                    'button:has-text("Yes")',
                    'input[value="Да"]',
                    'input[value="Yes"]',
                    '#idSIButton9',
                    '#acceptButton',
                    'button[type="submit"]',
                ]
                
                for sel in yes_selectors:
                    try:
                        btn = self.page.locator(sel).first
                        if btn.count() > 0 and btn.is_visible():
                            yes_btn = btn
                            break
                    except Exception:
                        continue
                
                # Также пробуем по тексту
                if not yes_btn:
                    try:
                        btn = self.page.get_by_text("Да", exact=True)
                        if btn.count() > 0 and btn.first.is_visible():
                            yes_btn = btn.first
                    except Exception:
                        pass
                
                if yes_btn:
                    self._log("Подтверждение 'Stay signed in'...")
                    try:
                        yes_btn.click(force=True)
                    except Exception:
                        try:
                            yes_btn.evaluate("el => el.click()")
                        except Exception:
                            pass
                    time.sleep(2)
                    break
                    
                time.sleep(1)
            
            if self._should_stop():
                return False
            
            # Проверяем ошибки
            error_elem = self.page.locator('[id*="error"], [class*="error"]')
            if error_elem.count() > 0:
                try:
                    if error_elem.first.is_visible():
                        error_text = error_elem.first.text_content()
                        if error_text and ("incorrect" in error_text.lower() or "wrong" in error_text.lower() or "invalid" in error_text.lower()):
                            raise BadCredentialsError("Неверные учетные данные")
                except Exception:
                    pass
            
            # Ждём перенаправления обратно на Xbox
            self._log("Ожидание перенаправления...")
            for _ in range(20):
                if self._should_stop():
                    return False
                if "xbox.com" in self.page.url.lower():
                    self._log("Авторизация завершена!")
                    # Ждём полной загрузки и установки cookies
                    try:
                        self.page.wait_for_load_state("networkidle", timeout=10000)
                    except Exception:
                        pass
                    time.sleep(5)  # Даём время cookies установиться
                    return True
                time.sleep(1)
            
            self._log("Авторизация завершена")
            return True
        except BadCredentialsError:
            raise
        except Exception as e:
            self._log(f"Ошибка логина: {e}")
            return False
    
    def _find_and_launch_fortnite(self) -> bool:
        """Поиск и запуск Fortnite на Xbox Cloud Gaming."""
        self._log("Поиск Fortnite...")
        
        try:
            if self._should_stop():
                return False
            
            # Сначала убедимся что мы на xbox.com/play
            try:
                current_url = self.page.url.lower()
                if "xbox.com" not in current_url:
                    self._log("Не на Xbox, переходим...")
                    self.page.goto("https://www.xbox.com/play", wait_until="domcontentloaded", timeout=30000)
                    time.sleep(3)
            except Exception:
                pass
            
            # Ждём загрузки главной страницы
            try:
                self.page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            
            time.sleep(2)
            
            if self._should_stop():
                return False
            
            # Пробуем прямой URL на страницу Fortnite - это быстрее и надёжнее
            self._log("Переход на страницу Fortnite...")
            self.page.goto("https://www.xbox.com/play/games/fortnite/BT5P2X999VH2", 
                          wait_until="domcontentloaded", timeout=30000)
            
            # Ждём полной загрузки страницы
            try:
                self.page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            
            time.sleep(3)
            
            if self._should_stop():
                return False
            
            # Ищем кнопку Play/Играть - более специфичные селекторы
            # Кнопка Play на Xbox обычно зелёная и находится в верхней части страницы
            play_selectors = [
                # Специфичные селекторы для Xbox Cloud Gaming
                'button[class*="ProductActionButton"]:has-text("Hrát")',
                'button[class*="ProductActionButton"]:has-text("Play")',
                'button[class*="ProductActionButton"]:has-text("Играть")',
                'a[class*="ProductActionButton"]:has-text("Hrát")',
                'a[class*="ProductActionButton"]:has-text("Play")',
                # Зелёная кнопка Play
                'button[style*="background"]:has-text("Hrát")',
                'button[style*="background"]:has-text("Play")',
                # Кнопка рядом с иконкой play
                'button:has(svg):has-text("Hrát")',
                'button:has(svg):has-text("Play")',
                # По data атрибутам
                '[data-bi-id="play-button"]',
                '[data-bi-name*="play"]',
                # Общие но только button (не ссылки в описании)
                'button[aria-label*="Play"]',
                'button[aria-label*="Hrát"]',
            ]
            
            # Проверим, авторизованы ли мы - если видим "PŘIHLÁSIT SE" или "Sign in", значит нет
            try:
                not_logged_in_selectors = [
                    'button:has-text("PŘIHLÁSIT SE")',
                    'button:has-text("Sign in")', 
                    'button:has-text("Войти")',
                    'a:has-text("PŘIHLÁSIT SE")',
                    'a:has-text("Sign in")',
                ]
                for sel in not_logged_in_selectors:
                    try:
                        sign_in_btn = self.page.locator(sel).first
                        if sign_in_btn.count() > 0 and sign_in_btn.is_visible():
                            self._log("Не авторизованы на странице игры! Кликаем Sign in...")
                            sign_in_btn.click()
                            time.sleep(2)
                            
                            # Ждём перехода - может быть на login.live ИЛИ обратно на страницу игры
                            try:
                                self.page.wait_for_load_state("domcontentloaded", timeout=10000)
                            except Exception:
                                pass
                            
                            self._log("Ожидание авторизации...")
                            
                            # Ждём либо redirect на login.live, либо возврат на страницу игры
                            auth_start = time.time()
                            need_manual_login = False
                            while time.time() - auth_start < 30:
                                if self._should_stop():
                                    return False
                                try:
                                    current_url = self.page.url.lower()
                                    
                                    # Если попали на страницу Microsoft login - нужно авторизоваться
                                    if "login.live" in current_url or "login.microsoftonline" in current_url:
                                        self._log("Требуется авторизация Microsoft...")
                                        need_manual_login = True
                                        break
                                    
                                    # Если вернулись на страницу Fortnite - авторизация прошла автоматически
                                    if "xbox.com/play" in current_url or "xbox.com" in current_url and "fortnite" in current_url:
                                        # Проверяем есть ли теперь кнопка Play (а не Sign in)
                                        try:
                                            play_check = self.page.locator('button:has-text("Play"), button:has-text("Hrát"), button:has-text("Играть")')
                                            if play_check.count() > 0:
                                                self._log("Авторизация прошла автоматически!")
                                                break
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                                time.sleep(0.5)
                            
                            # Если нужна ручная авторизация - выполняем
                            if need_manual_login:
                                if not self._handle_login():
                                    self._log("Не удалось авторизоваться")
                                    return False
                                
                                # После авторизации возвращаемся на страницу игры
                                self._log("Возвращаемся на страницу Fortnite...")
                                time.sleep(2)
                                self.page.goto("https://www.xbox.com/play/games/fortnite/BT5P2X999VH2", 
                                              wait_until="domcontentloaded", timeout=30000)
                                try:
                                    self.page.wait_for_load_state("networkidle", timeout=15000)
                                except Exception:
                                    pass
                                time.sleep(3)
                            break
                    except Exception:
                        continue
            except Exception as e:
                self._log(f"Ошибка проверки авторизации: {e}")
            
            # Логируем URL один раз
            try:
                current_url = self.page.url
                self._log(f"Ищем кнопку Play на: {current_url[:60]}...")
            except Exception:
                pass
            
            for attempt in range(15):  # Ждём до 15 секунд появления кнопки Play
                if self._should_stop():
                    return False
                
                # Проверяем ещё раз не появилась ли кнопка входа
                if attempt == 5:  # На 5-й попытке
                    try:
                        sign_in_check = self.page.locator('button:has-text("PŘIHLÁSIT SE"), button:has-text("Sign in")')
                        if sign_in_check.count() > 0 and sign_in_check.first.is_visible():
                            self._log("Всё ещё не авторизованы - кнопка Play недоступна")
                            return False
                    except Exception:
                        pass
                
                for sel in play_selectors:
                    try:
                        play_btn = self.page.locator(sel).first
                        if play_btn.count() > 0 and play_btn.is_visible():
                            self._log("Нажимаем Play...")
                            try:
                                play_btn.click(force=True, timeout=5000)
                                time.sleep(2)
                                return True
                            except Exception as click_err:
                                self._log(f"Ошибка клика Play: {click_err}")
                                # Пробуем JavaScript клик
                                try:
                                    play_btn.evaluate("el => el.click()")
                                    time.sleep(2)
                                    return True
                                except Exception:
                                    pass
                    except Exception:
                        continue
                
                # Пробуем найти кнопку по роли - это надёжнее чем по тексту
                # Ищем кнопку (не ссылку!) с текстом Hrát/Play
                try:
                    play_btn = self.page.get_by_role("button", name="Hrát")
                    if play_btn.count() > 0 and play_btn.first.is_visible():
                        self._log("Нажимаем кнопку Hrát...")
                        play_btn.first.click()
                        time.sleep(2)
                        # Обрабатываем диалог контроллера
                        self._handle_controller_dialog()
                        return True
                except Exception:
                    pass
                
                try:
                    play_btn = self.page.get_by_role("button", name="Play")
                    if play_btn.count() > 0 and play_btn.first.is_visible():
                        self._log("Нажимаем кнопку Play...")
                        play_btn.first.click()
                        time.sleep(2)
                        # Обрабатываем диалог контроллера
                        self._handle_controller_dialog()
                        return True
                except Exception:
                    pass
                
                try:
                    play_btn = self.page.get_by_role("button", name="Играть")
                    if play_btn.count() > 0 and play_btn.first.is_visible():
                        self._log("Нажимаем кнопку Играть...")
                        play_btn.first.click()
                        time.sleep(2)
                        # Обрабатываем диалог контроллера
                        self._handle_controller_dialog()
                        return True
                except Exception:
                    pass
                    
                time.sleep(1)
            
            self._log("Кнопка Play не найдена")
            return False
            
        except Exception as e:
            self._log(f"Ошибка поиска Fortnite: {e}")
            return False
    
    def _wait_for_game(self) -> bool:
        """Ожидает загрузки игры."""
        self._log("Ожидание загрузки игры...")
        
        # Сразу проверяем диалог контроллера - он появляется после нажатия Play
        self._handle_controller_dialog()
        
        start = time.time()
        timeout = 300  # 5 минут - Xbox Cloud может долго подключаться
        connecting_start = None
        last_state = None
        login_attempts = 0
        controller_dialog_checked = 0
        
        while time.time() - start < timeout:
            # Проверяем остановку
            if self._should_stop():
                self._log("Остановлен во время ожидания игры")
                return False
            
            # Периодически проверяем диалог контроллера (первые 30 секунд)
            if controller_dialog_checked < 10 and time.time() - start < 30:
                self._handle_controller_dialog()
                self._handle_fullscreen_popup()
                controller_dialog_checked += 1
            
            try:
                # Проверяем состояние экрана
                state = vision.detect_screen_state(self.page)
                
                if state == vision.ScreenState.LOBBY:
                    self._log("Лобби обнаружено!")
                    return True
                
                if state == vision.ScreenState.IN_GAME:
                    self._log("Игра готова!")
                    return True
                
                if state == vision.ScreenState.MENU:
                    self._log("Меню обнаружено - пробуем закрыть...")
                    self.page.keyboard.press('Escape')
                    time.sleep(1)
                    continue
                
                # Титульный экран Fortnite - нужно нажать кнопку для продолжения
                if state == vision.ScreenState.TITLE_SCREEN:
                    self._log("Титульный экран Fortnite - нажимаем для продолжения...")
                    self.page.keyboard.press('Enter')
                    time.sleep(3)
                    # Проверяем изменилось ли состояние
                    new_state = vision.detect_screen_state(self.page)
                    if new_state in (vision.ScreenState.LOBBY, vision.ScreenState.IN_GAME):
                        self._log("Перешли в лобби!")
                        return True
                    continue
                
                # Резервная логика: если UNKNOWN слишком долго - пробуем нажать
                if state == vision.ScreenState.UNKNOWN and time.time() - start > 90:
                    # После 90 секунд если состояние UNKNOWN - пробуем нажать
                    self._log("Неизвестное состояние - пробуем нажать Enter...")
                    self.page.keyboard.press('Enter')
                    time.sleep(2)
                    # Проверяем изменилось ли состояние
                    new_state = vision.detect_screen_state(self.page)
                    if new_state in (vision.ScreenState.LOBBY, vision.ScreenState.IN_GAME, vision.ScreenState.TITLE_SCREEN):
                        self._log("Состояние изменилось!")
                        if new_state in (vision.ScreenState.LOBBY, vision.ScreenState.IN_GAME):
                            return True
                        continue
                
                # Если на странице входа Microsoft - пробуем войти
                # Но ТОЛЬКО если URL указывает на login.live или login.microsoftonline
                if state == vision.ScreenState.LOGIN_PAGE and login_attempts < 3:
                    try:
                        current_url = self.page.url.lower()
                        if "login.live" in current_url or "login.microsoftonline" in current_url:
                            self._log("Страница входа Microsoft - пробуем авторизоваться...")
                            login_attempts += 1
                            if self._handle_login():
                                time.sleep(3)
                                continue
                    except Exception:
                        pass
                
                # Ручной сигнал
                if self.manual_lobby_event and self.manual_lobby_event.is_set():
                    self._log("Ручное подтверждение лобби")
                    return True
                
                # Логируем только при смене состояния
                if state != last_state:
                    self._log(f"Состояние: {state.name}")
                    last_state = state
                    
                    # Сброс таймера CONNECTING при смене состояния
                    if state == vision.ScreenState.CONNECTING:
                        connecting_start = time.time()
                
                # Если CONNECTING слишком долго (> 3 минут), пробуем обновить страницу
                if state == vision.ScreenState.CONNECTING and connecting_start:
                    if time.time() - connecting_start > 180:
                        self._log("CONNECTING слишком долго, обновляем страницу...")
                        self.page.reload(wait_until="domcontentloaded", timeout=30000)
                        connecting_start = time.time()
                        time.sleep(5)
                        continue
                
                time.sleep(2)
                
            except Exception as e:
                self._log(f"Ошибка проверки состояния: {e}")
                time.sleep(1)
        
        self._log("Таймаут ожидания игры")
        return False
    
    def _navigate_to_island(self) -> bool:
        """Навигация к острову через UI Fortnite."""
        self._log(f"Навигация к острову: {self.island_code}")
        
        try:
            if self._should_stop():
                return False
            
            # В Fortnite лобби можно использовать клавиатуру для навигации
            # Открываем меню выбора режима (Tab или Escape)
            self._log("Открытие меню выбора режима...")
            time.sleep(3)  # Даём лобби полностью загрузиться
            
            if self._should_stop():
                return False
            
            # Нажимаем Tab для открытия меню режимов или Enter для Play
            # В Xbox Cloud Gaming управление через геймпад эмулируется клавиатурой
            
            # Пробуем открыть меню острова через интерфейс
            # Сначала попробуем использовать клавишу для открытия меню "Play"
            self.page.keyboard.press('Enter')
            time.sleep(2)
            
            if self._should_stop():
                return False
            
            # Ищем кнопку "Island Code" или "Discover" через UI
            # Эмулируем нажатия геймпада: стрелки для навигации
            
            # Нажимаем несколько раз вверх/вниз чтобы найти нужный пункт меню
            # В Fortnite обычно: вверх-вверх для "Island Code"
            
            self._log("Навигация к Island Code...")
            
            # Пробуем найти и кликнуть на "ISLAND CODE" или "Change Island" текст
            island_code_selectors = [
                'button:has-text("Island Code")',
                'button:has-text("ISLAND CODE")',
                'button:has-text("Change")',
                '[data-testid*="island"]',
            ]
            
            found_island_button = False
            for sel in island_code_selectors:
                try:
                    btn = self.page.locator(sel).first
                    if btn.count() > 0 and btn.is_visible():
                        self._log("Найдена кнопка Island Code")
                        btn.click()
                        found_island_button = True
                        time.sleep(2)
                        break
                except Exception:
                    continue
            
            if self._should_stop():
                return False
            
            # Если не нашли кнопку - используем клавиатурную навигацию
            if not found_island_button:
                self._log("Использование клавиатурной навигации...")
                # Tab открывает меню в Fortnite
                self.page.keyboard.press('Tab')
                time.sleep(1)
                
                # Стрелки для навигации
                for _ in range(3):
                    self.page.keyboard.press('ArrowUp')
                    time.sleep(0.3)
                
                self.page.keyboard.press('Enter')
                time.sleep(1)
            
            if self._should_stop():
                return False
            
            # Вводим код острова
            self._log(f"Ввод кода острова: {self.island_code}")
            
            # Ищем поле ввода
            input_selectors = [
                'input[type="text"]',
                'input[placeholder*="code"]',
                'input[placeholder*="Code"]',
                '[contenteditable="true"]',
            ]
            
            input_found = False
            for sel in input_selectors:
                try:
                    input_field = self.page.locator(sel).first
                    if input_field.count() > 0 and input_field.is_visible():
                        input_field.fill(self.island_code)
                        input_found = True
                        time.sleep(1)
                        break
                except Exception:
                    continue
            
            if not input_found:
                # Пробуем просто набрать код - может быть активное поле ввода
                self._log("Попытка прямого ввода кода...")
                self.page.keyboard.type(self.island_code, delay=100)
            
            time.sleep(1)
            
            if self._should_stop():
                return False
            
            # Подтверждаем
            self.page.keyboard.press('Enter')
            time.sleep(2)
            
            # Ищем кнопку Play/Launch
            launch_selectors = [
                'button:has-text("Play")',
                'button:has-text("PLAY")',
                'button:has-text("Launch")',
                'button:has-text("Go")',
                'button:has-text("Hrát")',
            ]
            
            for sel in launch_selectors:
                try:
                    btn = self.page.locator(sel).first
                    if btn.count() > 0 and btn.is_visible():
                        self._log("Запуск острова...")
                        btn.click()
                        time.sleep(2)
                        break
                except Exception:
                    continue
            
            self._log("Навигация завершена")
            return True
            
        except Exception as e:
            self._log(f"Ошибка навигации: {e}")
            return False
    
    def _cleanup(self):
        """Закрывает ресурсы."""
        close_browser(self.browser, self.context, self.page)


def run_bot(
    account: Dict[str, str],
    island_code: str,
    headless: bool = False,
    proxy: Optional[Dict[str, str]] = None,
    manual_lobby_event: Optional[threading.Event] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    runner_callback: Optional[Callable[["BotRunner"], None]] = None
) -> bool:
    """
    Запускает бота для одного аккаунта.
    
    Args:
        account: Словарь с 'login' и 'password'
        island_code: Код острова
        headless: Режим без интерфейса
        proxy: Прокси конфигурация
        manual_lobby_event: Событие ручного подтверждения лобби
        status_callback: Колбэк для статуса
        stop_check: Функция для проверки, запрошена ли остановка
        runner_callback: Колбэк для получения ссылки на runner
    
    Returns:
        True если успешно
    """
    runner = BotRunner(
        account=account,
        island_code=island_code,
        headless=headless,
        proxy=proxy,
        manual_lobby_event=manual_lobby_event,
        status_callback=status_callback,
        stop_check=stop_check,
    )
    
    # Передаём ссылку на runner для возможности принудительного закрытия
    if runner_callback:
        try:
            runner_callback(runner)
        except Exception:
            pass
    
    return runner.run()
