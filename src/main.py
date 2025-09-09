from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError
import time
import os
import re
# import pyautogui
import random
from threading import Lock
from typing import Dict
# Импорт, совместимый с запуском как модуля и как скрипта
try:
	from . import vision
	from .bot_logic import FortniteEnv
	from . import db as dbmod
except ImportError:
	import sys
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
	from src import vision
	from src.bot_logic import FortniteEnv
	from src import db as dbmod

try:
	from . import stream_input
except ImportError:
	import src.stream_input as stream_input

_ACTIVE_BROWSERS: Dict[str, object] = {}
_ACTIVE_BROWSERS_LOCK = Lock()

class BadCredentialsError(Exception):
	"""Выбрасывается при неверных логине/пароле Microsoft."""
	pass

class CodeRequiredError(Exception):
	"""Требуется вход по коду (не поддерживается выбранной стратегией)."""
	pass

class BrowserClosedError(Exception):
	"""Браузер/страница были закрыты пользователем во время сеанса."""
	pass

def load_settings():
	"""Читает настройки из БД."""
	try:
		dbmod.init_db()
		s = dbmod.get_settings()
		return s or {}
	except Exception:
		return {}

def load_accounts():
	"""Загружает учетные записи из БД."""
	try:
		dbmod.init_db()
		return dbmod.fetch_accounts()
	except Exception:
		return []

def load_island_code():
	"""Читает код острова из БД, иначе дефолт."""
	try:
		dbmod.init_db()
		code = dbmod.get_setting('island_code', '')
		return code or "1234-5678-9012"
	except Exception:
		return "1234-5678-9012"

def run_bot(account, island_code, headless=False, proxy=None, manual_lobby_event=None):
	"""
	Main function to run the bot for a single account.
	"""
	# Унифицированный ключ логина (поддержим и старый 'email' на всякий случай)
	account_login = account.get('login') or account.get('email', '')
	# Проверка на плейсхолдеры
	if 'your-email' in account_login or 'your-password' in account['password']:
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		print("!!! ERROR: You are using the default placeholder credentials. !!!")
		print("!!! Please open `config/accounts.json` and enter your real   !!!")
		print("!!! Microsoft account email and password.                   !!!")
		print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		return False

	def verify_required_assets():
		required = [
			'assets/creative_mode_button.png',
			'assets/island_code_button.png',
			'assets/island_code_input_field.png',
			'assets/launch_island_button.png',
		]
		missing = []
		for pth in required:
			rp = vision.resolve_asset_path(pth)
			if not os.path.exists(rp):
				missing.append(pth)
		if missing:
			print("Warning: The following assets are missing and RL in-game automation will be skipped:")
			for p in missing:
				print(f" - {p}")
			return False
		return True

	# --- Debug Setup ---
	debug_dir = 'debug'
	os.makedirs(debug_dir, exist_ok=True)
	step_counter = 0
	# MODIFIED: Now takes the 'page' object and screenshots the browser viewport
	def take_debug_screenshot(page, name):
		nonlocal step_counter
		step_counter += 1
		path = os.path.join(debug_dir, f"{step_counter:02d}_{name}.png")
		try:
			page.screenshot(path=path)
			print(f"DEBUG: Saved screenshot of the browser window to {path}")
		except Exception as e:
			print(f"DEBUG: Could not take screenshot: {e}")

	def close_xbox_overlay_if_present(p):
		"""Закрывает левую боковую панель/гид Xbox, если она открыта."""
		try:
			guide_markers = [
				'button:has-text("QUIT GAME")',
				r'text=/Currently playing/i',
				r'text=/Have a game session code\?/i',
			]
			visible = False
			for sel in guide_markers:
				try:
					loc = p.locator(sel).first
					if loc and loc.is_visible():
						visible = True
						break
				except Exception:
					continue
			if visible:
				# Попробуем нажать Х закрытия, иначе Esc
				try:
					close_btn = p.locator('button[aria-label*="Close"], [aria-label*="Закрыть"], [data-icon="close"]').first
					if close_btn and close_btn.is_visible():
						close_btn.click()
						p.wait_for_timeout(200)
					else:
						p.keyboard.press('Escape')
						p.wait_for_timeout(200)
					take_debug_screenshot(p, 'closed_xbox_overlay')
				except Exception:
					pass
		except Exception:
			pass

	print(f"Starting bot for account: {account_login}")
	with sync_playwright() as p:
		# Нормализуем и фиксируем код острова (из аргумента или из БД)
		island_code = (island_code or load_island_code() or "").strip()
		print(f"USING ISLAND CODE: '{island_code}'")
		result_success = False
		# Гарантия: один аккаунт → один браузер. Закрываем старый, если он ещё жив.
		try:
			with _ACTIVE_BROWSERS_LOCK:
				old = _ACTIVE_BROWSERS.get(account_login)
				if old is not None:
					try:
						old.close()
					except Exception:
						pass
					_ACTIVE_BROWSERS.pop(account_login, None)
		except Exception:
			pass
		def open_browser(headless_flag, proxy_conf=None):
			launch_kwargs = {"headless": headless_flag, "args": ["--start-maximized"]}
			# Configure proxy if provided
			if proxy_conf and proxy_conf.get('host') and proxy_conf.get('port'):
				server = f"http://{proxy_conf['host']}:{proxy_conf['port']}"
				proxy_opts = {"server": server}
				if proxy_conf.get('username'):
					proxy_opts["username"] = proxy_conf.get('username')
				if proxy_conf.get('password'):
					proxy_opts["password"] = proxy_conf.get('password')
				launch_kwargs["proxy"] = proxy_opts

			# Try to load browser extensions (ModernKit if available, plus RPA as fallback)
			try:
				# helper: find extension directory by manifest.json
				def _find_ext_dir(root):
					candidates = [root, os.path.join(root, 'dist'), os.path.join(root, 'build'), os.path.join(root, 'extension')]
					for c in candidates:
						if c and os.path.exists(os.path.join(c, 'manifest.json')):
							return c
					for cur_root, dirs, files in os.walk(root):
						depth = cur_root[len(root):].count(os.sep)
						if depth > 4:
							continue
						if 'manifest.json' in files:
							return cur_root
					return None

				ext_dirs = []

				# Better xCloud userscript loader (optional)
				bx_ext_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'third_party', 'better-xcloud-userscript'))
				if os.path.exists(os.path.join(bx_ext_dir, 'manifest.json')):
					ext_dirs.append(bx_ext_dir)

				# Ui.Vision RPA extension (bundled)
				rpa_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'third_party', 'RPA', 'extension'))
				if os.path.exists(os.path.join(rpa_dir, 'manifest.json')):
					ext_dirs.append(rpa_dir)

				# Extra extensions from env (semicolon/OS pathsep separated)
				extra_exts = os.environ.get('EXTRA_EXT_DIRS')
				if extra_exts:
					for raw in extra_exts.split(os.pathsep):
						pth = raw.strip().strip('"')
						if pth and os.path.exists(os.path.join(pth, 'manifest.json')):
							ext_dirs.append(pth)

				# Optionally allow ModernKit if explicitly enabled
				if os.environ.get('ALLOW_MODERNKIT') == '1':
					env_path = os.environ.get('MODERNKIT_EXT_DIR')
					if env_path and os.path.exists(os.path.join(env_path, 'manifest.json')):
						ext_dirs.append(env_path)
					else:
						base_modernkit = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'third_party', 'xcloud-keyboard-mouse'))
						if os.path.isdir(base_modernkit):
							mk = _find_ext_dir(base_modernkit)
							if mk:
								ext_dirs.append(mk)

				# de-duplicate while preserving order
				unique_ext_dirs = []
				for d in ext_dirs:
					if d not in unique_ext_dirs:
						unique_ext_dirs.append(d)

				if unique_ext_dirs:
					profile_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'chrome-profile'))
					os.makedirs(profile_dir, exist_ok=True)
					joined = ",".join(unique_ext_dirs)
					args = [
						"--start-maximized",
						f"--disable-extensions-except={joined}",
						f"--load-extension={joined}",
					]
					ctx = p.chromium.launch_persistent_context(user_data_dir=profile_dir, headless=False, args=args, proxy=launch_kwargs.get('proxy'))
					return ctx.browser, ctx, ctx.new_page()
			except Exception:
				pass
			
			# Fallback: обычный запуск без расширений
			b = p.chromium.launch(**launch_kwargs)
			c = b.new_context(no_viewport=True)
			return b, c, c.new_page()

		browser, context, page = open_browser(headless, proxy)
		# Сохраняем текущий браузер как активный для аккаунта
		try:
			with _ACTIVE_BROWSERS_LOCK:
				_ACTIVE_BROWSERS[account_login] = browser
		except Exception:
			pass

		try:
			print("Navigating to Xbox Cloud Gaming...")
			page.goto("https://www.xbox.com/play", wait_until="load")
			time.sleep(3)
			take_debug_screenshot(page, "initial_page")

			# Handle cookie consent banner
			try:
				print("Checking for cookie consent banner...")
				accept_button = page.locator('button:has-text("Accept"), button:has-text("Přijmout")').first
				accept_button.wait_for(timeout=7000)
				if accept_button.is_visible():
					print("Cookie consent banner found. Clicking it.")
					accept_button.click()
					page.wait_for_timeout(2000)
					take_debug_screenshot(page, "after_cookie_accept")
			except Exception:
				print("Cookie consent banner not found or already handled, skipping.")

			# Helper: optionally dismiss Microsoft Account Checkup prompt if it appears
			def handle_optional_account_checkup(p) -> bool:
				try:
					if "account.microsoft.com" in p.url:
						selectors = [
							'button[aria-label*="Close"]',
							'button[aria-label*="Закрыть"]',
							'div[role="dialog"] button[aria-label*="Close"]',
							'div[role="dialog"] button[aria-label*="Закрыть"]',
							'button:has-text("Skip")',
							'button:has-text("Not now")',
							'button:has-text("Remind me later")',
							'button:has-text("Skip for now")',
							'button:has-text("Пропустить")',
							'button:has-text("Не сейчас")',
							'button:has-text("Напомнить позже")',
							'button:has-text("Позже")',
							'a:has-text("Skip")',
							'a:has-text("Пропустить")',
							'a:has-text("Не сейчас")',
							'a:has-text("Позже")',
						]
						deadline = time.time() + 20
						while time.time() < deadline:
							for sel in selectors:
								try:
									btn = p.locator(sel).first
									if btn and btn.is_visible():
										btn.click()
										p.wait_for_timeout(500)
										return True
								except Exception:
									continue
							# frames
							for fr in p.frames:
								for sel in selectors:
									try:
										btn = fr.locator(sel).first
										if btn and btn.is_visible():
											btn.click()
											fr.wait_for_timeout(500)
											return True
									except Exception:
										continue
							# try Esc as fallback
							try:
								p.keyboard.press("Escape")
								p.wait_for_timeout(300)
							except Exception:
								pass
							time.sleep(0.3)
						# As a last resort, try navigating away
						try:
							p.goto("https://www.xbox.com/play", wait_until="domcontentloaded")
							p.wait_for_load_state('networkidle', timeout=15000)
							return True
						except Exception:
							pass
				except Exception:
					pass
				return False

			# --- Robust Login Flow with localization and fallback ---
			def microsoft_login(page, login, password):
				# 1) Email
				email_field = page.locator('#i0116, input[name="loginfmt"], input[type="email"]').first
				email_field.wait_for(timeout=30000)
				take_debug_screenshot(page, "email_page")
				email_field.fill(login)
				page.locator('#idSIButton9, input[type="submit"], button:has-text("Next"), button:has-text("Далее")').first.click()
				print("Email submitted.")
				page.wait_for_load_state('domcontentloaded', timeout=15000)
				take_debug_screenshot(page, "after_email_submit")
 
				# 1a*) Иногда система предлагает вход по коду. Переключаемся на вход по паролю.
				def switch_to_password_mode(p):
					texts = [
						"Use your password", "Use my password", "Use password instead",
						"Используйте свой пароль", "Использовать пароль",
						"Usar mi contraseña", "Usa tu contraseña", "Utiliser votre mot de passe",
						"Passwort verwenden"
					]
					selectors = (
						[f'a:has-text("{t}")' for t in texts]
						+ [f'text={t}' for t in texts]
						+ ['#idA_PWD_SwitchToPassword']
					)
					deadline = time.time() + 15
					while time.time() < deadline:
						# main document
						for sel in selectors:
							try:
								loc = p.locator(sel).first
								if loc and loc.is_visible():
									take_debug_screenshot(page, "switch_to_password_mode")
									loc.click()
									p.wait_for_load_state('domcontentloaded', timeout=10000)
									return True
							except Exception:
								pass
						# frames
						for fr in p.frames:
							for sel in selectors:
								try:
									loc = fr.locator(sel).first
									if loc and loc.is_visible():
										take_debug_screenshot(page, "switch_to_password_mode_iframe")
										loc.click()
										fr.wait_for_load_state('domcontentloaded', timeout=10000)
										return True
								except Exception:
									continue
						time.sleep(0.5)
					return False

				try:
					switched = switch_to_password_mode(page)
					if switched:
						print("Switched to password login mode.")
				except Exception:
					pass

				# 1a) Account type selection (Personal/Work) may appear
				try:
					# Accept cookies if OneTrust is shown on login.live.com
					cookie_accept = page.locator('#onetrust-accept-btn-handler, button:has-text("Accept all")').first
					if cookie_accept.is_visible():
						cookie_accept.click()
						page.wait_for_timeout(1000)
					account_choice = page.locator('#msaTile, #aadTile').first
					account_choice.wait_for(timeout=5000)
					# Prefer personal (MSA) if available
					if page.locator('#msaTile').count() > 0:
						page.locator('#msaTile').first.click()
					else:
						account_choice.click()
					page.wait_for_load_state('networkidle', timeout=15000)
				except Exception:
					pass

				# 1b) If account picker shows previous accounts, choose "Use another account"
				try:
					other = page.locator('#otherTile, a:has-text("Use another account"), a:has-text("Использовать другую учетную запись")').first
					if other.is_visible():
						other.click()
						page.wait_for_load_state('domcontentloaded', timeout=15000)
				except Exception:
					pass

				# 2) Password (поиск во всех фреймах)
				def find_password_field_in_any_frame(p):
					selectors = '#i0118, input[name="passwd"], input[type="password"]'
					deadline = time.time() + 45
					while time.time() < deadline:
						# сначала основной документ
						try:
							loc = p.locator(selectors).first
							loc.wait_for(timeout=1000)
							return p, loc
						except Exception:
							pass
						# затем фреймы
						for fr in p.frames:
							try:
								loc2 = fr.locator(selectors).first
								loc2.wait_for(timeout=1000)
								return fr, loc2
							except Exception:
								continue
						time.sleep(0.5)
					raise PWTimeoutError("Password field not found")

				try:
					frame_ctx, password_field = find_password_field_in_any_frame(page)
				except PWTimeoutError:
					# Если пароль не найден, но видим элементы сценария входа по коду — сообщаем об этом явно
					def code_flow_detected(p):
						try:
							if p.locator('text="Получите код для входа"').count() > 0:
								return True
						except Exception:
							pass
						try:
							if p.locator('button:has-text("Отправить код"), button:has-text("Send code")').count() > 0:
								return True
						except Exception:
							pass
						for fr in p.frames:
							try:
								if fr.locator('text="Получите код для входа"').count() > 0:
									return True
							except Exception:
								pass
							try:
								if fr.locator('button:has-text("Отправить код"), button:has-text("Send code")').count() > 0:
									return True
							except Exception:
								pass
						return False
					if code_flow_detected(page):
						raise CodeRequiredError("Microsoft предлагает вход по коду; переключение на пароль не удалось")
					raise
				take_debug_screenshot(page, "password_page")
				password_field.fill(password)
				# Универсальная отправка формы пароля: ищем кнопку везде, иначе жмём Enter
				def submit_password(p, ctx, pwd_field):
					selectors = '#idSIButton9, input[type="submit"], button:has-text("Sign in"), button:has-text("Войти"), button:has-text("Далее")'
					# в текущем контексте
					try:
						loc = ctx.locator(selectors).first
						loc.wait_for(timeout=2000)
						loc.click()
						return True
					except Exception:
						pass
					# в корневом документе
					try:
						loc = p.locator(selectors).first
						loc.wait_for(timeout=2000)
						loc.click()
						return True
					except Exception:
						pass
					# во фреймах
					for fr in p.frames:
						try:
							loc = fr.locator(selectors).first
							loc.wait_for(timeout=1000)
							loc.click()
							return True
						except Exception:
							continue
					# фолбэк: Enter по полю пароля
					try:
						pwd_field.press('Enter')
						return True
					except Exception:
						return False

				if not submit_password(page, frame_ctx, password_field):
					raise PWTimeoutError("Unable to submit password: no submit control and Enter fallback failed")
				print("Password submitted.")
				take_debug_screenshot(page, "after_password_submit")

				# 2a) Проверка ошибки неправильного пароля на login.live.com (многоязычно)
				def _has_bad_password_error() -> bool:
					selectors = '#passwordError, #usernameError, div[role="alert"], .error.pageLevel, .alert, .error'
					time.sleep(1)
					texts_all = []
					try:
						texts_all.extend(page.locator(selectors).all_inner_texts())
					except Exception:
						pass
					for fr in page.frames:
						try:
							texts_all.extend(fr.locator(selectors).all_inner_texts())
						except Exception:
							continue
					full_text = "\n".join(t.strip() for t in texts_all if t and t.strip())
					if not full_text:
						return False
					patterns = [
						r"incorrect", r"wrong", r"invalid", r"doesn't exist", r"does not exist",
						r"неверн", r"неправил", r"парол", r"не существует", r"не найден",
						r"密码", r"contraseña", r"mot de passe", r"passwort",
					]
					return any(re.search(p, full_text, re.IGNORECASE) for p in patterns)

				if _has_bad_password_error():
					raise BadCredentialsError("Неверный логин или пароль")
				
				# 3) Stay signed in?
				def handle_kmsi(p) -> bool:
					yes_texts = ["Да", "Yes", "Sí", "Oui", "Ja", "Sim", "Sì"]
					prompt_texts = ["Не выходить из системы", "Stay signed in", "No cerrar sesión", "Rester connecté"]
					deadline = time.time() + 20
					selectors = [
						'#idSIButton9',
						'input[type="submit"][value="Да"]',
					] + [f'button:has-text("{t}")' for t in yes_texts]

					def prompt_visible(ctx):
						for t in prompt_texts:
							try:
								if ctx.locator(f'text={t}').count() > 0:
									return True
							except Exception:
								continue
						return False

					while time.time() < deadline:
						# main document
						if prompt_visible(p):
							for sel in selectors:
								try:
									btn = p.locator(sel).first
									if btn and btn.is_visible():
										take_debug_screenshot(page, "stay_signed_in_prompt")
										btn.click()
										p.wait_for_load_state("networkidle", timeout=15000)
										take_debug_screenshot(page, "stay_signed_in_yes_clicked")
										return True
								except Exception:
									pass
						# frames
						for fr in p.frames:
							if prompt_visible(fr):
								for sel in selectors:
									try:
										btn = fr.locator(sel).first
										if btn and btn.is_visible():
											take_debug_screenshot(page, "stay_signed_in_prompt_iframe")
											btn.click()
											fr.wait_for_load_state("networkidle", timeout=15000)
											take_debug_screenshot(page, "stay_signed_in_yes_clicked_iframe")
											return True
									except Exception:
										continue
						time.sleep(0.5)
					return False

				if not handle_kmsi(page):
					print("'Stay signed in?' prompt did not appear, skipping.")
				
				# После KMSI попробуем закрыть/пропустить необязательную проверку безопасности
				try:
					if handle_optional_account_checkup(page):
						print("Optional Microsoft account checkup was shown and dismissed.")
				except Exception:
					pass

			def try_login_flow(page) -> bool:
				try:
					# broadened selector for many locales
					login_selector = (
						'a:has-text("Sign In"), a:has-text("Sign in"), a:has-text("Log in"), a:has-text("Войти"), '
						'a:has-text("Se connecter"), a:has-text("Anmelden"), a:has-text("Iniciar sesión"), a:has-text("Entrar"), a:has-text("Accedi"), '
						'button:has-text("Sign In"), button:has-text("Sign in"), button:has-text("Log in"), button:has-text("Войти"), '
						'button:has-text("Se connecter"), button:has-text("Anmelden"), button:has-text("Iniciar sesión"), button:has-text("Entrar"), button:has-text("Accedi")'
					)
					login_button = page.locator(login_selector).first
					login_button.wait_for(timeout=12000)
					print("Login button found. Clicking to sign in.")
					take_debug_screenshot(page, "before_login_click")
					login_button.click()
					# Microsoft login form opens on login.live.com
					microsoft_login(page, account_login, account['password'])
					print("Login button flow complete. Waiting for redirect...")
					page.wait_for_load_state('networkidle', timeout=45000)
					# If Microsoft redirected to account checkup, dismiss and go back to xbox.com/play
					try:
						if "account.microsoft.com" in page.url:
							if handle_optional_account_checkup(page):
								page.goto("https://www.xbox.com/play", wait_until="domcontentloaded")
								page.wait_for_load_state('networkidle', timeout=45000)
					except Exception:
						pass
					return True
				except (BadCredentialsError, CodeRequiredError):
					raise
				except Exception as e:
					print(f"Login button flow failed: {e}. Trying direct Microsoft login...")
					try:
						page.goto("https://login.live.com/", wait_until="domcontentloaded")
						microsoft_login(page, account_login, account['password'])
						# Не ждём жёстко networkidle — на медленной сети это часто таймаут
						try:
							page.wait_for_load_state('domcontentloaded', timeout=15000)
						except Exception:
							pass
						try:
							page.goto("https://www.xbox.com/play", wait_until="domcontentloaded")
						except Exception as nav_err:
							# Игнорируем прерванную навигацию, если это ERR_ABORTED (часто возникает из-за авто-редиректов после логина)
							if "ERR_ABORTED" not in str(nav_err):
								raise
						# Дождаться нужного URL; если не вышло — попробуем обработать checkup и вернуться на xbox.com
						try:
							page.wait_for_url(re.compile(r".*xbox\.com/.*/play.*"), timeout=45000)
						except Exception:
							if "xbox.com" not in page.url:
								# Если мы на странице учётной записи, закрываем и возвращаемся к xbox.com/play
								try:
									if "account.microsoft.com" in page.url and handle_optional_account_checkup(page):
										page.goto("https://www.xbox.com/play", wait_until="domcontentloaded")
										page.wait_for_load_state('networkidle', timeout=45000)
										# После возврата проверим домен ещё раз
										if "xbox.com" not in page.url:
											return False
									else:
										return False
								except Exception:
									return False
						return True
					except (BadCredentialsError, CodeRequiredError):
						raise
					except Exception as ie:
						print(f"Direct Microsoft login failed: {ie}")
						return False

			logged_in = try_login_flow(page)
			if not logged_in and headless:
				print("Login failed in headless mode. Retrying with visible browser...")
				try:
					browser.close()
				except Exception:
					pass
				browser, context, page = open_browser(False, proxy)
				logged_in = try_login_flow(page)
				if not logged_in:
					raise PWTimeoutError("Login failed in both headless and headful modes")

			# --- FINAL: Direct navigation to the game page ---
			if not logged_in:
				print("Login was not successful; aborting navigation to game page.")
				return False
			fortnite_url = "https://www.xbox.com/en-US/play/games/fortnite/BT5P2X999VH2"
			launch_url = "https://www.xbox.com/en-US/play/launch/fortnite/BT5P2X999VH2"
			print(f"Credentials accepted. Navigating to Fortnite page: {fortnite_url}")

			# На всякий случай перед переходом закроем необязательное окно checkup, если мы на домене account.microsoft.com
			try:
				if handle_optional_account_checkup(page):
					print("Optional Microsoft account checkup was shown and dismissed before navigating to game.")
			except Exception:
				pass
			try:
				page.goto(fortnite_url, wait_until="domcontentloaded")
			except Exception as nav_err:
				# Навигация может быть прервана авто-редиректами (ERR_ABORTED) — не считаем это критичным
				if "ERR_ABORTED" not in str(nav_err):
					raise
			# Дадим странице стабилизироваться (без жёсткого ожидания networkidle)
			try:
				page.wait_for_load_state('domcontentloaded', timeout=15000)
			except Exception:
				pass
			# Подождём появления корректного URL, но не падаем при таймауте
			try:
				page.wait_for_url(re.compile(r"xbox\.com/.*/fortnite/"), timeout=30000)
			except Exception:
				pass
			take_debug_screenshot(page, "fortnite_direct_page")

			# Try direct navigation to the stream (avoid opening any description at all)
			try:
				page.goto(launch_url, wait_until="domcontentloaded")
				page.wait_for_timeout(1000)
			except Exception:
				pass
			# If we are already on launch or stream canvas is present, skip any Play clicks
			launched_already = False
			try:
				if re.search(r"/play/launch/fortnite", page.url, re.I) or page.locator('canvas').count() > 0:
					launched_already = True
			except Exception:
				pass
 
			# Helpers for page overlays and Play button
			def dismiss_page_bubbles(p):
				"""Закрывает обучающие/всплывающие подсказки (иконки с крестиком)."""
				candidates = [
					'[role="dialog"] button[aria-label*="Close"]',
					'button[aria-label*="Close"]',
					'button[aria-label*="Закрыть"]',
					'button:has-text("Close")',
					'button:has-text("Закрыть")',
				]
				for sel in candidates:
					try:
						loc = p.locator(sel).first
						if loc and loc.is_visible():
							loc.click()
							p.wait_for_timeout(300)
					except Exception:
						continue

			def close_description_modal_if_present(p) -> bool:
				"""Закрывает модальное окно Description, если показано."""
				try:
					dlg = p.locator('[role="dialog"]')
					if dlg.count() > 0 and dlg.first.is_visible():
						btn = dlg.first.locator('button[aria-label*="Close"], button:has-text("Close"), button:has-text("Закрыть")').first
						if btn and btn.is_visible():
							btn.click()
							p.wait_for_timeout(300)
							return True
				except Exception:
					pass
				return False

			def close_optional_dialogs(p):
				print("Closing optional dialogs if any...")
				selectors = [
					'button:has-text("CLOSE")',
					'button:has-text("Close")',
					'button:has-text("Закрыть")',
					'button[aria-label*="Close"]'
				]
				closed = False
				for sel in selectors:
					try:
						btn = p.locator(sel).first
						if btn and btn.is_visible():
							btn.click()
							p.wait_for_timeout(500)
							closed = True
					except Exception:
						continue
				return closed

			def release_mouse_overlay(p):
				"""Снимает захват мыши подсказкой (Esc/F9)."""
				try:
					p.keyboard.press('F9')
				except Exception:
					pass
				try:
					p.keyboard.press('Escape')
				except Exception:
					pass

			def click_return_to_fullscreen_if_present(p) -> bool:
				"""Ищет попап с предупреждением и кнопкой 'Return to full screen' и нажимает её, если видна.
				Возвращает True, если клик выполнен."""
				try:
					selectors = [
						'button:has-text("Return to full screen")',
						'button:has-text("RETURN TO FULL SCREEN")',
						'button:has-text("Вернуться в полноэкранный режим")',
						'[role="dialog"] >> button:has-text("Return to full screen")',
						'[role="dialog"] >> button[data-auto-focus="true"]',
						'button[class*="PopupScreen-module__button"]:has-text("Return to full screen")',
					]
					for sel in selectors:
						try:
							btn = p.locator(sel).first
							if btn and btn.is_visible():
								btn.click()
								p.wait_for_timeout(350)
								take_debug_screenshot(p, 'return_fullscreen_clicked')
								return True
						except Exception:
							continue
				except Exception:
					pass
				return False

			def click_enter_fullscreen_overlay_if_present(p) -> bool:
				"""CV-фолбэк: ищет кнопку ENTER FULLSCREEN на оверлее (без стабильного DOM) по
				шаблону 'assets/enterfullscreen.png' и кликает по центру найденного шаблона."""
				try:
					rois = [
						(0.20, 0.03, 0.80, 0.20),  # верхняя центральная полоса
						(0.10, 0.03, 0.90, 0.24),  # шире по горизонтали
						None,                      # весь кадр как резерв
					]
					scales = [0.70, 0.85, 1.0, 1.15, 1.30]
					for roi in rois:
						pt = vision.find_image_on_page(p, 'assets/enterfullscreen.png', confidence=0.83, timeout=0.7, roi=roi, scales=scales)
						if pt:
							vision.page_click_from_screenshot(p, pt[0], pt[1], steps=8)
							p.wait_for_timeout(300)
							take_debug_screenshot(p, 'enterfullscreen_clicked')
							return True
				except Exception:
					pass
				# RPA fallback (optional via env): RPA_AUTORUN_HTML, RPA_BROWSER_PATH, RPA_DOWNLOAD_DIR, RPA_MACRO
				try:
					import os as _os
					au = _os.environ.get('RPA_AUTORUN_HTML')
					br = _os.environ.get('RPA_BROWSER_PATH')
					dl = _os.environ.get('RPA_DOWNLOAD_DIR')
					mc = _os.environ.get('RPA_MACRO', 'Demo/Core/DemoAutofill')
					if au and br and dl:
						try:
							from . import rpa_runner as _rpa
						except Exception:
							from src import rpa_runner as _rpa  # type: ignore
						ok = False
						try:
							ok = _rpa.run_ui_vision_macro(mc, timeout_seconds=20, path_downloaddir=dl, path_autorun_html=au, browser_path=br)
						except Exception:
							ok = False
						if ok:
							return True
				except Exception:
					pass
				return False

			def click_header_play_button(p, ctx) -> bool:
				"""Выбирает корректную зелёную PLAY вверху: исключает элементы в диалогах, отдаёт приоритет большой ширине и малой координате Y."""
				selectors = [
					'button:has-text("PLAY")',
					'button:has-text("Играть")',
				]
				candidates = []  # (score, element)
				for sel in selectors:
					try:
						locs = p.locator(sel)
						cnt = min(locs.count(), 12)
						for i in range(cnt):
							el = locs.nth(i)
							if not el.is_visible():
								continue
							# Исключаем кнопки внутри диалогов/модалок
							try:
								inside_dialog = el.evaluate("e => !!e.closest('[role=dialog]')")
								if inside_dialog:
									continue
							except Exception:
								pass
							box = el.bounding_box()
							if not box:
								continue
							w = float(box['width'] or 0)
							h = float(box['height'] or 0)
							y = float(box['y'] or 0)
							# Фильтр по размеру и положению: верхняя зелёная кнопка
							if w < 120 or h < 28 or h > 90 or y > 280:
								continue
							# Скоринг: больше ширина и меньше Y → выше приоритет
							score = (w * 2.0) - y
							candidates.append((score, el))
					except Exception:
						continue
				if not candidates:
					return False
				candidates.sort(key=lambda t: t[0], reverse=True)
				btn = candidates[0][1]
				try:
					with ctx.expect_page(timeout=3000) as pinfo:
						btn.click()
					_ = pinfo.value
				except Exception:
					btn.click(force=True)
				# Проверка: ушли ли на launch/появился canvas
				try:
					p.wait_for_timeout(800)
					if re.search(r"/play/launch/fortnite", p.url, re.I) or p.locator('canvas').count() > 0:
						return True
				except Exception:
					pass
				return True
 
			def click_play_with_retries(p, ctx, attempts: int = 3) -> bool:
				texts = r"^\s*(PLAY|Играть)\s*$"
				main_content = p.locator('[role="main"]').first
				for i in range(attempts):
					try:
						# Try button first (exact text)
						btn = main_content.get_by_role('button', name=re.compile(texts, re.I)).first
						btn.wait_for(timeout=20000)
						btn.scroll_into_view_if_needed()
						p.wait_for_timeout(300)
						take_debug_screenshot(p, f"play_button_found_and_waiting_try{i+1}")
						# If click opens new page/popup, adopt it
						new_page = None
						try:
							with ctx.expect_page(timeout=5000) as pinfo:
								btn.click()
							new_page = pinfo.value
						except Exception:
							btn.click(timeout=45000)
						if new_page is not None:
							p = new_page
							# replace outer variable by updating closure reference is not straightforward; return True and let caller swap if needed
						p.wait_for_timeout(3000)
						# если кнопка всё ещё видна, пробуем снова
						if btn.is_visible():
							# На всякий случай закроем возможные подсказки и кликнем с force
							dismiss_page_bubbles(p)
							btn.click(timeout=45000, force=True)
							p.wait_for_timeout(3000)
							if btn.is_visible():
								# Try clicking via coordinates
								try:
									box = btn.bounding_box()
									if box:
										p.mouse.click(box['x'] + box['width']/2, box['y'] + box['height']/2)
										p.wait_for_timeout(1500)
								except Exception:
									pass
								continue
						# Wait for launch URL or canvas
						try:
							p.wait_for_url(re.compile(r"/play/launch/fortnite/", re.I), timeout=20000)
							return True
						except Exception:
							if p.locator('canvas').count() > 0:
								return True
							# else, retry loop
					except Exception:
						# Попробуем альтернативные селекторы
						try:
							alt = p.locator(
								'button:has-text("PLAY"), button:has-text("Играть"), a[href*="/play/launch/fortnite"]'
							).first
							alt.scroll_into_view_if_needed()
							alt.click(timeout=45000)
							p.wait_for_timeout(2000)
							return True
						except Exception:
							pass
					dismiss_page_bubbles(p)
					p.wait_for_timeout(500)
					# CV fallback: green Play on catalog page
					# ОТКЛЮЧЕНО: desktop-CV использует системную мышь и может перехватывать курсор пользователя.
					# try:
					# 	if vision.click_on_image('assets/play_button.png', confidence=0.75, timeout=3):
					# 		p.wait_for_timeout(1500)
					# 		return True
					# except Exception:
					# 	pass
				return False

			def handle_continue_anyway_dom(p):
				"""Ищет и нажимает кнопку 'Continue anyway' в DOM (если не используем ассет)."""
				texts = ["Continue anyway", "Продолжить всё равно", "Продолжить несмотря"]
				for t in texts:
					try:
						loc = p.get_by_role('button', name=re.compile(re.escape(t), re.I)).first
						if loc and loc.is_visible():
							loc.click(timeout=15000)
							return True
					except Exception:
						continue
				return False

			# Click the "Play" button
			if not launched_already:
				print("Looking for the main 'Play' button...")
			# Use a more specific locator to find the button within the main content area
			main_content = page.locator('[role="main"]').first
			# Локализация: поддержим несколько языков для названия кнопки
			play_button = main_content.get_by_role('button', name=re.compile(r'^\s*(PLAY|Играть)\s*$', re.I)).first
			# Перед кликом закроем любые панели/оверлеи, прокрутим к началу
			try:
				close_optional_dialogs(page)
				release_mouse_overlay(page)
				page.keyboard.press('Escape')
				page.evaluate('window.scrollTo(0, 0)')
				close_description_modal_if_present(page)
			except Exception:
				pass
			print("Waiting for Play button to become clickable and clicking it...")
			# The .click() method automatically waits for the element to be actionable (visible, enabled).
			take_debug_screenshot(page, "play_button_found_and_waiting")
			try:
					# Пытаемся кликнуть с быстрым adopt новой страницы (если откроется отдельное окно стрима)
					new_page = None
					try:
						with context.expect_page(timeout=5000) as pinfo:
							play_button.click()
						new_page = pinfo.value
					except Exception:
						# Фолбэк: быстрый клик с force и умеренным таймаутом
						play_button.click(timeout=8000, force=True)
					if new_page is not None:
						page = new_page
			except Exception:
				# Попробуем с ретраями и закрытием всплывающих подсказок
				if not (click_header_play_button(page, context) or click_play_with_retries(page, context)):
					# Глобальный фолбэк: поиск по всей странице
					try:
						page.get_by_role('button', name=re.compile(r'Play|Играть', re.I)).first.click(timeout=20000)
					except Exception:
						pass
						# Не падаем: возможно, уже на launch/stream
			take_debug_screenshot(page, "after_play_button_click")
			# Верификация: если всё ещё на странице каталога, попробуем кликнуть ещё раз
			try:
				if re.search(r"/play/games/fortnite", page.url, re.I):
					print("Still on catalog page after click; retrying Play...")
					click_play_with_retries(page, context)
					page.wait_for_timeout(1500)
					# Если всё ещё каталог — закрыть модалки и перейти напрямую на launch URL
					if re.search(r"/play/games/fortnite", page.url, re.I):
						for _ in range(2):
							close_optional_dialogs(page)
							dismiss_page_bubbles(page)
							page.wait_for_timeout(300)
						try:
							page.goto(launch_url, wait_until="domcontentloaded")
							page.wait_for_timeout(1500)
						except Exception:
							pass
			except Exception:
				pass
			else:
				print("Already on launch/canvas — skipping Play click.")
 
			# Helper: дождаться старта облачного стрима (эвристически)
			def wait_for_stream_connected(p, timeout_sec: int = 120) -> bool:
				deadline = time.time() + timeout_sec
				while time.time() < deadline:
					try:
						# Явный признак запущенного стрима — URL '/play/launch/fortnite'
						if re.search(r"/play/launch/fortnite", p.url, re.I):
							return True
						# Альтернатива — наличие одного или более canvas (рендер облака)
						if p.locator('canvas').count() > 0:
							return True
					except Exception:
						pass
					p.wait_for_timeout(1000)
				return False

			def keep_stream_open(p, minutes: int) -> None:
				seconds = max(1, int(minutes)) * 60
				print(f"Keeping stream open for {minutes} minutes...")
				start = time.time()
				while time.time() - start < seconds:
					# Лёгкая активность, чтобы не отваливалось из‑за простоя
					try:
						p.keyboard.press('Shift')
					except Exception:
						pass
					# Попытаться вернуть фуллскрин при любом возникновении попапов
					try:
						if not click_return_to_fullscreen_if_present(p):
							click_enter_fullscreen_overlay_if_present(p)
					except Exception:
						pass
					p.wait_for_timeout(15000)  # 15s

			# NEW: минимальный пост‑Play поток — просто удерживаем стрим, без авто‑детекта/поиска/YOLO (будет переписано)
			# Удалено раннее удержание и return — продолжим к сценарию In-Game Automation ниже

			def ensure_stream_focus():
				try:
					vp = page.viewport_size or {"width": 1280, "height": 720}
					cx = int(vp["width"]) // 2
					cy = int(vp["height"]) // 2
					page.mouse.move(cx, cy)
					page.mouse.click(cx, cy)
					page.wait_for_timeout(200)
					page.keyboard.press('Space')
				except Exception:
					pass

			def wait_for_lobby_ui(p, timeout_sec: int = 120) -> bool:
				"""Ждём появления лупы (верх‑лево) и стабильности кадра. Дополнительно ждём исчезновения CONNECTING/LOGGING IN и подсказки F9.
				Добавлены: более широкие ROI/масштабы, сниженный порог, DOM‑фолбэк и канвас‑фолбэк без загрузочных оверлеев."""
				ensure_stream_focus()
				deadline = time.time() + max(1, int(timeout_sec))
				# Узкая ROI под реальную позицию лупы + более широкие варианты как резерв, включая верхнюю полосу целиком
				roi_priority = [
					(0.05, 0.08, 0.16, 0.18),  # узко верх‑лево
					(0.00, 0.00, 0.25, 0.22),  # шире верх‑лево
					(0.0, 0.0, 0.5, 0.28),     # верхняя половина слева
					(0.0, 0.0, 1.0, 0.22),     # вся верхняя полоса
				]
				scales = [0.70, 0.85, 1.0, 1.15, 1.30]
				high_conf = 0.86
				stable_needed = 3
				no_overlay_streak_needed = 3
				no_overlay_streak = 0
				while time.time() < deadline:
					# NEW: закрываем предупреждение об выходе из полноэкранного режима, если всплыло
					try:
						if not click_return_to_fullscreen_if_present(p):
							click_enter_fullscreen_overlay_if_present(p)
					except Exception:
						pass
					try:
						# Если видим подсказку про F9 — зажимаем F9 и ждём
						f9_hint = p.locator(r'text=/Press\s*F9\s*to\s*release\s*your\s*mouse/i').first
						if f9_hint and f9_hint.is_visible():
							p.keyboard.press('F9')
							p.wait_for_timeout(250)
					except Exception:
						pass
					# Закрыть боковую панель, если есть, и зафиксировать ввод в поток
					close_xbox_overlay_if_present(p)
					lock_mouse_into_stream(p)
					# Если CV по странице видит CONNECTING/LOGGING или экран с самолётом — рано
					try:
						still_loading = vision.detect_connecting_overlay_on_page(p) or vision.detect_plane_screen_on_page(p)
						if still_loading:
							print("[WAIT] Loading overlays (CONNECTING/PLANE) — waiting...")
							no_overlay_streak = 0
							p.wait_for_timeout(900)
							continue
						else:
							no_overlay_streak = min(no_overlay_streak + 1, no_overlay_streak_needed)
					except Exception:
						pass
					# Требуем несколько подряд циклов без оверлея перед поиском лупы
					if no_overlay_streak < no_overlay_streak_needed:
						p.wait_for_timeout(600)
						continue
					# DOM‑фолбэк: если элементы поиска видны — считаем лобби готовым
					try:
						dom_ready = False
						for sel in [
							'button[aria-label*="Search"]',
							'button[aria-label*="Поиск"]',
							'button:has-text("Search")',
							'button:has-text("Поиск")',
							'input[placeholder*="Search"]',
							'input[aria-label*="Search"]',
						]:
							el = p.locator(sel).first
							if el and el.is_visible():
								dom_ready = True
								break
						if dom_ready:
							print("[WAIT] Lobby ready: DOM search controls visible")
							return True
					except Exception:
						pass
					# Ищем лупу стабильно stable_needed раз подряд через CV
					for roi in roi_priority:
						hits = []
						for _ in range(stable_needed):
							pt = vision.find_image_on_page(p, 'assets/search_icon.png', confidence=high_conf, timeout=1, roi=roi, scales=scales)
							if not pt:
								hits = []
								break
							hits.append(pt)
							p.wait_for_timeout(300)
						if len(hits) == stable_needed:
							xs = [c[0] for c in hits]
							ys = [c[1] for c in hits]
							if max(xs) - min(xs) <= 14 and max(ys) - min(ys) <= 12:
								# финальная проверка: ещё раз убедимся, что загрузочные экраны исчезли
								p.wait_for_timeout(900)
								try:
									if vision.detect_connecting_overlay_on_page(p) or vision.detect_plane_screen_on_page(p):
										no_overlay_streak = 0
										break
								except Exception:
									pass
								print(f"[WAIT] Lobby ready: stable search icon at ~{hits[-1]}")
								return True
					# Канвас‑фолбэк: если поток есть и давно нет оверлеев — принимаем как готовность
					try:
						if p.locator('canvas').count() > 0 and no_overlay_streak >= no_overlay_streak_needed:
							print("[WAIT] Lobby likely ready: canvas present and no loading overlays")
							return True
					except Exception:
						pass
					p.wait_for_timeout(700)
				print("[WAIT] Lobby not ready by timeout")
				return False

			def lock_mouse_into_stream(p):
				"""Фокус и захват мыши внутри канваса: Esc, клик по самому canvas (если найден), затем F9."""
				# Автовозврат в полноэкранный режим, если показано предупреждение
				try:
					if not click_return_to_fullscreen_if_present(p):
						click_enter_fullscreen_overlay_if_present(p)
				except Exception:
					pass
				try:
					p.keyboard.press('Escape')
					p.wait_for_timeout(100)
				except Exception:
					pass
				clicked = False
				# Попробуем кликнуть центр реального canvas по его bounding_box
				try:
					cnv = p.locator('canvas').first
					if cnv and cnv.is_visible():
						box = cnv.bounding_box()
						if box:
							cx = int((box.get('x') or 0) + (box.get('width') or 0) / 2)
							cy = int((box.get('y') or 0) + (box.get('height') or 0) / 2)
							p.mouse.move(cx, cy)
							p.mouse.click(cx, cy)
							clicked = True
				except Exception:
					pass
				# Фолбэк: центр вьюпорта
				if not clicked:
					try:
						vp = p.viewport_size or {"width": 1280, "height": 720}
						cx = int(vp["width"]) // 2
						cy = int(vp["height"]) // 2
						p.mouse.move(cx, cy)
						p.mouse.click(cx, cy)
					except Exception:
						pass

			def acquire_stream_control(p):
				try:
					vp = p.viewport_size or {"width": 1280, "height": 720}
					cx = int(vp["width"]) // 2
					cy = int(vp["height"]) // 2
					p.mouse.move(cx, cy)
					p.mouse.click(cx, cy)
					p.wait_for_timeout(120)
					p.mouse.click(cx, cy)
					p.wait_for_timeout(200)
					# лёгкое движение, чтобы убедиться, что курсор захвачен
					p.mouse.move(cx + 6, cy + 4)
					p.mouse.move(cx - 6, cy - 4)
				except Exception:
					pass

			def open_search_icon_only(p) -> bool:
				try:
					# Сначала снимем возможный оверлей "Click or tap to continue playing"
					try:
						vision.dismiss_tap_to_continue_on_page(p)
					except Exception:
						pass
					# YOLO сначала, если доступны веса модели
					try:
						yolo_weights = os.path.exists('config/yolo/model.pt') and (getattr(vision, '_YOLO', None) is not None)
					except Exception:
						yolo_weights = False
					if not yolo_weights:
						print("[YOLO] Unavailable (no ultralytics or weights). Fallback to CV.")
					if yolo_weights:
						try:
							# Детектируем только в верхней полосе экрана и слева (несколько ROI)
							img_full = vision._capture_page_bgr(p)
							h, w = img_full.shape[:2]
							roi_search = [
								(0.00, 0.00, 0.50, 0.28),
								(0.50, 0.00, 1.00, 0.28),
								(0.25, 0.00, 0.75, 0.36),
							]
							best = None  # (conf, cx_abs, cy_abs)
							for fr in roi_search:
								x0 = max(0, min(w - 1, int(w * fr[0]))); y0 = max(0, min(h - 1, int(h * fr[1])))
								x1 = max(0, min(w,     int(w * fr[2]))); y1 = max(0, min(h,     int(h * fr[3])))
								if x1 <= x0 or y1 <= y0:
									continue
								crop = img_full[y0:y1, x0:x1]
								dets = vision.yolo_detect(crop, conf=0.45)
								for d in dets:
									name = (d.get('name', '') or '').lower()
									if name in ('search', 'magnifier', 'icon_search'):
										x1b, y1b, x2b, y2b = d['xyxy']
										w_box = max(1, int(x2b - x1b)); h_box = max(1, int(y2b - y1b))
										# Отсечём слишком мелкие боксы (шум)
										if w_box < 14 or h_box < 14:
											continue
										cx = int((x1b + x2b) / 2) + x0
										cy = int((y1b + y2b) / 2) + y0
										# Игнорируем бокс, если он слишком близко к краю — часто это шум, который даёт клики (0,0)
										if cx < 8 or cy < 8 or cx > w - 8 or cy > h - 8:
											continue
										conf_v = float(d.get('conf', 0.0) or 0.0)
										if best is None or conf_v > best[0]:
											best = (conf_v, cx, cy)
							if best is not None:
								vp = p.viewport_size or {"width": w, "height": h}
								cx = max(1, min(int(vp["width"]) - 2, int(best[1])))
								cy = max(1, min(int(vp["height"]) - 2, int(best[2])))
								print(f"[YOLO] Click search at ({cx},{cy}) conf={best[0]:.2f}")
								p.mouse.move(cx, cy)
								p.mouse.click(cx, cy)
								p.wait_for_timeout(120)
								p.mouse.click(cx, cy)  # двойной клик для надёжности
								p.wait_for_timeout(220)
								take_debug_screenshot(p, 'clicked_search_icon')
								return True
						except Exception:
							pass
					# CV‑поиск иконки лупы (multi‑scale + ROI)
					roi_search = [
						(0.0, 0.0, 0.5, 0.28),
						(0.5, 0.0, 1.0, 0.28),
						(0.25, 0.0, 0.75, 0.36),
						None,
					]
					scales_common = [0.40, 0.45, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3]
					for roi in roi_search:
						probe = vision.find_image_on_page(p, 'assets/search_icon.png', confidence=0.72, timeout=1, roi=roi, scales=scales_common)
						if probe:
							print(f"[CV] Click search at image coords {probe}")
							vision.page_click_from_screenshot(p, probe[0], probe[1], steps=10)
							p.wait_for_timeout(160)
							# Дополнительный клик подтверждения по изображению (пересчёт координат внутри)
							if vision.click_on_image_on_page(p, 'assets/search_icon.png', confidence=0.72, timeout=2, roi=roi, scales=scales_common):
								take_debug_screenshot(p, 'clicked_search_icon')
								return True
					# Фолбэк: горячая клавиша '/'
					try:
						print("[FALLBACK] Press Slash to open search")
						p.keyboard.press('Slash')
						p.wait_for_timeout(200)
						take_debug_screenshot(p, 'clicked_search_icon')
						return True
					except Exception:
						pass
				except Exception:
					pass
				return False

			def search_and_launch_island_canvas(p, code: str) -> bool:
				"""Открывает поиск, вводит код и запускает карту, используя только события Playwright (с CV-поиском ассетов)."""
				try:
					# Снимем центральный оверлей кликом по центру, если обнаружен
					try:
						vision.dismiss_tap_to_continue_on_page(p)
					except Exception:
						pass
					print("[CANVAS] Start island search")
					# Страховка: дождаться окончания CONNECTING/LOGGING перед любыми действиями
					block_start = time.time()
					while time.time() - block_start < 6:
						try:
							if vision.detect_connecting_overlay_on_page(p):
								p.wait_for_timeout(600)
								continue
							break
						except Exception:
							break
					take_debug_screenshot(p, "canvas_start")
					lock_mouse_into_stream(p)
					# 0) Попробуем открыть поиск горячей клавишей '/' сразу
					try:
						p.keyboard.press('Slash')
						p.wait_for_timeout(200)
						take_debug_screenshot(p, "canvas_try_slash")
					except Exception:
						pass
					# Multi-scale + ROI для иконки лупы
					roi_search = [
						(0.0, 0.0, 0.5, 0.28),
						(0.5, 0.0, 1.0, 0.28),
						(0.25, 0.0, 0.75, 0.36),
						None,
					]
					scales_common = [0.55, 0.7, 0.85, 1.0, 1.15, 1.3]
					icon_path = vision.resolve_asset_path('assets/search_icon.png')
					print(f"[DBG] search_icon asset: {icon_path}")
					opened = False
					for idx, roi in enumerate(roi_search):
						try:
							probe = vision.find_image_on_page(p, 'assets/search_icon.png', confidence=0.72, timeout=1, roi=roi, scales=scales_common)
							print(f"[DBG] ROI#{idx} {roi}: {'FOUND at '+str(probe) if probe else 'not found'}")
							if probe:
								if vision.click_on_image_on_page(p, 'assets/search_icon.png', confidence=0.72, timeout=2, roi=roi, scales=scales_common):
									opened = True
									print("[ACT] Clicked search icon")
									take_debug_screenshot(p, "clicked_search_icon")
									break
						except Exception as e:
							print(f"[DBG] ROI#{idx} probe error: {e}")
					if not opened:
						# Координатный фолбэк на верхнюю панель
						def click_relative(x_frac: float, y_frac: float):
							vp = p.viewport_size or {"width": 1280, "height": 720}
							x = max(0, min(int(vp["width"]) - 1, int(int(vp["width"]) * x_frac)))
							y = max(0, min(int(vp["height"]) - 1, int(int(vp["height"]) * y_frac)))
							p.mouse.move(x, y)
							p.mouse.click(x, y)
						print("[DBG] ROI fallback click near header (0.12, 0.11)")
						click_relative(0.12, 0.11)
						p.wait_for_timeout(300)
						take_debug_screenshot(p, "fallback_header_click")

					# Фокус на поле ввода кода (поиск большой строки)
					roi_input = [
						(0.05, 0.08, 0.95, 0.35),
						None,
					]
					input_path = vision.resolve_asset_path('assets/island_code_input_field.png')
					print(f"[DBG] input_field asset: {input_path}")
					focused = False
					for idx, roi in enumerate(roi_input):
						try:
							probe = vision.find_image_on_page(p, 'assets/island_code_input_field.png', confidence=0.7, timeout=1, roi=roi, scales=scales_common)
							print(f"[DBG] INPUT ROI#{idx} {roi}: {'FOUND at '+str(probe) if probe else 'not found'}")
							if probe:
								if vision.click_on_image_on_page(p, 'assets/island_code_input_field.png', confidence=0.7, timeout=2, roi=roi, scales=scales_common):
									focused = True
									print("[ACT] Focused input field")
									take_debug_screenshot(p, "focused_input")
									break
						except Exception as e:
							print(f"[DBG] INPUT ROI#{idx} probe error: {e}")
					if not focused:
						# Чуть правее/левее кликаем как фолбэк
						p.wait_for_timeout(200)
						p.mouse.click((p.viewport_size or {"width":1280})["width"]//2, int((p.viewport_size or {"height":720})["height"]*0.2))
						p.wait_for_timeout(200)
						take_debug_screenshot(p, "input_fallback_click")

					# Ввод кода
					print(f"[ACT] Typing island code: {code}")
					try:
						p.keyboard.type(code, delay=50)  # ввод через Playwright
					except Exception:
						pass
					p.wait_for_timeout(300)
					take_debug_screenshot(p, "typed_code")

					# Submit кнопка или Enter (multi-ROI)
					roi_submit = [
						(0.20, 0.45, 0.80, 0.90),
					]
					sub_path = vision.resolve_asset_path('assets/submit_button.png')
					print(f"[DBG] submit asset: {sub_path}")
					submitted = False
					for idx, roi in enumerate(roi_submit):
						try:
							probe = vision.find_image_on_page(p, 'assets/submit_button.png', confidence=0.75, timeout=1, roi=roi, scales=scales_common)
							print(f"[DBG] SUBMIT ROI#{idx} {roi}: {'FOUND at '+str(probe) if probe else 'not found'}")
							if probe:
								if vision.click_on_image_on_page(p, 'assets/submit_button.png', confidence=0.75, timeout=2, roi=roi, scales=scales_common):
									submitted = True
									print("[ACT] Clicked SUBMIT")
									take_debug_screenshot(p, "clicked_submit")
									break
						except Exception as e:
							print(f"[DBG] SUBMIT ROI#{idx} probe error: {e}")
					if not submitted:
						try:
							p.keyboard.press('Enter')
						except Exception:
							pass
						print("[ACT] Pressed Enter as submit")
						p.wait_for_timeout(1200)
						take_debug_screenshot(p, "pressed_enter_submit")

					# After submit: если открылся список результатов — кликаем первую карту → SELECT → PLAY
					try:
						p.wait_for_timeout(800)
						first_card = p.locator('section:has-text("For You"), section:has-text("Results"), div:has(a[href*="/play/launch/"])').locator('a[href*="/play/launch/"]').first
						if first_card and first_card.is_visible():
							first_card.click()
							p.wait_for_timeout(800)
							take_debug_screenshot(p, "dom_first_result_clicked")
							sel = p.locator('button:has-text("SELECT"), button:has-text("Выбрать")').first
							if sel and sel.is_visible():
								sel.click()
								p.wait_for_timeout(800)
								take_debug_screenshot(p, "dom_select_clicked")
						print("[DOM] First result selected via DOM")
					except Exception:
						pass

					# Ожидание появления PLAY и клик по нему (только внутри окна браузера)
					btn = vision.find_image_on_page(p, 'assets/play_button_yellow.png', confidence=0.70, timeout=10, roi=None, scales=[0.75, 0.9, 1.0, 1.2])
					if btn:
						vision.click_on_image_on_page(p, 'assets/play_button_yellow.png', confidence=0.70, timeout=3, scales=[0.75, 0.9, 1.0, 1.2])
						print("[ACT] Clicked PLAY (yellow)")
						take_debug_screenshot(p, "clicked_play_yellow")
					else:
						btn2 = vision.find_image_on_page(p, 'assets/play_button.png', confidence=0.75, timeout=10, roi=None, scales=[0.75, 0.9, 1.0, 1.2])
						if btn2:
							vision.click_on_image_on_page(p, 'assets/play_button.png', confidence=0.75, timeout=3, scales=[0.75, 0.9, 1.0, 1.2])
							print("[ACT] Clicked PLAY (green)")
							take_debug_screenshot(p, "clicked_play_green")
						else:
							try:
								p.keyboard.press('Enter')
							except Exception:
								pass
							print("[ACT] Pressed Enter as PLAY fallback")
					p.wait_for_timeout(1000)
					take_debug_screenshot(p, "after_play")
					return True
				except Exception as e:
					print(f"[CV] Flow failed with error: {e}")
					return False

			def search_and_launch_island_unified(p, code: str) -> bool:
				"""Сначала пробуем DOM-поиск (лупа → поле → submit → SELECT → PLAY), если не удалось — канвас-метод."""
				print("[DOM] Try island search via page DOM")
				# Страховка: не начинать поиск, пока висит CONNECTING/LOGGING
				start_block = time.time()
				while time.time() - start_block < 5:
					try:
						if vision.detect_connecting_overlay_on_page(p):
							p.wait_for_timeout(600)
							continue
						break
					except Exception:
						break
				try:
					# Открыть поиск по DOM
					trigger = p.locator('button[aria-label*="Search"], button:has-text("Search"), [role="button"]:has([data-icon="search"])').first
					if trigger and trigger.is_visible():
						trigger.click()
						p.wait_for_timeout(300)
						take_debug_screenshot(p, "dom_opened_search")
					# Ввести код в поле с плейсхолдером
					inp = p.get_by_placeholder(re.compile(r"Search.*Islands|Search.*Creators|Search", re.I)).first
					if inp and inp.is_visible():
						inp.fill("")
						inp.type(code, delay=50)
						take_debug_screenshot(p, "dom_typed_code")
						# Нажать Submit, если есть, иначе Enter
						sub = p.locator('button:has-text("SUBMIT")').first
						if sub and sub.is_visible():
							sub.click()
							p.wait_for_timeout(300)
							take_debug_screenshot(p, "dom_clicked_submit")
						else:
							inp.press('Enter')
							p.wait_for_timeout(300)
							take_debug_screenshot(p, "dom_pressed_enter_submit")
						p.wait_for_timeout(800)
						# Перейти на страницу первой карты или сразу нажать SELECT
						first_card = p.locator('a[href*="/play/launch/"]').first
						if first_card and first_card.is_visible():
							first_card.click()
							p.wait_for_timeout(800)
							take_debug_screenshot(p, "dom_clicked_first_card")
						# SELECT на странице карты
						sel = p.locator('button:has-text("SELECT"), button:has-text("Выбрать")').first
						if sel and sel.is_visible():
							sel.click()
							p.wait_for_timeout(800)
							take_debug_screenshot(p, "dom_clicked_select")
						# PLAY в лобби
						play = p.locator('button:has-text("PLAY"), button:has-text("Играть")').first
						if play and play.is_visible():
							play.click()
							p.wait_for_timeout(1000)
							take_debug_screenshot(p, "dom_clicked_play")
						print("[DOM] Island selected and Play pressed")
						return True
					print("[DOM] Input not visible — fallback to canvas")
				except Exception as e:
					print(f"[DOM] Failed with {e}, fallback to canvas")
				# Канвас‑резерв
				return search_and_launch_island_canvas(p, code)

			def do_active_ingame_actions(p, minutes: int):
				duration = max(1, int(minutes)) * 60
				start = time.time()
				vp = p.viewport_size or {"width": 1280, "height": 720}
				cx = int(vp["width"]) // 2
				cy = int(vp["height"]) // 2
				def heuristic_action_step():
					try:
						frame = vision.capture_obs_frame()
						has_target = vision.detect_enemy_health_bar(frame)
						center_b = vision.center_brightness(frame)
						# Коррекция ориентации, если смотрим в пол/небо
						if center_b > 210 or center_b < 40:
							p.mouse.move(cx, cy)
							p.mouse.down(button='right')
							p.mouse.move(cx, cy + (-140 if center_b > 210 else 140), steps=10)
							p.mouse.up(button='right')
						# Тактика
						if has_target:
							# Прицелиться и атаковать
							p.mouse.move(cx, cy)
							p.mouse.down(button='right')
							p.mouse.move(cx + random.randint(-160, 160), cy, steps=10)
							p.mouse.up(button='right')
							p.mouse.down(button='left')
							p.wait_for_timeout(120)
							p.mouse.up(button='left')
							p.keyboard.down('w')
							p.wait_for_timeout(300)
							p.keyboard.up('w')
							if random.random() < 0.4:
								p.keyboard.press('1')
						else:
							# Поиск цели: TAB, разворот, шаг в сторону
							if random.random() < 0.5:
								p.keyboard.press('tab')
							p.mouse.move(cx, cy)
							p.mouse.down(button='right')
							p.mouse.move(cx + random.choice([-220, 220]), cy, steps=10)
							p.mouse.up(button='right')
							side = random.choice(['a', 'd'])
							p.keyboard.down(side)
							p.wait_for_timeout(250)
							p.keyboard.up(side)
					except Exception:
						pass
				while time.time() - start < duration:
					heuristic_action_step()
					p.wait_for_timeout(random.randint(700, 1400))

			# --- Helpers for island navigation via DOM ---
			def skip_trailer_if_present(p):
				print("Attempting to skip trailer if present...")
				candidates = [
					'button:has-text("SKIP")',
					'button:has-text("Skip")',
					'button:has-text("Пропустить")',
					'[role="dialog"] button:has-text("Skip")',
					'[role="dialog"] button:has-text("Пропустить")',
				]
				for sel in candidates:
					try:
						btn = p.locator(sel).first
						if btn and btn.is_visible():
							btn.click(timeout=10000)
							p.wait_for_timeout(1000)
							return True
					except Exception:
						continue
				# как фолбэк — Esc
				try:
					p.keyboard.press('Escape')
				except Exception:
					pass
				return False

			def close_optional_dialogs(p):
				print("Closing optional dialogs if any...")
				selectors = [
					'button:has-text("CLOSE")',
					'button:has-text("Close")',
					'button:has-text("Закрыть")',
					'button[aria-label*="Close"]'
				]
				closed = False
				for sel in selectors:
					try:
						btn = p.locator(sel).first
						if btn and btn.is_visible():
							btn.click()
							p.wait_for_timeout(500)
							closed = True
					except Exception:
						continue
				return closed

			def release_mouse_overlay(p):
				"""Снимает захват мыши подсказкой (Esc/F9)."""
				try:
					p.keyboard.press('F9')
				except Exception:
					pass
				try:
					p.keyboard.press('Escape')
				except Exception:
					pass

			def search_and_open_island(p, code: str) -> bool:
				print(f"Opening search and entering island code: {code}")
				# Открыть поиск
				opened = False
				for sel in [
					'button[aria-label*="Search"]',
					'button:has-text("Search")',
					'input[placeholder*="Search for Islands or Creators"]',
					'input[placeholder*="Search for Islands"]',
					'input[placeholder*="Search for"]',
					'input[type="search"]',
					'input[type="text"]',
				]:
					try:
						el = p.locator(sel).first
						if el and el.is_visible():
							el.click()
							opened = True
							break
					except Exception:
						continue
				p.wait_for_timeout(500)
				# 1) Попробуем по placeholder напрямую (наиболее надёжно)
				try:
					inp = p.get_by_placeholder(re.compile(r"Search.*Islands|Search.*Creators|Search", re.I)).first
					if inp and inp.is_visible():
						inp.click()
						inp.fill("")
						inp.type(code, delay=50)
						p.keyboard.press('Enter')
						p.wait_for_timeout(1000)
						return True
				except Exception:
					pass
				# 2) Сканируем все текстовые поля и берём крупное в верхней части страницы
				try:
					candidates = p.locator('input, textarea, [role="textbox"]')
					best_idx = -1
					best_w = 0
					count = candidates.count()
					for i in range(min(count, 20)):
						el = candidates.nth(i)
						try:
							box = el.bounding_box()
							if box and box['y'] < 400 and box['width'] > best_w and el.is_visible():
								best_w = box['width']
								best_idx = i
						except Exception:
							continue
					if best_idx >= 0:
						inp = candidates.nth(best_idx)
						inp.click()
						inp.fill("")
						p.keyboard.type(code, delay=50)
						p.keyboard.press('Enter')
						p.wait_for_timeout(1000)
						return True
				except Exception:
					pass
				# 3) Геометрия: клик слева от вкладки PLAY, где иконка лупы
				try:
					play_tab = p.get_by_text(re.compile(r'^PLAY$', re.I)).first
					box = play_tab.bounding_box()
					if box:
						p.mouse.click(box['x'] - 30, box['y'] + box['height'] / 2)
						p.wait_for_timeout(300)
						p.keyboard.type(code, delay=50)
						p.keyboard.press('Enter')
						p.wait_for_timeout(1000)
						return True
				except Exception:
					pass
				# 4) Горячая клавиша как последний шанс
				try:
					p.keyboard.press('/')
					p.wait_for_timeout(200)
					p.keyboard.type(code, delay=50)
					p.keyboard.press('Enter')
					return True
				except Exception:
					pass
				# Если всё неудачно — вернуть False
				print("Search input not found via DOM strategies")
				return False

			def select_map_and_press_play(p) -> bool:
				print("Selecting map and pressing PLAY...")
				# SELECT на странице карты
				for sel in ['button:has-text("SELECT")', 'button:has-text("Выбрать")', 'button:has-text("Select")']:
					try:
						b = p.locator(sel).first
						if b and b.is_visible():
							b.click(timeout=15000)
							p.wait_for_timeout(1500)
							break
					except Exception:
						continue
				# Кнопка PLAY в лобби после выбора
				for sel in ['button:has-text("PLAY")', 'button:has-text("Играть")', 'button:has-text("Play")']:
					try:
						btn = p.locator(sel).first
						btn.scroll_into_view_if_needed()
						btn.click(timeout=20000)
						return True
					except Exception:
						continue
				# CV-фолбэк: зелёная/жёлтая кнопка PLAY
				# ОТКЛЮЧЕНО: desktop-CV использует системную мышь и может перехватывать курсор пользователя.
				# try:
				# 	if vision.click_on_image('assets/play_button_yellow.png', confidence=0.7, timeout=4):
				# 		return True
				# 	if vision.click_on_image('assets/play_button.png', confidence=0.75, timeout=4):
				# 		return True
				# except Exception:
				# 	pass

			def search_and_launch_island_yolo(p, code: str) -> bool:
				"""YOLO-путь: детект иконки поиска/поля ввода/кнопки отправки/PLAY по скриншотам страницы.
				Требует наличия весов в config/yolo/model.pt и настроенных классов: 'search', 'island_input', 'submit', 'play'."""
				try:
					# Тестовая детекция лупы → клик
					img = vision._capture_page_bgr(p)
					detections = vision.yolo_detect(img, conf=0.35)
					def center(box):
						(x1, y1, x2, y2) = box
						return int((x1+x2)/2), int((y1+y2)/2)
					# 1) Открыть поиск
					for d in detections:
						if d.get('name', '').lower() in ('search', 'magnifier', 'icon_search'):
							cx, cy = center(d['xyxy'])
							p.mouse.click(cx, cy)
							p.wait_for_timeout(200)
							break
					if not detections:
						# Нет детекций — вероятно, нет весов. Дадим шанс фолбэкам выше.
						return False
					# 2) Фокус на поле ввода
					img = vision._capture_page_bgr(p)
					detections = vision.yolo_detect(img, conf=0.35)
					focused = False
					for d in detections:
						if d.get('name', '').lower() in ('island_input', 'search_input', 'input'):
							cx, cy = center(d['xyxy'])
							p.mouse.click(cx, cy)
							focused = True
							break
					if not focused:
						# fallback: клик в верхней трети по центру
						vp = p.viewport_size or {"width": 1280, "height": 720}
						p.mouse.click(int(vp['width']*0.5), int(vp['height']*0.18))
					p.wait_for_timeout(200)
					p.keyboard.type(code, delay=50)
					p.wait_for_timeout(200)
					# 3) Submit
					img = vision._capture_page_bgr(p)
					detections = vision.yolo_detect(img, conf=0.35)
					clicked_submit = False
					for d in detections:
						if d.get('name', '').lower() in ('submit', 'apply', 'go'):
							cx, cy = center(d['xyxy'])
							p.mouse.click(cx, cy)
							clicked_submit = True
							break
					if not clicked_submit:
						p.keyboard.press('Enter')
					p.wait_for_timeout(600)
					# 4) PLAY
					img = vision._capture_page_bgr(p)
					detections = vision.yolo_detect(img, conf=0.35)
					for d in detections:
						if d.get('name', '').lower() in ('play', 'play_button'):
							cx, cy = center(d['xyxy'])
							p.mouse.click(cx, cy)
							p.wait_for_timeout(800)
					return True
					# Если PLAY не распознали — пусть сработает fallback (DOM/канвас)
					return False
				except Exception as e:
					print(f"[YOLO] Failed with error: {e}")
					return False

			def screen_open_search_and_enter_code(p, code: str) -> bool:
				"""Открыть поиск и ввести код, приоритет: фокус канваса → ассет лупы внутри канваса → хоткей."""
				try:
					# Фокус на канвасе
					stream_input.ensure_stream_focus(p)
					print("[SCREEN] Step 1: Click search icon (page CV)")
					if not stream_input.open_search(p):
						print("[FALLBACK] Press Slash to open search")
						try:
							p.keyboard.press('/')
							p.wait_for_timeout(120)
						except Exception:
							pass
						ipt = vision.find_image_on_page(
							p,
							'assets/island_code_input_field.png',
							confidence=0.66,
							timeout=3,
							roi=(0.05, 0.08, 0.95, 0.40),
							scales=[0.6, 0.8, 1.0, 1.2],
						)
						if not ipt:
							print("[SCREEN] Error: search panel did not open after click")
							return False

					print("[SCREEN] Step 2: Focus input and type code")
					try:
						p.mouse.move(ipt[0], ipt[1])
						p.mouse.click(ipt[0], ipt[1])
						p.keyboard.type(code, delay=50)
						p.keyboard.press('Enter')
					except Exception:
						return False

					print("[SCREEN] Step 3: Submit")
					if not vision.click_on_image_on_page(p, 'assets/submit_button.png', confidence=0.75, timeout=3, roi=(0.18, 0.40, 0.82, 0.92), scales=[0.7, 1.0, 1.2]):
						try:
							p.keyboard.press('Enter')
						except Exception:
							pass

					print("[SCREEN] Step 4: Wait for PLAY")
					btn = vision.find_image_on_page(p, 'assets/play_button_yellow.png', confidence=0.70, timeout=10, roi=None, scales=[0.75, 0.9, 1.0, 1.2])
					if btn:
						try:
							p.mouse.click(btn[0], btn[1])
							return True
						except Exception:
							pass
					btn2 = vision.find_image_on_page(p, 'assets/play_button.png', confidence=0.75, timeout=10, roi=None, scales=[0.75, 0.9, 1.0, 1.2])
					if btn2:
						try:
							p.mouse.click(btn2[0], btn2[1])
							return True
						except Exception:
							pass
					try:
						p.keyboard.press('Enter')
						print("[SCREEN] Fallback: Enter for PLAY")
						return True
					except Exception:
						return False
				except Exception as e:
					print(f"[SCREEN] Unexpected error: {e}")
					return False

			# ... existing code ...
			# --- Ручной контроль: не переходить к шагу 12 без вашей команды ---
			if manual_lobby_event is not None:
				print("[WAIT] Жду команду 'Лобби готово' перед шагом 12...")
				try:
					# Ждём до 10 минут, чтобы не висеть бесконечно
					manual_lobby_event.wait(timeout=600)
					if manual_lobby_event.is_set():
						print("[WAIT] Команда получена — продолжаю к шагу 12.")
						try:
							manual_lobby_event.clear()
						except Exception:
							pass
					else:
						print("[WAIT] Таймаут ожидания команды — продолжаю по таймауту.")
				except Exception:
					pass

			if not screen_open_search_and_enter_code(page, island_code):
				# Используем только геймпад‑режим из stream_input (без переимпорта)
				stream_input.open_search(page)

		except BadCredentialsError as e:
			print(f"Login failed for account {account_login}: {e}")
			raise
		except CodeRequiredError as e:
			print(f"Login requires code for account {account_login}: {e}")
			return False
		except Exception as e:
			print(f"An error occurred for account {account_login}: {e}")
			print("The bot will now close.")
			result_success = False
		finally:
			print(f"Closing browser for account: {account_login}")
			# Удалим из реестра, чтобы следующий запуск открыл новый
			try:
				with _ACTIVE_BROWSERS_LOCK:
					cur = _ACTIVE_BROWSERS.get(account_login)
					if cur is browser:
						_ACTIVE_BROWSERS.pop(account_login, None)
			except Exception:
				pass
			browser.close()
			return result_success

def close_all_active_browsers():
	"""Безопасно закрывает все активные браузеры из реестра _ACTIVE_BROWSERS."""
	items = []
	try:
		with _ACTIVE_BROWSERS_LOCK:
			items = list(_ACTIVE_BROWSERS.items())
			_ACTIVE_BROWSERS.clear()
	except Exception:
		items = []
	for login, browser in items:
		try:
			browser.close()
		except Exception:
			pass

def main():
	"""
	Main entry point for the application.
	Loads accounts and runs the bot for each one.
	"""
	accounts = load_accounts()
	if not accounts:
		print("No accounts found. Exiting.")
		return

	settings = load_settings()
	# --- Load Island Code ---
	island_code = settings.get('island_code') or load_island_code()
	headless = settings.get('headless', True)

	# Запрет системного ввода (мышь/клава ОС) — используем только виртуальный ввод страницы
	try:
		vision.set_disable_os_input(True)
	except Exception:
		pass

	for account in accounts:
		run_bot(account, island_code, headless=headless)

if __name__ == "__main__":
	main() 