import os
import datetime
import subprocess
import time
from typing import Optional


def run_ui_vision_macro(
	macro: str,
	timeout_seconds: int,
	path_downloaddir: str,
	path_autorun_html: str,
	browser_path: str,
	var1: str = '-',
	var2: str = '-',
	var3: str = '-',
) -> bool:
	"""Запускает Ui.Vision макрос через autorun.html и ждёт завершения по лог-файлу.
	Возвращает True при Status=OK, иначе False.
	"""
	if not (os.path.exists(path_downloaddir) and os.path.exists(path_autorun_html) and os.path.exists(browser_path)):
		return False

	log_name = 'log_' + datetime.datetime.now().strftime('%m-%d-%Y_%H_%M_%S') + '.txt'
	path_log = os.path.join(path_downloaddir, log_name)

	args_url = (
		'file:///'
		+ path_autorun_html
		+ '?macro='
		+ macro
		+ '&cmd_var1='
		+ var1
		+ '&cmd_var2='
		+ var2
		+ '&cmd_var3='
		+ var3
		+ '&closeRPA=0&direct=1&savelog='
		+ log_name
	)

	try:
		proc = subprocess.Popen([browser_path, args_url])
	except Exception:
		return False

	status_runtime = 0
	while (not os.path.exists(path_log)) and status_runtime < timeout_seconds:
		time.sleep(1)
		status_runtime += 1

	if status_runtime < timeout_seconds:
		try:
			with open(path_log, 'r', encoding='utf-8', errors='ignore') as f:
				first_line = f.readline()
			success = ('Status=OK' in first_line)
			return success
		except Exception:
			return False
	else:
		try:
			proc.kill()
		except Exception:
			pass
		return False 