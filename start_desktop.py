import os
import sys
import shutil
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DESKTOP_DIR = os.path.join(ROOT, 'desktop')


def run(cmd, cwd=None, env=None):
    print('> ' + ' '.join(cmd))
    res = subprocess.run(cmd, cwd=cwd, shell=False, env=env)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def which_npm():
    return shutil.which('npm') or shutil.which('npm.cmd')


def which_node():
    return shutil.which('node')

 
def main():
    # Проверки Node/NPM
    node_exe = which_node()
    if not node_exe:
        print('Ошибка: требуется установленный Node.js (node в PATH).')
        print('Скачайте с https://nodejs.org и перезапустите.')
        sys.exit(1)

    # Установка зависимостей UI (однократно)
    node_modules = os.path.join(DESKTOP_DIR, 'node_modules')
    npm_exe = which_npm()
    if not os.path.exists(node_modules):
        if npm_exe:
            run([npm_exe, 'install'], cwd=DESKTOP_DIR)
        else:
            print('Внимание: npm не найден в PATH. Установите зависимости вручную:')
            print(f'  cd {DESKTOP_DIR} && npm install')
            sys.exit(1)

    # Сборка renderer (Vite → desktop/dist)
    vite_cli = os.path.join(DESKTOP_DIR, 'node_modules', 'vite', 'bin', 'vite.js')
    if os.path.exists(vite_cli):
        run([node_exe, vite_cli, 'build', '--config', os.path.join(DESKTOP_DIR, 'vite.config.js')], cwd=DESKTOP_DIR)
    else:
        if not npm_exe:
            print('Не найден vite CLI и npm. Переустановите deps: npm install')
            sys.exit(1)
        run([npm_exe, 'run', 'build:renderer'], cwd=DESKTOP_DIR)

    # Запуск Electron c prod билдом (без Vite dev-сервера)
    env = os.environ.copy()
    env['UI_DEV'] = '0'
    env['PYTHON_EXE'] = sys.executable  # передаём путь к текущему Python для ipc_server

    electron_cli = os.path.join(DESKTOP_DIR, 'node_modules', 'electron', 'cli.js')
    if not os.path.exists(electron_cli):
        if npm_exe:
            run([npm_exe, 'install'], cwd=DESKTOP_DIR)
        else:
            print('Не найден electron CLI. Выполните: npm install в папке desktop')
            sys.exit(1)

    cmd = [node_exe, electron_cli, os.path.join('electron', 'main.js')]
    print('Запускаю Electron...')
    p = subprocess.Popen(cmd, cwd=DESKTOP_DIR, env=env)
    try:
        p.wait()
    except KeyboardInterrupt:
        try:
            p.terminate()
        except Exception:
            pass


if __name__ == '__main__':
    main() 