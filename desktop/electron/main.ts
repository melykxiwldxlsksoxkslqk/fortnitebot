/**
 * Electron Main Process — запуск Python IPC-сервера и управление окном.
 *
 * FIX: установлен Content-Security-Policy через session.defaultSession,
 *      чтобы убрать Electron Security Warning (Insecure CSP).
 */

import { app, BrowserWindow, ipcMain, session } from 'electron';
import { spawn, ChildProcess } from 'child_process';
import * as path from 'path';
import * as readline from 'readline';

let mainWindow: BrowserWindow | null = null;
let pythonProcess: ChildProcess | null = null;
let requestId = 0;
const pendingRequests = new Map<number, { resolve: Function; reject: Function }>();

// ============================================================================
// PYTHON IPC МОСТ
// ============================================================================

function startPythonIPC(): void {
  const projectRoot = path.resolve(__dirname, '..', '..');
  const venvPython = path.join(projectRoot, '..', '.venv', 'Scripts', 'python.exe');

  pythonProcess = spawn(venvPython, [path.join(projectRoot, 'ipc_entry.py')], {
    cwd: projectRoot,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      PYTHONIOENCODING: 'utf-8',
    },
  });

  if (!pythonProcess.stdout || !pythonProcess.stdin) {
    console.error('Не удалось запустить Python IPC процесс');
    return;
  }

  // Чтение stdout построчно
  const rl = readline.createInterface({ input: pythonProcess.stdout });

  rl.on('line', (line: string) => {
    try {
      const msg = JSON.parse(line);

      // JSON-RPC ответ (есть id)
      if (msg.id !== undefined && msg.id !== null) {
        const pending = pendingRequests.get(msg.id);
        if (pending) {
          pendingRequests.delete(msg.id);
          if (msg.error) {
            pending.reject(msg.error);
          } else {
            pending.resolve(msg.result);
          }
        }
        // Пересылаем ответ в renderer
        mainWindow?.webContents.send('ipc:response', msg);
      }
      // JSON-RPC нотификация (нет id) — событие
      else if (msg.method) {
        mainWindow?.webContents.send('ipc:event', msg);
      }
    } catch (e) {
      console.error('Ошибка парсинга IPC-сообщения:', line);
    }
  });

  // Логируем stderr
  pythonProcess.stderr?.on('data', (data: Buffer) => {
    console.error('[Python]', data.toString());
  });

  pythonProcess.on('exit', (code) => {
    console.log(`Python IPC завершился с кодом ${code}`);
    pythonProcess = null;
  });
}

function sendToPython(method: string, params?: any): Promise<any> {
  return new Promise((resolve, reject) => {
    if (!pythonProcess?.stdin) {
      reject(new Error('Python IPC не запущен'));
      return;
    }

    const id = ++requestId;
    const request = {
      jsonrpc: '2.0',
      id,
      method,
      params: params || {},
    };

    pendingRequests.set(id, { resolve, reject });

    pythonProcess.stdin.write(JSON.stringify(request) + '\n');

    // Таймаут 30 секунд
    setTimeout(() => {
      if (pendingRequests.has(id)) {
        pendingRequests.delete(id);
        reject(new Error(`IPC таймаут: ${method}`));
      }
    }, 30000);
  });
}

// ============================================================================
// IPC ОБРАБОТЧИКИ (Renderer → Main → Python)
// ============================================================================

ipcMain.handle('ipc:call', async (_event, method: string, params?: any) => {
  try {
    return await sendToPython(method, params);
  } catch (err: any) {
    throw new Error(err.message || String(err));
  }
});

ipcMain.handle('ipc:ping', async () => {
  try {
    return await sendToPython('ping');
  } catch {
    return null;
  }
});

// ============================================================================
// ОКНО
// ============================================================================

function createWindow(): void {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 800,
    minWidth: 1024,
    minHeight: 600,
    title: 'EpicBot',
    icon: path.join(__dirname, '..', '..', 'assets', 'icon.png'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
    backgroundColor: '#0f172a',
    show: false,
  });

  // Обработка ошибки загрузки
  mainWindow.webContents.on('did-fail-load', (_event, code, desc) => {
    console.error(`Ошибка загрузки страницы: ${code} ${desc}`);
    if (code !== -3) {
      setTimeout(() => {
        mainWindow?.loadURL('http://localhost:5173');
      }, 2000);
    }
  });

  mainWindow.webContents.on('render-process-gone', (_event, details) => {
    console.error('Render-процесс завершился:', details.reason);
  });

  // Dev или production
  const isDev =
    process.env.NODE_ENV === 'development' || process.argv.includes('--dev');

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173').catch((err) => {
      console.error('Не удалось загрузить dev URL:', err);
    });
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ============================================================================
// ЖИЗНЕННЫЙ ЦИКЛ ПРИЛОЖЕНИЯ
// ============================================================================

app.whenReady().then(() => {
  // FIX: Content-Security-Policy — убираем Electron CSP-предупреждение
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self' ws://localhost:* http://localhost:*; img-src 'self' data:",
        ],
      },
    });
  });

  startPythonIPC();
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  // Graceful shutdown: отправляем stop_all в Python, потом убиваем
  if (pythonProcess?.stdin && !pythonProcess.killed) {
    try {
      const stopReq = JSON.stringify({
        jsonrpc: '2.0',
        id: 999999,
        method: 'stop_all',
      });
      pythonProcess.stdin.write(stopReq + '\n');
      pythonProcess.stdin.end();
    } catch (e) {
      console.error('Ошибка отправки stop в Python:', e);
    }
  }

  setTimeout(() => {
    if (pythonProcess && !pythonProcess.killed) {
      pythonProcess.kill();
    }
    app.quit();
  }, 2000);
});
