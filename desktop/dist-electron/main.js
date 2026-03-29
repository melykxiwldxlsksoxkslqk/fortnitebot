"use strict";
/**
 * Electron Main Process — запуск Python IPC-сервера и управление окном.
 *
 * FIX: установлен Content-Security-Policy через session.defaultSession,
 *      чтобы убрать Electron Security Warning (Insecure CSP).
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
const child_process_1 = require("child_process");
const path = __importStar(require("path"));
const readline = __importStar(require("readline"));
let mainWindow = null;
let pythonProcess = null;
let requestId = 0;
const pendingRequests = new Map();
// ============================================================================
// PYTHON IPC МОСТ
// ============================================================================
function startPythonIPC() {
    const projectRoot = path.resolve(__dirname, '..', '..');
    const venvPython = path.join(projectRoot, '..', '.venv', 'Scripts', 'python.exe');
    pythonProcess = (0, child_process_1.spawn)(venvPython, [path.join(projectRoot, 'ipc_entry.py')], {
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
    rl.on('line', (line) => {
        try {
            const msg = JSON.parse(line);
            // JSON-RPC ответ (есть id)
            if (msg.id !== undefined && msg.id !== null) {
                const pending = pendingRequests.get(msg.id);
                if (pending) {
                    pendingRequests.delete(msg.id);
                    if (msg.error) {
                        pending.reject(msg.error);
                    }
                    else {
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
        }
        catch (e) {
            console.error('Ошибка парсинга IPC-сообщения:', line);
        }
    });
    // Логируем stderr
    pythonProcess.stderr?.on('data', (data) => {
        console.error('[Python]', data.toString());
    });
    pythonProcess.on('exit', (code) => {
        console.log(`Python IPC завершился с кодом ${code}`);
        pythonProcess = null;
    });
}
function sendToPython(method, params) {
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
electron_1.ipcMain.handle('ipc:call', async (_event, method, params) => {
    try {
        return await sendToPython(method, params);
    }
    catch (err) {
        throw new Error(err.message || String(err));
    }
});
electron_1.ipcMain.handle('ipc:ping', async () => {
    try {
        return await sendToPython('ping');
    }
    catch {
        return null;
    }
});
// ============================================================================
// ОКНО
// ============================================================================
function createWindow() {
    mainWindow = new electron_1.BrowserWindow({
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
    const isDev = process.env.NODE_ENV === 'development' || process.argv.includes('--dev');
    if (isDev) {
        mainWindow.loadURL('http://localhost:5173').catch((err) => {
            console.error('Не удалось загрузить dev URL:', err);
        });
        mainWindow.webContents.openDevTools();
    }
    else {
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
electron_1.app.whenReady().then(() => {
    // FIX: Content-Security-Policy — убираем Electron CSP-предупреждение
    electron_1.session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
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
    electron_1.app.on('activate', () => {
        if (electron_1.BrowserWindow.getAllWindows().length === 0) {
            createWindow();
        }
    });
});
electron_1.app.on('window-all-closed', () => {
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
        }
        catch (e) {
            console.error('Ошибка отправки stop в Python:', e);
        }
    }
    setTimeout(() => {
        if (pythonProcess && !pythonProcess.killed) {
            pythonProcess.kill();
        }
        electron_1.app.quit();
    }, 2000);
});
//# sourceMappingURL=main.js.map