const { app, BrowserWindow, ipcMain, session } = require('electron')
const path = require('path')
const { spawn } = require('child_process')
const fs = require('fs')

// Используем контролируемую папку профиля, чтобы избежать проблем прав в Roaming
const userDataPath = path.resolve(__dirname, '..', '..', '.electron-user-data')
try { fs.mkdirSync(userDataPath, { recursive: true }) } catch {}
try { app.setPath('userData', userDataPath) } catch {}
try { app.setAppLogsPath(path.join(userDataPath, 'logs')) } catch {}

// Отключаем GPU/disk кэши, которые вызывают ошибки в некоторых окружениях
app.commandLine.appendSwitch('disable-gpu-shader-disk-cache')
app.commandLine.appendSwitch('disable-gpu-program-cache')
app.commandLine.appendSwitch('disk-cache-size', '0')
app.commandLine.appendSwitch('media-cache-size', '0')
// Полностью отключим аппаратное ускорение, чтобы избежать серых экранов на некоторых GPU
app.disableHardwareAcceleration()

let py = null
let win = null
let reqId = 0
const pending = new Map()
let logBuffer = ''

function startPython() {
  const exe = process.env.PYTHON_EXE || 'python'
  const projectRoot = path.resolve(__dirname, '..', '..')
  // Запускаем как модуль, чтобы пакет src.* корректно импортировался
  const env = { ...process.env, PYTHONUNBUFFERED: '1' }
  py = spawn(exe, ['-u', '-m', 'src.ipc_server'], { stdio: ['pipe', 'pipe', 'pipe'], cwd: projectRoot, env })
  py.stdout.setEncoding('utf8')
  py.stderr.setEncoding('utf8')
  py.stdout.on('data', (chunk) => {
    const lines = chunk.toString().split(/\r?\n/).filter(Boolean)
    for (const line of lines) {
      try {
        const msg = JSON.parse(line)
        if (msg.event === 'status') {
          const lineText = `[${msg.login}] ${msg.text}`
          logBuffer += lineText + '\n'
          win?.webContents.send('status-event', msg)
          continue
        }
        const p = pending.get(msg.id)
        if (p) {
          pending.delete(msg.id)
          p.resolve(msg.result ?? msg.error)
        }
      } catch (e) {
        // Не-JSON вывод — обычный print-лог из Python. Прокинем в renderer как текстовый лог.
        try {
          const lineText = `[system] ${line}`
          logBuffer += lineText + '\n'
          win?.webContents.send('status-event', { event: 'status', login: 'system', text: line })
        } catch {}
      }
    }
  })
  py.stderr.on('data', (chunk) => {
    const lines = chunk.toString().split(/\r?\n/).filter(Boolean)
    for (const line of lines) {
      const lineText = `[system:stderr] ${line}`
      logBuffer += lineText + '\n'
      try { win?.webContents.send('status-event', { event: 'status', login: 'system', text: lineText }) } catch {}
    }
  })
  py.on('exit', (c) => console.log('python exit', c))
}

function rpc(method, params) {
  return new Promise((resolve) => {
    const id = ++reqId
    pending.set(id, { resolve })
    py.stdin.write(JSON.stringify({ id, method, params }) + '\n')
  })
}

function createWindow() {
  win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })
  const devUrl = 'http://localhost:5173'
  const useDev = process.env.UI_DEV === '1'

  // Зарегистрируем диагностические события ДО загрузки контента,
  // чтобы не пропустить их и всегда иметь запись в буфере логов
  win.webContents.on('did-finish-load', () => {
    const t = 'UI loaded.'
    logBuffer += `[system] ${t}\n`
    try { win.webContents.send('status-event', { event: 'status', login: 'system', text: t }) } catch {}
    // Гарантируем корректный хэш для HashRouter
    try {
      win.webContents.executeJavaScript(`(function(){ if(!location.hash || location.hash === '#'){ location.hash = '#/'; } })()`)
    } catch {}
  })
  win.webContents.on('did-fail-load', (_e, code, desc, url, isMainFrame) => {
    const t = `[renderer] did-fail-load code=${code} desc=${desc} url=${url} main=${isMainFrame}`
    logBuffer += `[system] ${t}\n`
    try { win.webContents.send('status-event', { event: 'status', login: 'system', text: t }) } catch {}
  })
  win.webContents.on('render-process-gone', (_e, details) => {
    const t = `[renderer] gone reason=${details.reason} exitCode=${details.exitCode}`
    logBuffer += `[system] ${t}\n`
    try { win.webContents.send('status-event', { event: 'status', login: 'system', text: t }) } catch {}
  })
  win.on('unresponsive', () => {
    const t = '[renderer] window unresponsive'
    logBuffer += `[system] ${t}\n`
    try { win.webContents.send('status-event', { event: 'status', login: 'system', text: t }) } catch {}
  })

  // Теперь загружаем контент
  if (useDev) win.loadURL(devUrl)
  else win.loadFile(path.resolve(__dirname, '..', 'dist', 'index.html'))
}

app.whenReady().then(async () => {
  startPython()
  // Чистим кэш до создания окна, чтобы не вмешиваться в загрузку
  try { await session.defaultSession.clearCache() } catch {}
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

ipcMain.handle('rpc', async (_evt, { method, params }) => {
  if (method === 'get_logs') return { ok: true, text: logBuffer }
  if (method === 'clear_logs') { logBuffer = ''; return { ok: true } }
  return await rpc(method, params)
}) 