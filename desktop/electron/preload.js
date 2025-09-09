const { contextBridge, ipcRenderer } = require('electron')
 
contextBridge.exposeInMainWorld('desktop', {
  rpc: (method, params) => ipcRenderer.invoke('rpc', { method, params }),
  onStatus: (cb) => {
    const handler = (_e, msg) => cb(msg)
    ipcRenderer.on('status-event', handler)
    // вернуть функцию отписки
    return () => ipcRenderer.removeListener('status-event', handler)
  }
}) 