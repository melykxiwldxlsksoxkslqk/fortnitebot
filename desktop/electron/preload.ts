/**
 * Preload-скрипт — безопасный IPC-мост для renderer-процесса.
 */

import { contextBridge, ipcRenderer } from 'electron';

export interface IPCBridge {
  /** Вызов JSON-RPC метода на Python-бэкенде */
  call: (method: string, params?: any) => Promise<any>;
  /** Пинг бэкенда */
  ping: () => Promise<string | null>;
  /** Подписка на события от бэкенда */
  onEvent: (callback: (event: any) => void) => () => void;
  /** Подписка на ответы от бэкенда */
  onResponse: (callback: (response: any) => void) => () => void;
}

contextBridge.exposeInMainWorld('epicbot', {
  call: (method: string, params?: any) => {
    return ipcRenderer.invoke('ipc:call', method, params);
  },

  ping: () => {
    return ipcRenderer.invoke('ipc:ping');
  },

  onEvent: (callback: (event: any) => void) => {
    const handler = (_: any, event: any) => callback(event);
    ipcRenderer.on('ipc:event', handler);
    return () => ipcRenderer.removeListener('ipc:event', handler);
  },

  onResponse: (callback: (response: any) => void) => {
    const handler = (_: any, response: any) => callback(response);
    ipcRenderer.on('ipc:response', handler);
    return () => ipcRenderer.removeListener('ipc:response', handler);
  },
} as IPCBridge);
