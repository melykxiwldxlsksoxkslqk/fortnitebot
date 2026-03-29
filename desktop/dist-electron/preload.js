"use strict";
/**
 * Preload-скрипт — безопасный IPC-мост для renderer-процесса.
 */
Object.defineProperty(exports, "__esModule", { value: true });
const electron_1 = require("electron");
electron_1.contextBridge.exposeInMainWorld('epicbot', {
    call: (method, params) => {
        return electron_1.ipcRenderer.invoke('ipc:call', method, params);
    },
    ping: () => {
        return electron_1.ipcRenderer.invoke('ipc:ping');
    },
    onEvent: (callback) => {
        const handler = (_, event) => callback(event);
        electron_1.ipcRenderer.on('ipc:event', handler);
        return () => electron_1.ipcRenderer.removeListener('ipc:event', handler);
    },
    onResponse: (callback) => {
        const handler = (_, response) => callback(response);
        electron_1.ipcRenderer.on('ipc:response', handler);
        return () => electron_1.ipcRenderer.removeListener('ipc:response', handler);
    },
});
//# sourceMappingURL=preload.js.map