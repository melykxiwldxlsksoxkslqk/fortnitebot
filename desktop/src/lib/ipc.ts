/**
 * IPC-клиент — типизированная обёртка для JSON-RPC вызовов к Python-бэкенду.
 *
 * Использование:
 *   import { ipc } from '@/lib/ipc';
 *   const status = await ipc.getStatus();
 */

import type {
  Account,
  Instance,
  BotStatus,
  AppVersion,
  Settings,
  EmulatorConfig,
} from './types';

function call(method: string, params?: any): Promise<any> {
  if (window.epicbot) {
    return window.epicbot.call(method, params);
  }
  // Фоллбэк для разработки в браузере (без Electron)
  console.warn(`[IPC Мок] ${method}`, params);
  return Promise.resolve(null);
}

// ============================================================================
// API
// ============================================================================

export const ipc = {
  // --- Global ---
  ping: (): Promise<string> => call('ping'),
  getVersion: (): Promise<AppVersion> => call('get_version'),
  getStatus: (): Promise<BotStatus> => call('get_status'),

  // --- Accounts ---
  getAccounts: (): Promise<Account[]> => call('get_accounts'),
  addAccount: (login: string, password: string): Promise<{ success: boolean }> =>
    call('add_account', { login, password }),
  deleteAccount: (login: string): Promise<{ success: boolean }> =>
    call('delete_account', { login }),
  importAccounts: (text: string): Promise<{ imported: number }> =>
    call('import_accounts', { text }),

  // --- Instances ---
  listInstances: (): Promise<Instance[]> => call('list_instances'),
  setupInstance: (name: string): Promise<Instance> =>
    call('setup_instance', { name }),
  cloneInstance: (source: string, newName: string): Promise<Instance> =>
    call('clone_instance', { source, new_name: newName }),
  removeInstance: (name: string): Promise<{ success: boolean }> =>
    call('remove_instance', { name }),

  // --- Farm ---
  startFarm: (instanceName: string, email: string): Promise<{ success: boolean }> =>
    call('start_farm', { instance_name: instanceName, email }),
  stopFarm: (instanceName: string): Promise<{ success: boolean }> =>
    call('stop_farm', { instance_name: instanceName }),
  stopAll: (): Promise<{ success: boolean }> => call('stop_all'),
  shutdownAll: (): Promise<{ success: boolean }> => call('shutdown_all'),

  // --- Settings ---
  getSettings: (): Promise<Settings> => call('get_settings'),
  setSettings: (settings: Settings): Promise<{ success: boolean }> =>
    call('set_settings', { settings }),
  getEmulatorConfig: (): Promise<EmulatorConfig> => call('get_emulator_config'),
  setEmulatorConfig: (config: EmulatorConfig): Promise<{ success: boolean }> =>
    call('set_emulator_config', { config_data: config }),

  // --- Logs ---
  getRecentLogs: (count?: number): Promise<string[]> =>
    call('get_recent_logs', { count: count ?? 200 }),
};
