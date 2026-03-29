/**
 * Типы данных для IPC-моста, доступного через preload.
 */

export interface IPCBridge {
  call: (method: string, params?: any) => Promise<any>;
  ping: () => Promise<string | null>;
  onEvent: (callback: (event: IPCEvent) => void) => () => void;
  onResponse: (callback: (response: any) => void) => () => void;
}

declare global {
  interface Window {
    epicbot: IPCBridge;
  }
}

// ============================================================================
// Domain Types
// ============================================================================

export interface Account {
  login: string;
  password?: string;
  created_at?: string;
}

export interface Instance {
  name: string;
  index: number;
  status: 'running' | 'stopped' | 'unknown';
  farm_state: string;
  pid?: number;
}

export interface BotStatus {
  instances: Instance[];
  active_sessions: number;
  total_instances: number;
  accounts_in_db: number;
}

export interface AppVersion {
  version: string;
  mode: string;
}

export interface Settings {
  island_code?: string;
  time_on_island_min?: number;
  log_level?: string;
  max_instances?: number;
  vpn_region?: string;
  [key: string]: any;
}

export interface EmulatorConfig {
  ldplayer?: {
    install_path?: string;
    console_exe?: string;
  };
  vpn?: {
    package_name?: string;
    region?: string;
  };
  session?: {
    island_code?: string;
    loop_forever?: boolean;
    max_sessions?: number;
  };
  [key: string]: any;
}

export interface IPCEvent {
  jsonrpc: string;
  method: string;
  params?: any;
}

export interface LogEntry {
  text: string;
  timestamp?: string;
  level?: string;
}
