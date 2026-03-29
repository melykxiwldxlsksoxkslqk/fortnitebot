/**
 * React-хуки для управления состоянием через IPC.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { ipc } from './ipc';
import type { BotStatus, IPCEvent } from './types';

// ============================================================================
// useIPC — универсальный хук для загрузки данных
// ============================================================================

export function useIPC<T>(
  fetcher: () => Promise<T>,
  intervalMs: number = 0,
): { data: T | null; loading: boolean; error: string | null; refresh: () => void } {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const result = await fetcher();
      setData(result);
      setError(null);
    } catch (err: any) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [fetcher]);

  useEffect(() => {
    fetchData();
    if (intervalMs > 0) {
      const timer = setInterval(fetchData, intervalMs);
      return () => clearInterval(timer);
    }
  }, [fetchData, intervalMs]);

  return { data, loading, error, refresh: fetchData };
}

// ============================================================================
// useStatus — статус бота в реальном времени
// ============================================================================

export function useStatus(pollMs: number = 3000) {
  return useIPC<BotStatus>(() => ipc.getStatus(), pollMs);
}

// ============================================================================
// useEvents — подписка на события бэкенда
// ============================================================================

export function useEvents(onEvent?: (event: IPCEvent) => void) {
  const [events, setEvents] = useState<IPCEvent[]>([]);
  const callbackRef = useRef(onEvent);
  callbackRef.current = onEvent;

  useEffect(() => {
    if (!window.epicbot) return;

    const unsub = window.epicbot.onEvent((event) => {
      setEvents((prev) => [...prev.slice(-99), event]);
      callbackRef.current?.(event);
    });

    return unsub;
  }, []);

  return events;
}

// ============================================================================
// useConnection — проверка подключения к бэкенду
// ============================================================================

export function useConnection() {
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const check = async () => {
      try {
        const result = await ipc.ping();
        setConnected(result === 'pong');
      } catch {
        setConnected(false);
      }
    };

    check();
    const timer = setInterval(check, 5000);
    return () => clearInterval(timer);
  }, []);

  return connected;
}
