/**
 * Страница «Инстансы» — управление эмуляторами LDPlayer.
 *
 * Все стили — из theme.ts. Маппинг состояния — getFarmBadgeColor().
 */

import { useState, useCallback } from 'react';
import { Monitor, Plus, Play, Square, Trash2, Copy } from 'lucide-react';
import { ipc } from '../lib/ipc';
import { useIPC } from '../lib/hooks';
import type { Instance, Account } from '../lib/types';
import {
  PageHeader,
  Card,
  Button,
  Badge,
  EmptyState,
  Spinner,
} from '../components/ui';
import { theme, getFarmBadgeColor } from '../lib/theme';

export default function InstancesPage() {
  const {
    data: instances,
    loading,
    refresh,
  } = useIPC<Instance[]>(
    useCallback(() => ipc.listInstances(), []),
    5000,
  );
  const { data: accounts } = useIPC<Account[]>(
    useCallback(() => ipc.getAccounts(), []),
  );
  const [newName, setNewName] = useState('');
  const [busy, setBusy] = useState<string | null>(null);

  // ---- Обработчики ----

  const handleSetup = async () => {
    if (!newName.trim()) return;
    setBusy('setup');
    try {
      await ipc.setupInstance(newName.trim());
      setNewName('');
      refresh();
    } finally {
      setBusy(null);
    }
  };

  const handleStart = async (instance: Instance) => {
    if (!accounts?.length) {
      alert('Сначала добавьте аккаунты!');
      return;
    }
    const email = accounts[0].login;
    setBusy(instance.name);
    try {
      await ipc.startFarm(instance.name, email);
      refresh();
    } finally {
      setBusy(null);
    }
  };

  const handleStop = async (name: string) => {
    setBusy(name);
    try {
      await ipc.stopFarm(name);
      refresh();
    } finally {
      setBusy(null);
    }
  };

  const handleRemove = async (name: string) => {
    if (!confirm(`Удалить инстанс «${name}»?`)) return;
    setBusy(name);
    try {
      await ipc.removeInstance(name);
      refresh();
    } finally {
      setBusy(null);
    }
  };

  const handleClone = async (source: string) => {
    const cloneName = prompt('Имя нового инстанса:', `${source}-клон`);
    if (!cloneName) return;
    setBusy(source);
    try {
      await ipc.cloneInstance(source, cloneName);
      refresh();
    } finally {
      setBusy(null);
    }
  };

  // ---- Рендер ----

  if (loading && !instances) {
    return (
      <div className={theme.layout.center}>
        <Spinner size={32} />
      </div>
    );
  }

  return (
    <div className={theme.layout.page}>
      <PageHeader
        title="Инстансы"
        subtitle="Управление эмуляторами LDPlayer"
        actions={
          <div className={theme.header.actions}>
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="Имя инстанса..."
              className={theme.input.base}
              onKeyDown={(e) => e.key === 'Enter' && handleSetup()}
            />
            <Button onClick={handleSetup} disabled={busy === 'setup' || !newName.trim()}>
              <Plus size={16} />
              Создать
            </Button>
          </div>
        }
      />

      {instances && instances.length > 0 ? (
        <div className={theme.layout.grid3}>
          {instances.map((inst) => (
            <Card key={inst.name} className={theme.card.hover}>
              <div className={theme.instance.cardHeader}>
                <div className={theme.instance.cardHeaderLeft}>
                  <Monitor size={18} className={theme.text.brand} />
                  <span className={`font-semibold ${theme.text.white}`}>{inst.name}</span>
                </div>
                <Badge color={getFarmBadgeColor(inst.farm_state)}>
                  {inst.farm_state || inst.status}
                </Badge>
              </div>

              <div className={theme.instance.meta}>
                <div>Индекс: {inst.index}</div>
                <div>Статус: {inst.status}</div>
                {inst.pid && <div>PID: {inst.pid}</div>}
              </div>

              <div className={theme.instance.actions}>
                {inst.farm_state === 'idle' ||
                !inst.farm_state ||
                inst.farm_state === 'stopped' ? (
                  <Button
                    size="sm"
                    onClick={() => handleStart(inst)}
                    disabled={busy === inst.name}
                  >
                    <Play size={14} />
                    Запуск
                  </Button>
                ) : (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => handleStop(inst.name)}
                    disabled={busy === inst.name}
                  >
                    <Square size={14} />
                    Стоп
                  </Button>
                )}
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => handleClone(inst.name)}
                  disabled={busy === inst.name}
                >
                  <Copy size={14} />
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => handleRemove(inst.name)}
                  disabled={busy === inst.name}
                >
                  <Trash2 size={14} className={theme.text.red} />
                </Button>
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<Monitor size={48} />}
          title="Нет инстансов"
          description="Создайте инстанс LDPlayer, чтобы начать фарм."
        />
      )}
    </div>
  );
}
