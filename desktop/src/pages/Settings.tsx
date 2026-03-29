/**
 * Страница «Настройки» — конфигурация поведения бота.
 *
 * Все стили — из theme.ts. SettingField — из ui.tsx (DRY).
 */

import { useState, useEffect, useCallback } from 'react';
import { Save, RotateCcw } from 'lucide-react';
import { ipc } from '../lib/ipc';
import { useIPC } from '../lib/hooks';
import type { Settings } from '../lib/types';
import { PageHeader, Card, Button, Spinner, SettingField } from '../components/ui';
import { theme } from '../lib/theme';

export default function SettingsPage() {
  const { data: settings, loading, refresh } = useIPC<Settings>(
    useCallback(() => ipc.getSettings(), []),
  );
  const [form, setForm] = useState<Settings>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (settings) {
      setForm(settings);
    }
  }, [settings]);

  const handleChange = (key: string, value: any) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    setSaved(false);
  };

  const handleSave = async () => {
    setSaving(true);
    try {
      await ipc.setSettings(form);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      refresh();
    } finally {
      setSaving(false);
    }
  };

  const handleReset = () => {
    if (settings) setForm(settings);
    setSaved(false);
  };

  if (loading && !settings) {
    return (
      <div className={theme.layout.center}>
        <Spinner size={32} />
      </div>
    );
  }

  return (
    <div className={theme.layout.page}>
      <PageHeader
        title="Настройки"
        subtitle="Конфигурация фарма и эмулятора"
        actions={
          <div className={theme.header.actions}>
            <Button variant="ghost" onClick={handleReset}>
              <RotateCcw size={16} />
              Сбросить
            </Button>
            <Button onClick={handleSave} disabled={saving}>
              <Save size={16} />
              {saved ? 'Сохранено ✓' : 'Сохранить'}
            </Button>
          </div>
        }
      />

      <div className={theme.layout.grid2}>
        {/* Настройки фарма */}
        <Card>
          <h3 className={theme.card.title}>Настройки фарма</h3>
          <div className="space-y-4">
            <SettingField
              label="Код острова"
              description="Код острова Fortnite Creative для фарма XP"
            >
              <input
                type="text"
                value={form.island_code ?? ''}
                onChange={(e) => handleChange('island_code', e.target.value)}
                placeholder="1234-5678-9012"
                className={theme.input.base}
              />
            </SettingField>

            <SettingField
              label="Время на острове (мин)"
              description="Минуты пребывания на острове перед перезапуском"
            >
              <input
                type="number"
                value={form.time_on_island_min ?? 59}
                onChange={(e) =>
                  handleChange('time_on_island_min', Number(e.target.value))
                }
                min={1}
                max={120}
                className={`${theme.input.base} w-24`}
              />
            </SettingField>

            <SettingField
              label="Макс. инстансов"
              description="Максимальное количество эмуляторов"
            >
              <input
                type="number"
                value={form.max_instances ?? 3}
                onChange={(e) =>
                  handleChange('max_instances', Number(e.target.value))
                }
                min={1}
                max={20}
                className={`${theme.input.base} w-24`}
              />
            </SettingField>
          </div>
        </Card>

        {/* VPN и сеть */}
        <Card>
          <h3 className={theme.card.title}>VPN и сеть</h3>
          <div className="space-y-4">
            <SettingField
              label="Регион VPN"
              description="Регион JumpJumpVPN для ротации IP"
            >
              <select
                value={form.vpn_region ?? 'us'}
                onChange={(e) => handleChange('vpn_region', e.target.value)}
                className={theme.input.base}
              >
                <option value="us">🇺🇸 США</option>
                <option value="uk">🇬🇧 Великобритания</option>
                <option value="de">🇩🇪 Германия</option>
                <option value="nl">🇳🇱 Нидерланды</option>
                <option value="fr">🇫🇷 Франция</option>
                <option value="jp">🇯🇵 Япония</option>
                <option value="au">🇦🇺 Австралия</option>
              </select>
            </SettingField>

            <SettingField
              label="Уровень логирования"
              description="Детализация вывода логов"
            >
              <select
                value={form.log_level ?? 'INFO'}
                onChange={(e) => handleChange('log_level', e.target.value)}
                className={theme.input.base}
              >
                <option value="DEBUG">DEBUG</option>
                <option value="INFO">INFO</option>
                <option value="WARNING">WARNING</option>
                <option value="ERROR">ERROR</option>
              </select>
            </SettingField>
          </div>
        </Card>

        {/* Внешний вид */}
        <Card>
          <h3 className={theme.card.title}>Внешний вид</h3>
          <div className="space-y-4">
            <SettingField label="Тема" description="Цветовая тема интерфейса">
              <select
                value={form.theme ?? 'dark-blue'}
                onChange={(e) => handleChange('theme', e.target.value)}
                className={theme.input.base}
              >
                <option value="dark-blue">Тёмно-синяя</option>
                <option value="dark">Тёмная</option>
              </select>
            </SettingField>
          </div>
        </Card>
      </div>
    </div>
  );
}
