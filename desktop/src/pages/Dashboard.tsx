/**
 * Страница «Панель» — обзор состояния ботов и фарма.
 *
 * Все стили — из theme.ts. Локальных Tailwind-строк нет.
 */

import { Monitor, Users, Zap, Activity } from 'lucide-react';
import { useStatus, useEvents } from '../lib/hooks';
import { PageHeader, StatCard, Card, Spinner } from '../components/ui';
import { theme, getInstanceDotColor } from '../lib/theme';

export default function DashboardPage() {
  const { data: status, loading } = useStatus(3000);
  const events = useEvents();

  if (loading && !status) {
    return (
      <div className={theme.layout.center}>
        <Spinner size={32} />
      </div>
    );
  }

  return (
    <div className={theme.layout.page}>
      <PageHeader
        title="Панель управления"
        subtitle="Обзор состояния ботов и фарма"
      />

      {/* Сетка статистики */}
      <div className={theme.layout.grid4}>
        <StatCard
          label="Всего инстансов"
          value={status?.total_instances ?? 0}
          icon={<Monitor size={24} />}
          color="brand"
        />
        <StatCard
          label="Активные сессии"
          value={status?.active_sessions ?? 0}
          icon={<Zap size={24} />}
          color="green"
        />
        <StatCard
          label="Аккаунтов"
          value={status?.accounts_in_db ?? 0}
          icon={<Users size={24} />}
          color="yellow"
        />
        <StatCard
          label="Аптайм"
          value="—"
          icon={<Activity size={24} />}
          color="brand"
        />
      </div>

      {/* Обзор инстансов и событий */}
      <div className={theme.layout.grid2}>
        {/* Активные инстансы */}
        <Card>
          <h3 className={theme.card.title}>Активные инстансы</h3>
          {status?.instances && status.instances.length > 0 ? (
            <div className="space-y-2">
              {status.instances.map((inst: any, i: number) => (
                <div key={i} className={theme.instance.row}>
                  <div className={theme.instance.cardHeaderLeft}>
                    <div
                      className={`${theme.instance.dot} ${getInstanceDotColor(
                        inst.farm_state ?? '',
                        inst.status ?? '',
                      )}`}
                    />
                    <span className={theme.instance.name}>{inst.name}</span>
                  </div>
                  <span className={theme.instance.state}>
                    {inst.farm_state || inst.status}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className={theme.text.muted}>Инстансов пока нет.</p>
          )}
        </Card>

        {/* Последние события */}
        <Card>
          <h3 className={theme.card.title}>Последние события</h3>
          {events.length > 0 ? (
            <div className={theme.event.list}>
              {events
                .slice()
                .reverse()
                .slice(0, 20)
                .map((evt, i) => (
                  <div
                    key={i}
                    className={theme.event.item}
                  >
                    {evt.params?.message || evt.method}
                  </div>
                ))}
            </div>
          ) : (
            <p className={theme.text.muted}>
              Нет событий. Запустите фарм, чтобы увидеть активность.
            </p>
          )}
        </Card>
      </div>
    </div>
  );
}
