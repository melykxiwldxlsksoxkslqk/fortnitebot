/**
 * Страница «Логи» — просмотр логов в реальном времени.
 *
 * Все стили — из theme.ts. Маппинг цвета строки — getLogLineColor().
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { ScrollText, RefreshCw, ArrowDown } from 'lucide-react';
import { ipc } from '../lib/ipc';
import { useIPC, useEvents } from '../lib/hooks';
import { PageHeader, Card, Button, Spinner } from '../components/ui';
import { theme, getLogLineColor } from '../lib/theme';

export default function LogsPage() {
  const { data: logs, loading, refresh } = useIPC<string[]>(
    useCallback(() => ipc.getRecentLogs(500), []),
    10000,
  );
  const [filter, setFilter] = useState('');
  const [autoScroll, setAutoScroll] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);

  // Подписка на события статуса
  const events = useEvents();

  // Автопрокрутка вниз
  useEffect(() => {
    if (autoScroll && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs, events, autoScroll]);

  const filteredLogs =
    logs?.filter((line) =>
      filter ? line.toLowerCase().includes(filter.toLowerCase()) : true,
    ) ?? [];

  if (loading && !logs) {
    return (
      <div className={theme.layout.center}>
        <Spinner size={32} />
      </div>
    );
  }

  return (
    <div className={theme.layout.pageFullHeight}>
      <PageHeader
        title="Логи"
        subtitle={`${filteredLogs.length} строк`}
        actions={
          <div className={theme.header.actions}>
            <input
              type="text"
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              placeholder="Фильтр логов..."
              className={`${theme.input.base} w-48`}
            />
            <Button
              variant={autoScroll ? 'primary' : 'secondary'}
              size="sm"
              onClick={() => setAutoScroll(!autoScroll)}
            >
              <ArrowDown size={14} />
            </Button>
            <Button variant="ghost" size="sm" onClick={() => refresh()}>
              <RefreshCw size={14} />
            </Button>
          </div>
        }
      />

      <Card className={`flex-1 min-h-0 ${theme.card.noPadding}`}>
        <div ref={containerRef} className={theme.logs.container}>
          {filteredLogs.length > 0 ? (
            filteredLogs.map((line, i) => (
              <div key={i} className={`${theme.logs.line} ${getLogLineColor(line)}`}>
                {line}
              </div>
            ))
          ) : (
            <div className={theme.logs.emptyRoot}>
              <ScrollText size={48} className="mb-4 opacity-50" />
              <p>Логов пока нет.</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
}
