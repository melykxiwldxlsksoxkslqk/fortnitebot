/**
 * Страница «Аккаунты» — управление аккаунтами Microsoft/Xbox.
 *
 * Все стили — из theme.ts.
 */

import { useState, useCallback } from 'react';
import { Users, Plus, Trash2, Upload, Eye, EyeOff } from 'lucide-react';
import { ipc } from '../lib/ipc';
import { useIPC } from '../lib/hooks';
import type { Account } from '../lib/types';
import { PageHeader, Card, Button, EmptyState, Spinner } from '../components/ui';
import { theme } from '../lib/theme';

export default function AccountsPage() {
  const { data: accounts, loading, refresh } = useIPC<Account[]>(
    useCallback(() => ipc.getAccounts(), []),
  );

  const [login, setLogin] = useState('');
  const [password, setPassword] = useState('');
  const [showPasswords, setShowPasswords] = useState(false);
  const [importText, setImportText] = useState('');
  const [showImport, setShowImport] = useState(false);
  const [busy, setBusy] = useState(false);

  // ---- Обработчики ----

  const handleAdd = async () => {
    if (!login.trim() || !password.trim()) return;
    setBusy(true);
    try {
      await ipc.addAccount(login.trim(), password.trim());
      setLogin('');
      setPassword('');
      refresh();
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async (email: string) => {
    if (!confirm(`Удалить аккаунт «${email}»?`)) return;
    setBusy(true);
    try {
      await ipc.deleteAccount(email);
      refresh();
    } finally {
      setBusy(false);
    }
  };

  const handleImport = async () => {
    if (!importText.trim()) return;
    setBusy(true);
    try {
      const result = await ipc.importAccounts(importText);
      alert(`Импортировано: ${result?.imported ?? 0} аккаунтов`);
      setImportText('');
      setShowImport(false);
      refresh();
    } finally {
      setBusy(false);
    }
  };

  // ---- Рендер ----

  if (loading && !accounts) {
    return (
      <div className={theme.layout.center}>
        <Spinner size={32} />
      </div>
    );
  }

  return (
    <div className={theme.layout.page}>
      <PageHeader
        title="Аккаунты"
        subtitle={`${accounts?.length ?? 0} аккаунтов в базе`}
        actions={
          <div className={theme.header.actions}>
            <Button
              variant="secondary"
              onClick={() => setShowImport(!showImport)}
            >
              <Upload size={16} />
              Импорт
            </Button>
            <Button
              variant="ghost"
              onClick={() => setShowPasswords(!showPasswords)}
            >
              {showPasswords ? <EyeOff size={16} /> : <Eye size={16} />}
            </Button>
          </div>
        }
      />

      {/* Форма добавления аккаунта */}
      <Card className="mb-4">
        <div className="flex items-center gap-3">
          <input
            type="email"
            value={login}
            onChange={(e) => setLogin(e.target.value)}
            placeholder="Email"
            className={theme.input.full}
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Пароль"
            className={theme.input.full}
            onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
          />
          <Button onClick={handleAdd} disabled={busy || !login.trim() || !password.trim()}>
            <Plus size={16} />
            Добавить
          </Button>
        </div>
      </Card>

      {/* Массовый импорт */}
      {showImport && (
        <Card className="mb-4 animate-fade-in">
          <h4 className={`${theme.setting.label} mb-2`}>
            Массовый импорт (email:пароль по одной строке)
          </h4>
          <textarea
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
            placeholder={`user1@email.com:password1\nuser2@email.com:password2`}
            rows={6}
            className={theme.input.textarea}
          />
          <div className="flex justify-end mt-2">
            <Button onClick={handleImport} disabled={busy || !importText.trim()}>
              <Upload size={16} />
              Импортировать
            </Button>
          </div>
        </Card>
      )}

      {/* Список аккаунтов */}
      {accounts && accounts.length > 0 ? (
        <Card>
          <div className={theme.account.divider}>
            {accounts.map((acct) => (
              <div key={acct.login} className={theme.account.row}>
                <div>
                  <p className={theme.account.email}>{acct.login}</p>
                  <p className={theme.account.password}>
                    {showPasswords ? acct.password || '••••••' : '••••••••'}
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => handleDelete(acct.login)}
                  disabled={busy}
                >
                  <Trash2 size={14} className={theme.text.red} />
                </Button>
              </div>
            ))}
          </div>
        </Card>
      ) : (
        <EmptyState
          icon={<Users size={48} />}
          title="Нет аккаунтов"
          description="Добавьте аккаунты Microsoft/Xbox для начала фарма."
        />
      )}
    </div>
  );
}
