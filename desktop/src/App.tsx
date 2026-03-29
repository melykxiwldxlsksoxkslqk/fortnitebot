/**
 * Корневой компонент приложения — боковое меню + маршрутизация.
 *
 * FIX: v7_startTransition + v7_relativeSplatPath — убирают предупреждения React Router v6.
 * Все стили из theme.ts — нет ни одного хардкод-класса.
 */

import {
  BrowserRouter,
  Routes,
  Route,
  NavLink,
} from 'react-router-dom';
import {
  LayoutDashboard,
  Monitor,
  Users,
  Settings,
  ScrollText,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { useConnection } from './lib/hooks';
import { theme, connectionColors, getNavLinkClass } from './lib/theme';

import DashboardPage from './pages/Dashboard';
import InstancesPage from './pages/Instances';
import AccountsPage from './pages/Accounts';
import SettingsPage from './pages/Settings';
import LogsPage from './pages/Logs';

/** Элементы навигации */
const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Панель' },
  { to: '/instances', icon: Monitor, label: 'Инстансы' },
  { to: '/accounts', icon: Users, label: 'Аккаунты' },
  { to: '/settings', icon: Settings, label: 'Настройки' },
  { to: '/logs', icon: ScrollText, label: 'Логи' },
] as const;

export default function App() {
  const connected = useConnection();
  const connColor = connected ? connectionColors.online : connectionColors.offline;

  return (
    <BrowserRouter
      future={{ v7_startTransition: true, v7_relativeSplatPath: true }}
    >
      <div className={theme.layout.root}>
        {/* Боковое меню */}
        <aside className={theme.sidebar.root}>
          {/* Логотип */}
          <div className={theme.sidebar.logo}>
            <h1 className={theme.sidebar.logoTitle}>⚡ EpicBot</h1>
            <p className={theme.sidebar.logoSubtitle}>v4.0 — Режим эмулятора</p>
          </div>

          {/* Навигация */}
          <nav className={theme.sidebar.nav}>
            {NAV_ITEMS.map(({ to, icon: Icon, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) => getNavLinkClass(isActive)}
              >
                <Icon size={18} />
                {label}
              </NavLink>
            ))}
          </nav>

          {/* Статус подключения */}
          <div className={theme.sidebar.footer}>
            <div className={theme.sidebar.footerContent}>
              {connected ? (
                <>
                  <Wifi size={14} className={connColor} />
                  <span className={connColor}>Бэкенд подключён</span>
                </>
              ) : (
                <>
                  <WifiOff size={14} className={connColor} />
                  <span className={connColor}>Нет подключения</span>
                </>
              )}
            </div>
          </div>
        </aside>

        {/* Основной контент */}
        <main className={theme.layout.main}>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/instances" element={<InstancesPage />} />
            <Route path="/accounts" element={<AccountsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/logs" element={<LogsPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
