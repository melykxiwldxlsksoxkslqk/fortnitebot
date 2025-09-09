import React, { useEffect, useState } from 'react'
import { AppShell, Group, Title, Button, ScrollArea, NavLink as MantineNavLink } from '@mantine/core'
import { Link, Outlet, useLocation } from 'react-router-dom'
///import PulsingBackground from './PulsingBackground'
import ErrorBoundary from './ErrorBoundary'

declare global {
  interface Window { desktop: any }
}

export default function AppLayout() {
  const location = useLocation()
  const [invertBg, setInvertBg] = useState(false)
  const [lobbySent, setLobbySent] = useState(false)

  useEffect(() => {
    const load = async () => {
      try {
      const s = await window.desktop.rpc('get_settings', null)
      setInvertBg(!!s?.invert_bg)
      } catch (e) {
        // ignore RPC errors to avoid renderer crash
      }
    }
    load()
  }, [])

  const startAll = async () => { await window.desktop.rpc('start', null) }
  const stopAll = async () => { await window.desktop.rpc('stop', null) }
  const lobbyReady = async () => {
    if (lobbySent) return
    setLobbySent(true)
    try {
      await window.desktop.rpc('signal_lobby_ready', null)
    } catch (e) {
      // ignore
    }
  }

  return (
    <AppShell padding="md" header={{ height: 56 }} navbar={{ width: 240, breakpoint: 'sm' }}>
      <AppShell.Header>
        <Group justify="space-between" px="md" h="100%" style={{ color: 'rgba(255,255,255,0.92)' }}>
          <Title order={3}>EpicBot</Title>
          <Group>
            <Button onClick={startAll}>Запустить ботов</Button>
            <Button color="red" onClick={stopAll}>Остановить всех</Button>
            <Button variant="light" onClick={lobbyReady} disabled={lobbySent}>Лобби готово → продолжить</Button>
          </Group>
        </Group>
      </AppShell.Header>

      <AppShell.Navbar p="sm">
        <ScrollArea style={{ height: '100%' }}>
          <MantineNavLink component={Link} to="/dashboard" label="Управление" active={location.pathname === '/dashboard'} />
          <MantineNavLink component={Link} to="/accounts" label="Аккаунты" active={location.pathname === '/accounts'} />
          <MantineNavLink component={Link} to="/proxies" label="Прокси" active={location.pathname === '/proxies'} />
          <MantineNavLink component={Link} to="/settings" label="Настройки" active={location.pathname === '/settings'} />
          <MantineNavLink component={Link} to="/logs" label="Логи" active={location.pathname === '/logs'} />
          <MantineNavLink component={Link} to="/test" label="Тест" active={location.pathname === '/test'} />
        </ScrollArea>
      </AppShell.Navbar>

      <AppShell.Main style={{ position: 'relative', isolation: 'isolate' }}>
        {/*<PulsingBackground invert={invertBg} />*/}
        <div style={{ position: 'relative', zIndex: 10 }}>
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </div>
      </AppShell.Main>
    </AppShell>
  )
} 