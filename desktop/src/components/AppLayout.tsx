import React, { useEffect, useState } from 'react'
import { 
  AppShell, 
  Group, 
  Title, 
  Button, 
  ScrollArea, 
  NavLink as MantineNavLink,
  Text,
  Badge,
  Tooltip,
  ActionIcon,
  Divider,
  Box,
  Stack
} from '@mantine/core'
import { 
  IconDashboard, 
  IconUsers, 
  IconNetwork, 
  IconSettings, 
  IconFileText, 
  IconTestPipe,
  IconPlayerPlay,
  IconPlayerStop,
  IconCheck,
  IconRocket,
  IconBrandGithub,
  IconRefresh
} from '@tabler/icons-react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import ErrorBoundary from './ErrorBoundary'

declare global {
  interface Window { desktop: any }
}

interface NavItem {
  path: string
  label: string
  icon: React.ReactNode
  description?: string
}

const navItems: NavItem[] = [
  { path: '/dashboard', label: 'Управление', icon: <IconDashboard size={20} />, description: 'Статусы и логи ботов' },
  { path: '/accounts', label: 'Аккаунты', icon: <IconUsers size={20} />, description: 'Epic Games аккаунты' },
  { path: '/proxies', label: 'Прокси', icon: <IconNetwork size={20} />, description: 'Настройка прокси-серверов' },
  { path: '/settings', label: 'Настройки', icon: <IconSettings size={20} />, description: 'Параметры бота' },
  { path: '/logs', label: 'Логи', icon: <IconFileText size={20} />, description: 'Полные логи системы' },
  { path: '/test', label: 'Тест', icon: <IconTestPipe size={20} />, description: 'Тестовый экран' },
]

export default function AppLayout() {
  const location = useLocation()
  const [lobbySent, setLobbySent] = useState(false)
  const [botsRunning, setBotsRunning] = useState(false)
  const [botCount, setBotCount] = useState(0)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    const loadStatus = async () => {
      try {
        const s = await window.desktop?.rpc?.('get_status', null)
        if (s) {
          const activeCount = s.active?.length || Object.keys(s.status || {}).length || 0
          setBotCount(activeCount)
          setBotsRunning(activeCount > 0)
        }
      } catch (e) {
        // ignore
      }
    }
    loadStatus()
    
    // Subscribe to status updates
    let unsub: () => void = () => {}
    try {
      if (window.desktop?.onStatus) {
        unsub = window.desktop.onStatus(() => loadStatus())
      }
    } catch {}
    
    return () => { try { unsub() } catch {} }
  }, [])

  const startAll = async () => { 
    setIsLoading(true)
    try {
      await window.desktop.rpc('start', null)
      setBotsRunning(true)
    } finally {
      setIsLoading(false)
    }
  }
  
  const stopAll = async () => { 
    setIsLoading(true)
    try {
      await window.desktop.rpc('stop', null)
      setBotsRunning(false)
      setLobbySent(false)
    } finally {
      setIsLoading(false)
    }
  }
  
  const lobbyReady = async () => {
    if (lobbySent) return
    setLobbySent(true)
    try {
      await window.desktop.rpc('signal_lobby_ready', null)
    } catch (e) {
      setLobbySent(false)
    }
  }

  return (
    <AppShell
      padding="lg"
      header={{ height: 64 }}
      navbar={{ width: 260, breakpoint: 'sm' }}
      styles={{
        main: {
          background: 'var(--epic-bg-primary)',
          minHeight: '100vh',
        },
      }}
    >
      {/* Header */}
      <AppShell.Header className="header-gradient">
        <Group justify="space-between" px="lg" h="100%">
          {/* Logo */}
          <Group gap="sm">
            <Box
              style={{
                width: 40,
                height: 40,
                borderRadius: 10,
                background: 'linear-gradient(135deg, var(--epic-accent) 0%, var(--epic-accent-dark) 100%)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                boxShadow: '0 4px 12px rgba(139, 92, 246, 0.3)',
              }}
            >
              <IconRocket size={22} color="white" />
            </Box>
            <Stack gap={0}>
              <Title order={4} style={{ lineHeight: 1.2 }}>EpicBot</Title>
              <Text size="xs" c="dimmed">v2.1.0</Text>
            </Stack>
          </Group>

          {/* Status & Controls */}
          <Group gap="md">
            {botsRunning && (
              <Badge 
                size="lg" 
                variant="dot" 
                color="green"
                styles={{ root: { textTransform: 'none' } }}
              >
                {botCount} бот{botCount === 1 ? '' : botCount < 5 ? 'а' : 'ов'} активно
              </Badge>
            )}
            
            <Divider orientation="vertical" />
            
            <Group gap="xs">
              <Tooltip label="Запустить всех ботов" position="bottom">
                <Button
                  leftSection={<IconPlayerPlay size={18} />}
                  onClick={startAll}
                  loading={isLoading}
                  disabled={botsRunning}
                  variant={botsRunning ? 'light' : 'filled'}
                  color="green"
                >
                  Запустить
                </Button>
              </Tooltip>
              
              <Tooltip label="Остановить всех ботов" position="bottom">
                <Button
                  leftSection={<IconPlayerStop size={18} />}
                  onClick={stopAll}
                  loading={isLoading}
                  disabled={!botsRunning}
                  variant="light"
                  color="red"
                >
                  Стоп
                </Button>
              </Tooltip>
              
              <Tooltip label="Сигнал готовности лобби" position="bottom">
                <Button
                  leftSection={lobbySent ? <IconCheck size={18} /> : <IconRefresh size={18} />}
                  onClick={lobbyReady}
                  disabled={lobbySent || !botsRunning}
                  variant="light"
                  color={lobbySent ? 'green' : 'gray'}
                >
                  Лобби готово
                </Button>
              </Tooltip>
            </Group>
          </Group>
        </Group>
      </AppShell.Header>

      {/* Sidebar */}
      <AppShell.Navbar className="sidebar-gradient" p="md">
        <AppShell.Section grow component={ScrollArea} scrollbarSize={6}>
          <Stack gap={4}>
            {navItems.map((item) => (
              <Tooltip 
                key={item.path}
                label={item.description} 
                position="right"
                transitionProps={{ duration: 200 }}
                disabled={!item.description}
              >
                <MantineNavLink
                  component={Link}
                  to={item.path}
                  label={item.label}
                  leftSection={item.icon}
                  active={location.pathname === item.path}
                  styles={{
                    root: {
                      borderRadius: 8,
                      transition: 'all 0.15s ease',
                    },
                    label: {
                      fontWeight: 500,
                    },
                  }}
                />
              </Tooltip>
            ))}
          </Stack>
        </AppShell.Section>
        
        <AppShell.Section>
          <Divider my="sm" />
          <Group justify="center" gap="xs">
            <Tooltip label="GitHub">
              <ActionIcon 
                variant="subtle" 
                color="gray" 
                size="lg"
                component="a"
                href="https://github.com/melykxiwldxlsksoxkslqk/fortnitebot"
                target="_blank"
              >
                <IconBrandGithub size={20} />
              </ActionIcon>
            </Tooltip>
          </Group>
          <Text size="xs" c="dimmed" ta="center" mt="xs">
            © 2026 EpicBot
          </Text>
        </AppShell.Section>
      </AppShell.Navbar>

      {/* Main Content */}
      <AppShell.Main>
        <Box className="fade-in">
          <ErrorBoundary>
            <Outlet />
          </ErrorBoundary>
        </Box>
      </AppShell.Main>
    </AppShell>
  )
} 