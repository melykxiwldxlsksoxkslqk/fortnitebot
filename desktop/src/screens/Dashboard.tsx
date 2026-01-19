import React from 'react'
import { useEffect, useState, useRef } from 'react'
import { 
  Group, 
  Table, 
  Title, 
  Badge, 
  Text, 
  Textarea, 
  Button, 
  Paper,
  SimpleGrid,
  Stack,
  Box,
  Tooltip,
  ActionIcon,
  Transition
} from '@mantine/core'
import {
  IconActivity,
  IconUsers,
  IconClock,
  IconCopy,
  IconTrash,
  IconRefresh,
  IconRobot,
  IconAlertCircle
} from '@tabler/icons-react'

declare global {
  interface Window { desktop: any }
}

interface StatCardProps {
  title: string
  value: string | number
  icon: React.ReactNode
  color: string
  subtitle?: string
}

function StatCard({ title, value, icon, color, subtitle }: StatCardProps) {
  return (
    <Paper p="md" radius="lg" className="stat-card">
      <Group justify="space-between" align="flex-start">
        <Stack gap={4}>
          <Text size="sm" c="dimmed" fw={500}>{title}</Text>
          <Text size="xl" fw={700} style={{ color: `var(--mantine-color-${color}-5)` }}>
            {value}
          </Text>
          {subtitle && <Text size="xs" c="dimmed">{subtitle}</Text>}
        </Stack>
        <Box
          style={{
            width: 42,
            height: 42,
            borderRadius: 10,
            background: `rgba(var(--mantine-color-${color}-filled-rgb), 0.15)`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: `var(--mantine-color-${color}-5)`,
          }}
        >
          {icon}
        </Box>
      </Group>
    </Paper>
  )
}

export default function Dashboard() {
  const [status, setStatus] = useState<any>({ bots: [], threads: [], accounts: [], status: {}, settings: {} })
  const [logText, setLogText] = useState('')
  const [isRefreshing, setIsRefreshing] = useState(false)
  const logRef = useRef<HTMLTextAreaElement>(null)

  const refresh = async () => {
    setIsRefreshing(true)
    try {
      const s = await window.desktop?.rpc?.('get_status', null)
      if (s) setStatus(s)
    } catch {} finally {
      setIsRefreshing(false)
    }
  }

  useEffect(() => {
    let unsub: () => void = () => {}
    refresh()
    try {
      if (window.desktop?.onStatus) {
        unsub = window.desktop.onStatus((msg: any) => {
          refresh()
          setLogText((t: string) => {
            const timestamp = new Date().toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            const line = `[${timestamp}] [${msg.login}] ${msg.text}`
            const next = t ? (t + (t.endsWith('\n') ? '' : '\n') + line + '\n') : (line + '\n')
            const lines = next.split('\n')
            return lines.length > 4000 ? lines.slice(-4000).join('\n') : next
          })
        })
      }
    } catch {}
    return () => { try { unsub() } catch {} }
  }, [])

  useEffect(() => {
    (async () => {
      try {
        const res = await window.desktop?.rpc?.('get_logs', null)
        if (res?.ok) setLogText(res.text || '')
      } catch {}
    })()
  }, [])

  // Auto-scroll logs
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logText])

  const copyLogs = async () => { 
    try { 
      await navigator.clipboard.writeText(logText)
    } catch {} 
  }
  
  const clearLogs = async () => { 
    try { 
      await window.desktop?.rpc?.('clear_logs', null)
      setLogText('') 
    } catch {} 
  }

  const activeLogins: string[] = (status.active && Array.isArray(status.active) && status.active.length > 0)
    ? status.active
    : Array.from(new Set([
        ...Object.keys(status.status || {}),
        ...((status.threads || []) as string[]),
      ]))

  const colorFor = (text: string) => {
    const t = (text || '').toLowerCase()
    if (t.includes('ошибка') || t.includes('fatal') || t.includes('failed') || t.includes('error')) return 'red'
    if (t.includes('закрыт') || t.includes('останов') || t.includes('stop')) return 'orange'
    if (t.includes('успех') || t.includes('готов') || t.includes('запущен') || t.includes('success')) return 'green'
    if (t.includes('ожидан') || t.includes('wait') || t.includes('load')) return 'yellow'
    return 'blue'
  }

  const runningCount = activeLogins.filter(login => {
    const text = (status.status?.[login]?.status || '').toLowerCase()
    return text.includes('запущен') || text.includes('работа') || text.includes('running')
  }).length

  return (
    <Stack gap="lg">
      {/* Stats Grid */}
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 4 }} spacing="md">
        <StatCard
          title="Всего ботов"
          value={activeLogins.length}
          icon={<IconRobot size={22} />}
          color="violet"
          subtitle="активных сессий"
        />
        <StatCard
          title="Работают"
          value={runningCount}
          icon={<IconActivity size={22} />}
          color="green"
          subtitle="в данный момент"
        />
        <StatCard
          title="Аккаунтов"
          value={status.accounts?.length || 0}
          icon={<IconUsers size={22} />}
          color="blue"
          subtitle="в базе данных"
        />
        <StatCard
          title="Uptime"
          value={status.uptime || '—'}
          icon={<IconClock size={22} />}
          color="cyan"
          subtitle="время работы"
        />
      </SimpleGrid>

      {/* Status Table */}
      <Paper p="md" radius="lg" withBorder>
        <Group justify="space-between" mb="md">
          <Group gap="sm">
            <Box className="section-header-icon">
              <IconActivity size={18} />
            </Box>
            <Title order={4}>Статусы ботов</Title>
          </Group>
          <Tooltip label="Обновить">
            <ActionIcon 
              variant="light" 
              onClick={refresh}
              loading={isRefreshing}
            >
              <IconRefresh size={18} />
            </ActionIcon>
          </Tooltip>
        </Group>

        {activeLogins.length === 0 ? (
          <Box className="empty-state">
            <IconAlertCircle size={48} style={{ opacity: 0.3 }} />
            <Text size="lg" fw={500} c="dimmed" mt="md">Нет активных ботов</Text>
            <Text size="sm" c="dimmed">Нажмите «Запустить» в шапке, чтобы начать</Text>
          </Box>
        ) : (
          <Table highlightOnHover withTableBorder withColumnBorders>
            <Table.Thead>
              <Table.Tr>
                <Table.Th>Логин</Table.Th>
                <Table.Th>Статус</Table.Th>
                <Table.Th>Время</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {activeLogins.map((login: string) => {
                const statusInfo = status.status?.[login]
                const text = statusInfo?.status || (status.threads?.includes(login) ? 'Запуск...' : '—')
                const time = statusInfo?.updated_at 
                  ? new Date(statusInfo.updated_at).toLocaleTimeString('ru-RU')
                  : '—'
                return (
                  <Table.Tr key={login}>
                    <Table.Td>
                      <Group gap="xs">
                        <Box
                          style={{
                            width: 8,
                            height: 8,
                            borderRadius: '50%',
                            background: colorFor(text) === 'green' 
                              ? 'var(--mantine-color-green-5)' 
                              : colorFor(text) === 'red'
                              ? 'var(--mantine-color-red-5)'
                              : 'var(--mantine-color-gray-5)',
                          }}
                        />
                        <Text fw={500}>{login}</Text>
                      </Group>
                    </Table.Td>
                    <Table.Td>
                      <Badge color={colorFor(text)} variant="light" radius="md">
                        {text}
                      </Badge>
                    </Table.Td>
                    <Table.Td>
                      <Text size="sm" c="dimmed">{time}</Text>
                    </Table.Td>
                  </Table.Tr>
                )
              })}
            </Table.Tbody>
          </Table>
        )}
      </Paper>

      {/* Live Logs */}
      <Paper p="md" radius="lg" withBorder>
        <Group justify="space-between" mb="md">
          <Group gap="sm">
            <Box className="section-header-icon">
              <IconActivity size={18} />
            </Box>
            <Title order={4}>Live Логи</Title>
            <Badge variant="dot" color="green" size="sm">streaming</Badge>
          </Group>
          <Group gap="xs">
            <Tooltip label="Копировать логи">
              <ActionIcon variant="light" onClick={copyLogs}>
                <IconCopy size={18} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Очистить">
              <ActionIcon variant="light" color="gray" onClick={clearLogs}>
                <IconTrash size={18} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
        
        <Textarea
          ref={logRef}
          value={logText}
          onChange={(e) => setLogText(e.currentTarget.value)}
          autosize
          minRows={12}
          maxRows={20}
          readOnly
          styles={{
            input: {
              fontFamily: 'var(--mantine-font-family-monospace)',
              fontSize: '0.85rem',
              backgroundColor: 'var(--epic-bg-primary)',
              border: '1px solid var(--epic-border)',
            },
          }}
          placeholder="Логи появятся здесь после запуска ботов..."
        />
      </Paper>
    </Stack>
  )
} 