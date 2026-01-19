import React from 'react'
import { useEffect, useState } from 'react'
import { 
  Button, 
  Group, 
  Title, 
  Table, 
  TextInput, 
  Textarea, 
  Modal,
  Paper,
  Stack,
  Box,
  Text,
  Badge,
  ActionIcon,
  Tooltip,
  PasswordInput,
  Alert
} from '@mantine/core'
import {
  IconNetwork,
  IconPlus,
  IconTrash,
  IconDeviceFloppy,
  IconRefresh,
  IconUpload,
  IconServer,
  IconLock,
  IconUser,
  IconAlertCircle,
  IconWorldWww
} from '@tabler/icons-react'

declare global {
  interface Window { desktop: any }
}

interface Row { host: string; port: string; username?: string; password?: string }

export default function Proxies() {
  const [rows, setRows] = useState<Row[]>([])
  const [bulkOpen, setBulkOpen] = useState(false)
  const [bulkText, setBulkText] = useState('')
  const [isSaving, setIsSaving] = useState(false)

  const load = async () => {
    const r = await window.desktop.rpc('get_proxies', null)
    if (r?.ok) setRows(r.proxies || [])
  }
  useEffect(() => { load() }, [])

  const save = async () => {
    setIsSaving(true)
    try {
      const cleaned = rows.filter(r => (r.host || '').trim() && (r.port || '').trim())
      await window.desktop.rpc('save_proxies', { proxies: cleaned })
      await load()
    } finally {
      setIsSaving(false)
    }
  }
  
  const add = () => setRows([...rows, { host: '', port: '', username: '', password: '' }])
  const remove = (i: number) => setRows(rows.filter((_, idx) => idx !== i))
  const clearAll = () => setRows([])

  const applyBulk = () => {
    const lines = bulkText.split(/\r?\n/).map(s => s.trim()).filter(Boolean)
    const next: Row[] = [...rows]
    for (const ln of lines) {
      const m = ln.split(/[;,:\s]+/).filter(Boolean)
      if (m.length >= 2) {
        const [host, port, username = '', password = ''] = m
        next.push({ host, port, username, password })
      }
    }
    setRows(next)
    setBulkOpen(false)
    setBulkText('')
  }

  return (
    <Stack gap="lg">
      {/* Header */}
      <Paper p="md" radius="lg" withBorder>
        <Group justify="space-between">
          <Group gap="sm">
            <Box className="section-header-icon">
              <IconNetwork size={18} />
            </Box>
            <div>
              <Title order={4}>Прокси-серверы</Title>
              <Text size="sm" c="dimmed">HTTP/SOCKS5 прокси для ротации IP</Text>
            </div>
          </Group>
          
          <Group gap="xs">
            <Tooltip label="Массовый импорт">
              <Button 
                leftSection={<IconUpload size={18} />}
                variant="light" 
                onClick={() => setBulkOpen(true)}
              >
                Импорт
              </Button>
            </Tooltip>
            <Tooltip label="Добавить прокси">
              <Button 
                leftSection={<IconPlus size={18} />}
                variant="light"
                color="green"
                onClick={add}
              >
                Добавить
              </Button>
            </Tooltip>
            <Tooltip label="Сохранить изменения">
              <Button 
                leftSection={<IconDeviceFloppy size={18} />}
                onClick={save}
                loading={isSaving}
              >
                Сохранить
              </Button>
            </Tooltip>
            <Tooltip label="Обновить">
              <ActionIcon variant="light" size="lg" onClick={load}>
                <IconRefresh size={18} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Очистить всё">
              <ActionIcon variant="light" color="gray" size="lg" onClick={clearAll}>
                <IconTrash size={18} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
      </Paper>

      {/* Stats */}
      <Group gap="md">
        <Badge size="lg" variant="light" color="cyan" leftSection={<IconNetwork size={14} />}>
          {rows.length} прокси
        </Badge>
        <Badge size="lg" variant="light" color="green" leftSection={<IconWorldWww size={14} />}>
          Ротация IP включена
        </Badge>
      </Group>

      {/* Table */}
      <Paper p="md" radius="lg" withBorder>
        {rows.length === 0 ? (
          <Box className="empty-state">
            <IconNetwork size={48} style={{ opacity: 0.3 }} />
            <Text size="lg" fw={500} c="dimmed" mt="md">Нет прокси</Text>
            <Text size="sm" c="dimmed" mb="md">Добавьте прокси для обхода ограничений</Text>
            <Group gap="sm">
              <Button leftSection={<IconPlus size={18} />} onClick={add}>
                Добавить вручную
              </Button>
              <Button leftSection={<IconUpload size={18} />} variant="light" onClick={() => setBulkOpen(true)}>
                Массовый импорт
              </Button>
            </Group>
          </Box>
        ) : (
          <Table highlightOnHover withTableBorder withColumnBorders>
            <Table.Thead>
              <Table.Tr>
                <Table.Th style={{ width: 50 }}>#</Table.Th>
                <Table.Th>
                  <Group gap={6}>
                    <IconServer size={16} />
                    Хост
                  </Group>
                </Table.Th>
                <Table.Th style={{ width: 100 }}>Порт</Table.Th>
                <Table.Th>
                  <Group gap={6}>
                    <IconUser size={16} />
                    Логин
                  </Group>
                </Table.Th>
                <Table.Th>
                  <Group gap={6}>
                    <IconLock size={16} />
                    Пароль
                  </Group>
                </Table.Th>
                <Table.Th style={{ width: 80 }}>Действия</Table.Th>
              </Table.Tr>
            </Table.Thead>
            <Table.Tbody>
              {rows.map((r, i) => (
                <Table.Tr key={i}>
                  <Table.Td>
                    <Text size="sm" c="dimmed" fw={500}>{i + 1}</Text>
                  </Table.Td>
                  <Table.Td>
                    <TextInput 
                      value={r.host} 
                      placeholder="192.168.1.1"
                      onChange={(e) => { 
                        const v = [...rows]
                        v[i] = { ...v[i], host: e.currentTarget.value }
                        setRows(v) 
                      }}
                      styles={{ input: { border: 'none', background: 'transparent' } }}
                    />
                  </Table.Td>
                  <Table.Td>
                    <TextInput 
                      value={r.port} 
                      placeholder="8080"
                      onChange={(e) => { 
                        const v = [...rows]
                        v[i] = { ...v[i], port: e.currentTarget.value }
                        setRows(v) 
                      }}
                      styles={{ input: { border: 'none', background: 'transparent' } }}
                    />
                  </Table.Td>
                  <Table.Td>
                    <TextInput 
                      value={r.username || ''} 
                      placeholder="(опционально)"
                      onChange={(e) => { 
                        const v = [...rows]
                        v[i] = { ...v[i], username: e.currentTarget.value }
                        setRows(v) 
                      }}
                      styles={{ input: { border: 'none', background: 'transparent' } }}
                    />
                  </Table.Td>
                  <Table.Td>
                    <PasswordInput 
                      value={r.password || ''} 
                      placeholder="(опционально)"
                      onChange={(e) => { 
                        const v = [...rows]
                        v[i] = { ...v[i], password: e.currentTarget.value }
                        setRows(v) 
                      }}
                      styles={{ input: { border: 'none', background: 'transparent' } }}
                    />
                  </Table.Td>
                  <Table.Td>
                    <Tooltip label="Удалить">
                      <ActionIcon color="red" variant="light" onClick={() => remove(i)}>
                        <IconTrash size={16} />
                      </ActionIcon>
                    </Tooltip>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        )}
      </Paper>

      {/* Bulk Import Modal */}
      <Modal 
        opened={bulkOpen} 
        onClose={() => setBulkOpen(false)} 
        title={
          <Group gap="sm">
            <IconUpload size={20} />
            <Text fw={600}>Массовый импорт прокси</Text>
          </Group>
        }
        centered
        size="lg"
      >
        <Stack gap="md">
          <Alert icon={<IconAlertCircle size={16} />} color="blue" variant="light">
            Поддерживаемые форматы: <br />
            <code>host:port</code>, <code>host:port:login:pass</code>, <code>host;port;login;pass</code>
          </Alert>
          
          <Textarea
            placeholder="192.168.1.1:8080&#10;proxy.example.com:3128:user:pass&#10;10.0.0.1;1080;admin;secret"
            minRows={12}
            value={bulkText}
            onChange={(e) => setBulkText(e.currentTarget.value)}
            styles={{
              input: {
                fontFamily: 'var(--mantine-font-family-monospace)',
              }
            }}
          />
          
          <Group justify="flex-end">
            <Button variant="light" onClick={() => setBulkOpen(false)}>
              Отмена
            </Button>
            <Button 
              leftSection={<IconPlus size={18} />}
              onClick={applyBulk}
              disabled={!bulkText.trim()}
            >
              Добавить {bulkText.split('\n').filter(l => l.trim()).length} прокси
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  )
} 