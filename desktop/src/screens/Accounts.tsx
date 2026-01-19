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
  IconUsers,
  IconPlus,
  IconTrash,
  IconDeviceFloppy,
  IconRefresh,
  IconUpload,
  IconUser,
  IconLock,
  IconAlertCircle
} from '@tabler/icons-react'

declare global {
  interface Window { desktop: any }
}

interface Row { login: string; password: string }

export default function Accounts() {
  const [rows, setRows] = useState<Row[]>([])
  const [bulkOpen, setBulkOpen] = useState(false)
  const [bulkText, setBulkText] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [showPasswords, setShowPasswords] = useState<{[key: number]: boolean}>({})

  const load = async () => {
    const r = await window.desktop.rpc('get_accounts', null)
    if (r?.ok) setRows(r.accounts || [])
  }
  useEffect(() => { load() }, [])

  const save = async () => {
    setIsSaving(true)
    try {
      const cleaned = rows.filter(r => (r.login || '').trim())
      await window.desktop.rpc('save_accounts', { accounts: cleaned })
      await load()
    } finally {
      setIsSaving(false)
    }
  }
  
  const add = () => setRows([...rows, { login: '', password: '' }])
  const remove = (i: number) => setRows(rows.filter((_, idx) => idx !== i))
  const clearAll = () => setRows([])

  const applyBulk = () => {
    const lines = bulkText.split(/\r?\n/).map(s => s.trim()).filter(Boolean)
    const next: Row[] = [...rows]
    for (const ln of lines) {
      const m = ln.split(/[;,:\s]+/).filter(Boolean)
      if (m.length >= 1) {
        const login = (m[0] || '').trim()
        const password = (m[1] || '').trim()
        if (login) next.push({ login, password })
      }
    }
    setRows(next)
    setBulkOpen(false)
    setBulkText('')
  }

  const togglePassword = (index: number) => {
    setShowPasswords(prev => ({ ...prev, [index]: !prev[index] }))
  }

  return (
    <Stack gap="lg">
      {/* Header */}
      <Paper p="md" radius="lg" withBorder>
        <Group justify="space-between">
          <Group gap="sm">
            <Box className="section-header-icon">
              <IconUsers size={18} />
            </Box>
            <div>
              <Title order={4}>Аккаунты Epic Games</Title>
              <Text size="sm" c="dimmed">Управление аккаунтами для ботов</Text>
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
            <Tooltip label="Добавить аккаунт">
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
        <Badge size="lg" variant="light" color="violet" leftSection={<IconUsers size={14} />}>
          {rows.length} аккаунт{rows.length === 1 ? '' : rows.length < 5 ? 'а' : 'ов'}
        </Badge>
        <Badge size="lg" variant="light" color="green" leftSection={<IconLock size={14} />}>
          Пароли зашифрованы (AES-128)
        </Badge>
      </Group>

      {/* Table */}
      <Paper p="md" radius="lg" withBorder>
        {rows.length === 0 ? (
          <Box className="empty-state">
            <IconUsers size={48} style={{ opacity: 0.3 }} />
            <Text size="lg" fw={500} c="dimmed" mt="md">Нет аккаунтов</Text>
            <Text size="sm" c="dimmed" mb="md">Добавьте аккаунты вручную или используйте массовый импорт</Text>
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
                    <IconUser size={16} />
                    Логин / Email
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
                      value={r.login} 
                      placeholder="email@example.com"
                      onChange={(e) => {
                        const v = [...rows]
                        v[i] = { ...v[i], login: e.currentTarget.value }
                        setRows(v)
                      }}
                      styles={{
                        input: { border: 'none', background: 'transparent' }
                      }}
                    />
                  </Table.Td>
                  <Table.Td>
                    <PasswordInput 
                      value={r.password}
                      placeholder="••••••••"
                      visible={showPasswords[i]}
                      onVisibilityChange={() => togglePassword(i)}
                      onChange={(e) => {
                        const v = [...rows]
                        v[i] = { ...v[i], password: e.currentTarget.value }
                        setRows(v)
                      }}
                      styles={{
                        input: { border: 'none', background: 'transparent' }
                      }}
                    />
                  </Table.Td>
                  <Table.Td>
                    <Tooltip label="Удалить">
                      <ActionIcon 
                        color="red" 
                        variant="light" 
                        onClick={() => remove(i)}
                      >
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
            <Text fw={600}>Массовый импорт аккаунтов</Text>
          </Group>
        }
        centered
        size="lg"
      >
        <Stack gap="md">
          <Alert icon={<IconAlertCircle size={16} />} color="blue" variant="light">
            Поддерживаемые форматы: <br />
            <code>login;password</code> или <code>login:password</code> или <code>login password</code>
          </Alert>
          
          <Textarea
            placeholder="user1@mail.com;password123&#10;user2@mail.com:pass456&#10;user3@mail.com mypassword"
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
              Добавить {bulkText.split('\n').filter(l => l.trim()).length} аккаунт(ов)
            </Button>
          </Group>
        </Stack>
      </Modal>
    </Stack>
  )
} 