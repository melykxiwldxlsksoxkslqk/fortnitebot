import React from 'react'
import { useEffect, useState, useRef } from 'react'
import { 
  Button, 
  Group, 
  Title, 
  Textarea, 
  Text,
  Paper,
  Stack,
  Box,
  Badge,
  ActionIcon,
  Tooltip,
  SegmentedControl,
  TextInput
} from '@mantine/core'
import {
  IconFileText,
  IconCopy,
  IconTrash,
  IconSearch,
  IconFilter,
  IconDownload,
  IconArrowDown
} from '@tabler/icons-react'

declare global {
  interface Window { desktop: any }
}

export default function Logs() {
  const [text, setText] = useState('')
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [autoScroll, setAutoScroll] = useState(true)
  const areaRef = useRef<HTMLTextAreaElement | null>(null)

  useEffect(() => {
    let unsub: () => void = () => {}
    ;(async () => {
      try {
        const res = await window.desktop?.rpc?.('get_logs', null)
        if (res?.ok) setText(res.text || '')
      } catch {}
    })()
    try {
      if (window.desktop?.onStatus) {
        unsub = window.desktop.onStatus((msg: any) => setText((t: string) => {
          const timestamp = new Date().toLocaleTimeString('ru-RU', { 
            hour: '2-digit', 
            minute: '2-digit', 
            second: '2-digit' 
          })
          const next = t + `[${timestamp}] [${msg.login}] ${msg.text}\n`
          const lines = next.split('\n')
          if (lines.length > 5000) return lines.slice(-5000).join('\n')
          return next
        }))
      }
    } catch {}
    return () => { try { unsub() } catch {} }
  }, [])

  useEffect(() => {
    if (autoScroll && areaRef.current) {
      areaRef.current.scrollTop = areaRef.current.scrollHeight
    }
  }, [text, autoScroll])

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(filteredText)
    } catch {}
  }
  
  const clear = async () => {
    setText('')
    try { await window.desktop?.rpc?.('clear_logs', null) } catch {}
  }

  const download = () => {
    const blob = new Blob([filteredText], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `epicbot-logs-${new Date().toISOString().split('T')[0]}.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  const scrollToBottom = () => {
    if (areaRef.current) {
      areaRef.current.scrollTop = areaRef.current.scrollHeight
    }
  }

  // Filter and search
  const filteredText = text
    .split('\n')
    .filter(line => {
      if (!line.trim()) return false
      if (filter === 'errors' && !line.toLowerCase().includes('error') && !line.toLowerCase().includes('ошибка')) return false
      if (filter === 'system' && !line.includes('[system]')) return false
      if (search && !line.toLowerCase().includes(search.toLowerCase())) return false
      return true
    })
    .join('\n')

  const lineCount = text.split('\n').filter(l => l.trim()).length
  const errorCount = text.split('\n').filter(l => 
    l.toLowerCase().includes('error') || l.toLowerCase().includes('ошибка')
  ).length

  return (
    <Stack gap="lg">
      {/* Header */}
      <Paper p="md" radius="lg" withBorder>
        <Group justify="space-between">
          <Group gap="sm">
            <Box className="section-header-icon">
              <IconFileText size={18} />
            </Box>
            <div>
              <Title order={4}>Системные логи</Title>
              <Text size="sm" c="dimmed">Полная история событий бота</Text>
            </div>
          </Group>
          
          <Group gap="xs">
            <Tooltip label="Копировать логи">
              <ActionIcon variant="light" size="lg" onClick={copy}>
                <IconCopy size={18} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Скачать логи">
              <ActionIcon variant="light" size="lg" onClick={download}>
                <IconDownload size={18} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Прокрутить вниз">
              <ActionIcon variant="light" size="lg" onClick={scrollToBottom}>
                <IconArrowDown size={18} />
              </ActionIcon>
            </Tooltip>
            <Tooltip label="Очистить логи">
              <ActionIcon variant="light" color="red" size="lg" onClick={clear}>
                <IconTrash size={18} />
              </ActionIcon>
            </Tooltip>
          </Group>
        </Group>
      </Paper>

      {/* Stats & Filters */}
      <Group justify="space-between">
        <Group gap="md">
          <Badge size="lg" variant="light" color="blue" leftSection={<IconFileText size={14} />}>
            {lineCount} строк
          </Badge>
          {errorCount > 0 && (
            <Badge size="lg" variant="light" color="red">
              {errorCount} ошибок
            </Badge>
          )}
          <Badge 
            size="lg" 
            variant="dot" 
            color={autoScroll ? 'green' : 'gray'}
            style={{ cursor: 'pointer' }}
            onClick={() => setAutoScroll(!autoScroll)}
          >
            Автоскролл {autoScroll ? 'вкл' : 'выкл'}
          </Badge>
        </Group>
        
        <Group gap="sm">
          <TextInput
            placeholder="Поиск..."
            leftSection={<IconSearch size={16} />}
            value={search}
            onChange={(e) => setSearch(e.currentTarget.value)}
            size="sm"
            style={{ width: 200 }}
          />
          <SegmentedControl
            size="sm"
            value={filter}
            onChange={setFilter}
            data={[
              { label: 'Все', value: 'all' },
              { label: 'Ошибки', value: 'errors' },
              { label: 'Система', value: 'system' },
            ]}
          />
        </Group>
      </Group>

      {/* Logs Area */}
      <Paper p="md" radius="lg" withBorder style={{ flex: 1 }}>
        {filteredText ? (
          <Textarea
            ref={areaRef}
            value={filteredText}
            onChange={(e) => setText(e.currentTarget.value)}
            autosize
            minRows={25}
            maxRows={35}
            readOnly
            styles={{
              input: {
                fontFamily: 'var(--mantine-font-family-monospace)',
                fontSize: '0.8rem',
                lineHeight: 1.6,
                backgroundColor: 'var(--epic-bg-primary)',
                border: '1px solid var(--epic-border)',
              },
            }}
          />
        ) : (
          <Box className="empty-state">
            <IconFileText size={48} style={{ opacity: 0.3 }} />
            <Text size="lg" fw={500} c="dimmed" mt="md">Нет логов</Text>
            <Text size="sm" c="dimmed">
              {search || filter !== 'all' 
                ? 'Попробуйте изменить фильтры' 
                : 'Нажмите «Запустить» чтобы увидеть логи'}
            </Text>
          </Box>
        )}
      </Paper>
    </Stack>
  )
} 