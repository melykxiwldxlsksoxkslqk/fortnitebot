import React from 'react'
import { useEffect, useState } from 'react'
import { Group, Table, Title, Badge, Text } from '@mantine/core'
import { Textarea, Button, Alert } from '@mantine/core'

declare global {
  interface Window { desktop: any }
}

export default function Dashboard() {
  const [status, setStatus] = useState<any>({ bots: [], threads: [], accounts: [], status: {}, settings: {} })
  const [logText, setLogText] = useState('')

  const refresh = async () => {
    try {
      const s = await window.desktop?.rpc?.('get_status', null)
      if (s) setStatus(s)
    } catch {}
  }

  useEffect(() => {
    let unsub: () => void = () => {}
    refresh()
    try {
      if (window.desktop?.onStatus) {
        // Обновляем таблицу статусов и одновременно накапливаем текст логов
        unsub = window.desktop.onStatus((msg: any) => {
          refresh()
          setLogText((t: string) => {
            const line = `[${msg.login}] ${msg.text}`
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
    // первичная загрузка буфера логов при заходе на дашборд
    (async () => {
      try {
        const res = await window.desktop?.rpc?.('get_logs', null)
        if (res?.ok) setLogText(res.text || '')
      } catch {}
    })()
  }, [])

  const copyLogs = async () => { try { await navigator.clipboard.writeText(logText) } catch {} }
  const clearLogs = async () => { try { await window.desktop?.rpc?.('clear_logs', null); setLogText('') } catch {} }

  const activeLogins: string[] = (status.active && Array.isArray(status.active) && status.active.length > 0)
    ? status.active
    : Array.from(new Set([
        ...Object.keys(status.status || {}),
        ...((status.threads || []) as string[]),
      ]))

  const colorFor = (text: string) => {
    const t = (text || '').toLowerCase()
    if (t.includes('ошибка') || t.includes('fatal') || t.includes('failed')) return 'red'
    if (t.includes('закрыт') || t.includes('останов')) return 'orange'
    if (t.includes('успех') || t.includes('готов') || t.includes('запущен')) return 'green'
    return 'blue'
  }

  return (
    <div>
      <Alert color="blue" variant="light" mb="sm">Экран «Управление» смонтирован</Alert>
      <Group justify="space-between" mb="sm">
        <Title order={3}>Статусы</Title>
      </Group>
      {activeLogins.length === 0 ? (
        <Text>Нет активных ботов. Нажмите «Запустить ботов», чтобы начать.</Text>
      ) : (
        <Table highlightOnHover withTableBorder withColumnBorders>
          <Table.Thead>
            <Table.Tr>
              <Table.Th>Логин</Table.Th>
              <Table.Th>Статус</Table.Th>
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {activeLogins.map((login: string) => {
              const text = (status.status?.[login]?.status) || (status.threads?.includes(login) ? 'Запуск...' : '')
              return (
                <Table.Tr key={login}>
                  <Table.Td>{login}</Table.Td>
                  <Table.Td>
                    <Badge color={colorFor(text)}>{text}</Badge>
                  </Table.Td>
                </Table.Tr>
              )
            })}
          </Table.Tbody>
        </Table>
      )}

      {/* Лайв‑логи прямо на дашборде */}
      <Group justify="space-between" mt="lg" mb="xs">
        <Title order={3}>Лайв‑логи</Title>
        <Group>
          <Button size="xs" variant="light" onClick={copyLogs}>Копировать</Button>
          <Button size="xs" variant="light" color="gray" onClick={clearLogs}>Очистить</Button>
        </Group>
      </Group>
      <Textarea autosize minRows={12} value={logText} onChange={(e) => setLogText(e.currentTarget.value)} />
    </div>
  )
} 