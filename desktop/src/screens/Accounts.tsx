import React from 'react'
import { useEffect, useState } from 'react'
import { Button, Group, Title, Table, TextInput, Textarea, Modal } from '@mantine/core'

declare global {
  interface Window { desktop: any }
}

interface Row { login: string; password: string }

export default function Accounts() {
  const [rows, setRows] = useState<Row[]>([])
  const [bulkOpen, setBulkOpen] = useState(false)
  const [bulkText, setBulkText] = useState('')

  const load = async () => {
    const r = await window.desktop.rpc('get_accounts', null)
    if (r?.ok) setRows(r.accounts || [])
  }
  useEffect(() => { load() }, [])

  const save = async () => {
    // фильтр пустых строк
    const cleaned = rows.filter(r => (r.login || '').trim())
    const res = await window.desktop.rpc('save_accounts', { accounts: cleaned })
    await load()
  }
  const add = () => setRows([...rows, { login: '', password: '' }])
  const remove = (i: number) => setRows(rows.filter((_, idx) => idx !== i))
  const clearAll = () => setRows([])

  const applyBulk = () => {
    // формат: login;password или login:password или "login password" по одному на строку
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
    setBulkOpen(false); setBulkText('')
  }

  return (
    <div>
      <Group justify="space-between" mb="md">
        <Title order={3}>Аккаунты</Title>
        <Group>
          <Button onClick={() => setBulkOpen(true)}>Массовый ввод</Button>
          <Button onClick={add}>Добавить</Button>
          <Button onClick={save}>Сохранить</Button>
          <Button variant="light" onClick={load}>Обновить</Button>
          <Button color="gray" variant="light" onClick={clearAll}>Очистить</Button>
        </Group>
      </Group>

      <Table withTableBorder withColumnBorders>
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Логин</Table.Th>
            <Table.Th>Пароль</Table.Th>
            <Table.Th w={120}></Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {rows.map((r, i) => (
            <Table.Tr key={i}>
              <Table.Td>
                <TextInput value={r.login} onChange={(e) => {
                  const v = [...rows]; v[i] = { ...v[i], login: e.currentTarget.value }; setRows(v)
                }} />
              </Table.Td>
              <Table.Td>
                <TextInput value={r.password} onChange={(e) => {
                  const v = [...rows]; v[i] = { ...v[i], password: e.currentTarget.value }; setRows(v)
                }} />
              </Table.Td>
              <Table.Td>
                <Button color="red" variant="light" onClick={() => remove(i)}>Удалить</Button>
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>

      <Modal opened={bulkOpen} onClose={() => setBulkOpen(false)} title="Массовый ввод аккаунтов" centered>
        <Textarea
          placeholder="login1;password1\nlogin2;password2\n..."
          minRows={10}
          value={bulkText}
          onChange={(e) => setBulkText(e.currentTarget.value)}
        />
        <Group justify="flex-end" mt="md">
          <Button onClick={applyBulk}>Добавить</Button>
        </Group>
      </Modal>
    </div>
  )
} 