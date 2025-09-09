import React from 'react'
import { useEffect, useState } from 'react'
import { Button, Group, Title, TextInput, NumberInput, Switch, Select } from '@mantine/core'

declare global {
  interface Window { desktop: any }
}

export default function Settings() {
  const [s, setS] = useState<any>({ island_code: '', time_on_island_min: 15, headless: true, ingame_mode: 'passive', invert_bg: false })

  const load = async () => {
    const st = await window.desktop.rpc('get_settings', null)
    setS(st)
  }

  useEffect(() => { load() }, [])

  const save = async () => {
    await window.desktop.rpc('save_settings', s)
    await load() // показать сохранённые значения сразу
  }

  return (
    <div>
      <Group justify="space-between" mb="md">
        <Title order={3}>Настройки</Title>
        <Group>
          <Button onClick={save}>Сохранить</Button>
          <Button variant="light" onClick={load}>Обновить</Button>
        </Group>
      </Group>
      <Group grow>
        <TextInput label="Код острова" value={s.island_code} onChange={(e) => setS({ ...s, island_code: e.currentTarget.value })} />
        <NumberInput label="Время на острове (мин)" value={s.time_on_island_min} onChange={(v) => setS({ ...s, time_on_island_min: Number(v || 15) })} />
      </Group>
      <Group mt="md">
        <Switch label="Скрытый режим (Headless)" checked={!!s.headless} onChange={(e) => setS({ ...s, headless: e.currentTarget.checked })} />
        <Select label="Режим в игре" data={[{ value: 'passive', label: 'Пассивный' }, { value: 'rl', label: 'RL' }]} value={s.ingame_mode} onChange={(v) => setS({ ...s, ingame_mode: v })} />
        <Switch label="Инвертировать фон (чёрный↔белый)" checked={!!s.invert_bg} onChange={(e) => setS({ ...s, invert_bg: e.currentTarget.checked })} />
      </Group>
    </div>
  )
} 