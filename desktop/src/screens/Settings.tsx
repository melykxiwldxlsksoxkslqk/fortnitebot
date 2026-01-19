import React from 'react'
import { useEffect, useState, useRef } from 'react'
import { 
  Button, 
  Group, 
  Title, 
  TextInput, 
  NumberInput, 
  Switch, 
  Select,
  Paper,
  Stack,
  Box,
  Text,
  Divider,
  Alert,
  Badge,
  SimpleGrid,
  Tooltip
} from '@mantine/core'
import {
  IconSettings,
  IconDeviceFloppy,
  IconRefresh,
  IconMap,
  IconClock,
  IconEye,
  IconEyeOff,
  IconBrain,
  IconPalette,
  IconCheck,
  IconInfoCircle,
  IconRobot
} from '@tabler/icons-react'

declare global {
  interface Window { desktop: any }
}

export default function Settings() {
  const [s, setS] = useState<any>({ 
    island_code: '', 
    time_on_island_min: 15, 
    headless: true, 
    ingame_mode: 'passive', 
    invert_bg: false 
  })
  const [saved, setSaved] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const initializedRef = useRef(false)
  const debounceTimerRef = useRef<any>(null)

  const load = async () => {
    setIsLoading(true)
    try {
      const st = await window.desktop.rpc('get_settings', null)
      setS(st)
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const save = async () => {
    setIsLoading(true)
    try {
      await window.desktop.rpc('save_settings', s)
      await load()
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } finally {
      setIsLoading(false)
    }
  }

  // Auto-save (debounced)
  useEffect(() => {
    if (!initializedRef.current) {
      initializedRef.current = true
      return
    }
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current)
    debounceTimerRef.current = setTimeout(async () => {
      try { await window.desktop.rpc('save_settings', s) } catch {}
    }, 400)
    return () => { if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current) }
  }, [s])

  const modeOptions = [
    { value: 'passive', label: '🧘 Пассивный — AFK фарм' },
    { value: 'rl', label: '🤖 RL — Машинное обучение' }
  ]

  return (
    <Stack gap="lg">
      {/* Header */}
      <Paper p="md" radius="lg" withBorder>
        <Group justify="space-between">
          <Group gap="sm">
            <Box className="section-header-icon">
              <IconSettings size={18} />
            </Box>
            <div>
              <Title order={4}>Настройки бота</Title>
              <Text size="sm" c="dimmed">Параметры автоматизации и поведения</Text>
            </div>
          </Group>
          
          <Group gap="xs">
            {saved && (
              <Badge color="green" variant="light" leftSection={<IconCheck size={14} />}>
                Сохранено
              </Badge>
            )}
            <Tooltip label="Сохранить настройки">
              <Button 
                leftSection={<IconDeviceFloppy size={18} />}
                onClick={save}
                loading={isLoading}
              >
                Сохранить
              </Button>
            </Tooltip>
            <Tooltip label="Обновить">
              <Button variant="light" onClick={load} loading={isLoading}>
                <IconRefresh size={18} />
              </Button>
            </Tooltip>
          </Group>
        </Group>
      </Paper>

      {/* Island Settings */}
      <Paper p="lg" radius="lg" withBorder>
        <Group gap="sm" mb="lg">
          <IconMap size={20} style={{ color: 'var(--epic-accent)' }} />
          <Text fw={600}>Настройки острова</Text>
        </Group>
        
        <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
          <TextInput
            label="Код острова"
            description="Код креативного острова Fortnite"
            placeholder="1234-5678-9012"
            leftSection={<IconMap size={16} />}
            value={s.island_code}
            onChange={(e) => setS({ ...s, island_code: e.currentTarget.value })}
          />
          
          <NumberInput
            label="Время на острове"
            description="Минимальное время пребывания (минуты)"
            leftSection={<IconClock size={16} />}
            value={s.time_on_island_min}
            onChange={(v) => setS({ ...s, time_on_island_min: Number(v || 15) })}
            min={1}
            max={120}
            suffix=" мин"
          />
        </SimpleGrid>
      </Paper>

      {/* Bot Behavior */}
      <Paper p="lg" radius="lg" withBorder>
        <Group gap="sm" mb="lg">
          <IconRobot size={20} style={{ color: 'var(--epic-accent)' }} />
          <Text fw={600}>Поведение бота</Text>
        </Group>
        
        <Stack gap="md">
          <Select
            label="Режим в игре"
            description="Определяет поведение бота после входа на остров"
            leftSection={<IconBrain size={16} />}
            data={modeOptions}
            value={s.ingame_mode}
            onChange={(v) => setS({ ...s, ingame_mode: v })}
            allowDeselect={false}
          />
          
          {s.ingame_mode === 'rl' && (
            <Alert 
              icon={<IconInfoCircle size={16} />} 
              color="violet" 
              variant="light"
            >
              <Text size="sm">
                Режим RL использует обученную модель для принятия решений. 
                Убедитесь, что установлены зависимости из <code>requirements-ml.txt</code>
              </Text>
            </Alert>
          )}
        </Stack>
      </Paper>

      {/* Browser Settings */}
      <Paper p="lg" radius="lg" withBorder>
        <Group gap="sm" mb="lg">
          <IconEye size={20} style={{ color: 'var(--epic-accent)' }} />
          <Text fw={600}>Настройки браузера</Text>
        </Group>
        
        <Stack gap="md">
          <Switch
            label="Скрытый режим (Headless)"
            description="Запуск браузера без графического интерфейса"
            checked={!!s.headless}
            onChange={(e) => setS({ ...s, headless: e.currentTarget.checked })}
            thumbIcon={s.headless ? <IconEyeOff size={12} /> : <IconEye size={12} />}
            size="md"
          />
          
          <Alert 
            icon={<IconInfoCircle size={16} />} 
            color="blue" 
            variant="light"
          >
            <Text size="sm">
              Бот использует <strong>Camoufox</strong> — анти-детект браузер на основе Firefox 
              для обхода систем защиты.
            </Text>
          </Alert>
        </Stack>
      </Paper>

      {/* UI Settings */}
      <Paper p="lg" radius="lg" withBorder>
        <Group gap="sm" mb="lg">
          <IconPalette size={20} style={{ color: 'var(--epic-accent)' }} />
          <Text fw={600}>Интерфейс</Text>
        </Group>
        
        <Switch
          label="Инвертировать тему"
          description="Переключение между тёмной и светлой темой"
          checked={!!s.invert_bg}
          onChange={(e) => setS({ ...s, invert_bg: e.currentTarget.checked })}
          size="md"
        />
      </Paper>
    </Stack>
  )
} 