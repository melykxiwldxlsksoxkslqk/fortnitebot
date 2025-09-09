import React from 'react'
import { useEffect, useState } from 'react'
import { Button, Group, Title, Textarea, Text } from '@mantine/core'

declare global {
  interface Window { desktop: any }
}

export default function Logs() {
  const [text, setText] = useState('')
  const areaRef = React.useRef<HTMLTextAreaElement | null>(null)
  useEffect(() => {
    let unsub: () => void = () => {}
    // первичная загрузка буфера логов
    ;(async () => {
      try {
        const res = await window.desktop?.rpc?.('get_logs', null)
        if (res?.ok) setText(res.text || '')
      } catch {}
    })()
    // подписка на новые события
    try {
      if (window.desktop?.onStatus) {
        unsub = window.desktop.onStatus((msg: any) => setText((t: string) => {
          const next = t + `[${msg.login}] ${msg.text}\n`
          const lines = next.split('\n')
          if (lines.length > 5000) return lines.slice(-5000).join('\n')
          return next
        }))
      }
    } catch {}
    return () => { try { unsub() } catch {} }
  }, [])

  // автоскролл вниз при изменении текста
  useEffect(() => {
    const el = areaRef.current
    if (el) {
      el.scrollTop = el.scrollHeight
    }
  }, [text])
  const copy = async () => navigator.clipboard.writeText(text)
  const clear = async () => {
    setText('')
    try { await window.desktop?.rpc?.('clear_logs', null) } catch {}
  }
  return (
    <div>
      <Group justify="space-between" mb="md">
        <Title order={3}>Логи</Title>
        <Group>
          <Button onClick={copy}>Копировать</Button>
          <Button variant="light" onClick={clear}>Очистить</Button>
        </Group>
      </Group>
      {text ? (
        <Textarea ref={areaRef} autosize minRows={20} value={text} onChange={(e) => setText(e.currentTarget.value)} />
      ) : (
        <Text c="dimmed">Пока нет сообщений. Нажмите «Запустить ботов», чтобы увидеть поток логов. Системные события (загрузка UI и ошибки) тоже появятся здесь.</Text>
      )}
    </div>
  )
} 