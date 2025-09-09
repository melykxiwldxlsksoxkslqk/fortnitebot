import React, { useEffect } from 'react'

interface Props { invert?: boolean }

export default function PulsingBackground({ invert }: Props) {
  useEffect(() => {
    const root = document.documentElement
    const body = document.body
    body.classList.add('pulse-theme')

    const styleId = 'pulse-bg-css'
    const css = `
        :root {
          --bg-base: #0a0a0a;
          --bg-hi-1: rgba(255,255,255,0.10);
          --bg-hi-2: rgba(255,255,255,0.05);
          --bgx: 50%;
          --bgy: 50%;
          --ui-surface: rgba(255,255,255,0.06);
          --ui-border: rgba(255,255,255,0.12);
          --ui-glow: rgba(255,255,255,0.12);
          --ui-fg: rgba(255,255,255,0.92);
          --accent: 200, 160, 255;
        }

        .pulse-bg {
          position: fixed;
          inset: 0;
          z-index: -1; /* фон всегда ниже всего UI */
          pointer-events: none;
          background:
            radial-gradient(500px circle at var(--bgx) var(--bgy), var(--bg-hi-1) 0%, transparent 60%),
            radial-gradient(1100px circle at calc(var(--bgx) + 22%) calc(var(--bgy) + 12%), var(--bg-hi-2) 0%, transparent 70%),
            radial-gradient(1700px circle at calc(var(--bgx) - 26%) calc(var(--bgy) - 18%), var(--bg-hi-2) 0%, transparent 80%),
            var(--bg-base);
          animation: pulse 6s ease-in-out infinite, pulseMove 28s ease-in-out infinite;
        }

        /* Контейнеры */
        .pulse-theme .mantine-AppShell-root { background: transparent !important; }
        .pulse-theme .mantine-AppShell-main { position: relative; z-index: 1; background: transparent !important; color: var(--ui-fg); }
        .pulse-theme .mantine-AppShell-main * { color: var(--ui-fg) !important; }
        .pulse-theme .mantine-Title-root { color: var(--ui-fg) !important; }
        .pulse-theme .mantine-Text-root { color: var(--ui-fg) !important; }
        .pulse-theme .mantine-AppShell-header, .pulse-theme .mantine-AppShell-navbar { position: relative; z-index: 2; }

        @keyframes pulse { 0%, 100% { filter: brightness(1) contrast(1); } 50% { filter: brightness(1.08) contrast(1.03); } }
        @keyframes pulseMove { 0% { transform: translate3d(0,0,0); } 50% { transform: translate3d(0,-1.2%,0); } 100% { transform: translate3d(0,0,0); } }

        /* Header/Nav полупрозрачные панели */
        .pulse-theme .mantine-AppShell-header,
        .pulse-theme .mantine-AppShell-navbar {
          background: linear-gradient(180deg, rgba(0,0,0,0.24), rgba(0,0,0,0.10));
          border-color: var(--ui-border);
          border-bottom: 1px solid var(--ui-border);
          animation: softPulse 10s ease-in-out infinite;
          backdrop-filter: blur(6px) saturate(120%);
        }
        .pulse-theme .mantine-AppShell-navbar { border-right: 1px solid var(--ui-border); }

        /* Кнопки — «в цвет фона»: ghost/soft */
        .pulse-theme .mantine-Button-root {
          background-color: var(--ui-surface) !important;
          border: 1px solid var(--ui-border) !important;
          color: var(--ui-fg) !important;
          box-shadow: 0 0 0 0 rgba(var(--accent), 0.0), inset 0 0 0 0 rgba(255,255,255,0.02);
          transition: transform .12s ease, box-shadow .3s ease, border-color .2s ease, background-color .2s ease;
        }
        .pulse-theme .mantine-Button-root:hover {
          background-color: rgba(255,255,255,0.09) !important;
          border-color: rgba(255,255,255,0.18) !important;
          box-shadow: 0 0 30px 2px rgba(var(--accent), 0.10);
        }
        .pulse-theme .mantine-Button-root:active { transform: translateY(1px) scale(0.99); }
        .pulse-theme .mantine-Button-root[data-color="red"] { box-shadow: 0 0 0 0 rgba(255, 80, 80, 0.0); }
        .pulse-theme .mantine-Button-root[data-color="red"]:hover { box-shadow: 0 0 26px 2px rgba(255, 80, 80, 0.25); }

        /* Поля ввода и таблицы — мягкие панели */
        .pulse-theme input,
        .pulse-theme textarea,
        .pulse-theme .mantine-Select-input,
        .pulse-theme .mantine-NumberInput-input,
        .pulse-theme .mantine-TextInput-input,
        .pulse-theme .mantine-Textarea-input {
          background-color: var(--ui-surface) !important;
          border-color: var(--ui-border) !important;
          color: var(--ui-fg) !important;
          animation: softPulse 14s ease-in-out infinite;
        }
        .pulse-theme .mantine-Table-table { background: transparent; }
        .pulse-theme .mantine-Table-th,
        .pulse-theme .mantine-Table-td {
          background-color: rgba(255,255,255,0.03);
          border-color: var(--ui-border);
        }

        /* Навлинки и активные состояния */
        .pulse-theme .mantine-NavLink-root {
          background-color: transparent !important;
          color: var(--ui-fg) !important;
          border-radius: 10px;
          transition: background-color .2s ease, box-shadow .3s ease, border-color .2s ease;
        }
        .pulse-theme .mantine-NavLink-root:hover {
          background-color: rgba(255,255,255,0.05) !important;
        }
        .pulse-theme .mantine-NavLink-root[data-active="true"] {
          background-color: rgba(255,255,255,0.10) !important; /* тёмный мягкий тон вместо синего */
          border: 1px solid var(--ui-border) !important;
          box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06), 0 0 16px 2px rgba(var(--accent), 0.12);
          animation: glowPulse 8s ease-in-out infinite;
        }

        /* Фокус-эффекты */
        .pulse-theme *:focus-visible {
          outline: 2px solid rgba(var(--accent), 0.55) !important;
          outline-offset: 2px;
          box-shadow: 0 0 0 3px rgba(var(--accent), 0.20) !important;
        }

        @keyframes softPulse { 0%,100% { opacity: 1 } 50% { opacity: 0.97 } }
        @keyframes glowPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(var(--accent), 0.00) } 50% { box-shadow: 0 0 28px 6px rgba(var(--accent), 0.18) } }
      `

    let style = document.getElementById(styleId) as HTMLStyleElement | null
    if (!style) {
      style = document.createElement('style')
      style.id = styleId
      document.head.appendChild(style)
    }
    style.textContent = css

    // Палитра
    if (invert) {
      root.style.setProperty('--bg-base', '#ffffff')
      root.style.setProperty('--bg-hi-1', 'rgba(0,0,0,0.10)')
      root.style.setProperty('--bg-hi-2', 'rgba(0,0,0,0.06)')
      root.style.setProperty('--ui-surface', 'rgba(0,0,0,0.04)')
      root.style.setProperty('--ui-border', 'rgba(0,0,0,0.12)')
      root.style.setProperty('--ui-glow', 'rgba(0,0,0,0.15)')
      root.style.setProperty('--ui-fg', 'rgba(0,0,0,0.88)')
      root.style.setProperty('--accent', '120, 120, 255')
    } else {
      root.style.setProperty('--bg-base', '#0a0a0a')
      root.style.setProperty('--bg-hi-1', 'rgba(255,255,255,0.10)')
      root.style.setProperty('--bg-hi-2', 'rgba(255,255,255,0.05)')
      root.style.setProperty('--ui-surface', 'rgba(255,255,255,0.06)')
      root.style.setProperty('--ui-border', 'rgba(255,255,255,0.12)')
      root.style.setProperty('--ui-glow', 'rgba(255,255,255,0.14)')
      root.style.setProperty('--ui-fg', 'rgba(255,255,255,0.92)')
      root.style.setProperty('--accent', '200, 160, 255')
    }

    let raf = 0
    const onMove = (e: MouseEvent) => {
      if (raf) cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        const x = (e.clientX / window.innerWidth) * 100
        const y = (e.clientY / window.innerHeight) * 100
        root.style.setProperty('--bgx', `${x}%`)
        root.style.setProperty('--bgy', `${y}%`)
      })
    }
    window.addEventListener('mousemove', onMove)
    return () => {
      window.removeEventListener('mousemove', onMove)
      if (raf) cancelAnimationFrame(raf)
      body.classList.remove('pulse-theme')
    }
  }, [invert])

  return <div className="pulse-bg" />
} 