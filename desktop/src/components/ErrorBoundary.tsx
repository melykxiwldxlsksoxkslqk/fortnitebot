import React from 'react'

interface State { hasError: boolean; error?: any }

export default class ErrorBoundary extends React.Component<React.PropsWithChildren, State> {
  state: State = { hasError: false }

  static getDerivedStateFromError(error: any): State {
    return { hasError: true, error }
  }

  componentDidCatch(error: any, errorInfo: any) {
    try {
      // Попробуем передать в логи через IPC, если доступно
      // @ts-ignore
      window?.desktop?.onStatus?.((_msg: any) => {})
      // @ts-ignore
      window?.desktop?.rpc?.('renderer_error', { message: String(error), stack: String(errorInfo?.componentStack || error?.stack || '') })
    } catch {}
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 16 }}>
          <div style={{ fontWeight: 600, marginBottom: 8 }}>Ошибка рендера экрана</div>
          <pre style={{ whiteSpace: 'pre-wrap', opacity: 0.8 }}>{String(this.state.error)}</pre>
        </div>
      )
    }
    return this.props.children
  }
} 