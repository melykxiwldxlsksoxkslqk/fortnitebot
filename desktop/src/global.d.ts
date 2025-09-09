export {}

declare global {
  interface DesktopAPI {
    rpc: (method: string, params?: any) => Promise<any>
    onStatus: (cb: (msg: any) => void) => () => void
  }
  interface Window {
    desktop: any
  }
} 