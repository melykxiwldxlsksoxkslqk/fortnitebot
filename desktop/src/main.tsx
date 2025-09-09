import React from 'react'
import ReactDOM from 'react-dom/client'
import { MantineProvider } from '@mantine/core'
import { HashRouter, Routes, Route, Navigate } from 'react-router-dom'
import '@mantine/core/styles.css'
import Dashboard from './screens/Dashboard'
import Settings from './screens/Settings'
import Logs from './screens/Logs'
import Accounts from './screens/Accounts'
import Proxies from './screens/Proxies'
import Test from './screens/Test'
import AppLayout from './components/AppLayout'

function App() {
  return (
    <MantineProvider defaultColorScheme="dark">
      <HashRouter>
        <Routes>
          <Route path="/" element={<AppLayout />}> 
            <Route index element={<Navigate to="dashboard" replace />} />
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="settings" element={<Settings />} />
            <Route path="logs" element={<Logs />} />
            <Route path="accounts" element={<Accounts />} />
            <Route path="proxies" element={<Proxies />} />
            <Route path="test" element={<Test />} />
            <Route path="*" element={<Navigate to="dashboard" replace />} />
          </Route>
        </Routes>
      </HashRouter>
    </MantineProvider>
  )
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
) 