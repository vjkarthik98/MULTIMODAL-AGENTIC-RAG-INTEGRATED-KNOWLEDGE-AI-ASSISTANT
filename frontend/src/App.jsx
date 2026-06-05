import { useState, useEffect, useRef } from 'react'
import LoginPage from './pages/LoginPage'
import ChatPage from './pages/ChatPage'
import { getMe } from './api/client'
import { ToastProvider } from './context/ToastContext'
import Toast from './components/Toast'

// Two tiny inline SVG favicons — static and pulsing
const FAVICON_STATIC  = `data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%238b5cf6'/><stop offset='1' stop-color='%233b82f6'/></linearGradient></defs><rect width='32' height='32' rx='8' fill='url(%23g)'/><text x='50%' y='54%' dominant-baseline='middle' text-anchor='middle' font-size='18' fill='white'>✦</text></svg>`
const FAVICON_LOADING = `data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%23a78bfa'/><stop offset='1' stop-color='%2360a5fa'/></linearGradient></defs><rect width='32' height='32' rx='8' fill='url(%23g)'/><text x='50%' y='54%' dominant-baseline='middle' text-anchor='middle' font-size='18' fill='white' opacity='0.7'>✦</text></svg>`

function setFavicon(href) {
  let link = document.querySelector("link[rel~='icon']")
  if (!link) { link = document.createElement('link'); link.rel = 'icon'; document.head.appendChild(link) }
  link.href = href
}

export default function App() {
  const [auth, setAuth]         = useState(null)
  const [checking, setChecking] = useState(true)
  const [pageKey, setPageKey]   = useState(0)   // incremented to re-trigger fade-in

  const [dark, setDark] = useState(
    () => !document.documentElement.classList.contains('light')
  )

  const toggleTheme = () => {
    setDark(prev => {
      const next = !prev
      document.documentElement.classList.toggle('light', !next)
      localStorage.setItem('magik_theme', next ? 'dark' : 'light')
      return next
    })
  }

  useEffect(() => {
    setFavicon(FAVICON_STATIC)
  }, [])

  useEffect(() => {
    const params     = new URLSearchParams(window.location.search)
    const oauthToken = params.get('magik_token')
    const oauthEmail = params.get('magik_email')

    if (oauthToken && oauthEmail) {
      localStorage.setItem('magik_token', oauthToken)
      localStorage.setItem('magik_email', oauthEmail)
      window.history.replaceState({}, '', '/')
      setAuth({ token: oauthToken, email: oauthEmail })
      setChecking(false)
      return
    }

    const storedToken = localStorage.getItem('magik_token')
    const storedEmail = localStorage.getItem('magik_email')
    if (storedToken && storedEmail) {
      getMe(storedToken)
        .then(() => setAuth({ token: storedToken, email: storedEmail }))
        .catch(() => {
          localStorage.removeItem('magik_token')
          localStorage.removeItem('magik_email')
        })
        .finally(() => setChecking(false))
    } else {
      setChecking(false)
    }
  }, [])

  const handleLogin = ({ token, email }) => {
    localStorage.setItem('magik_token', token)
    localStorage.setItem('magik_email', email)
    setAuth({ token, email })
    setPageKey(k => k + 1)
  }

  const handleLogout = () => {
    localStorage.removeItem('magik_token')
    localStorage.removeItem('magik_email')
    setAuth(null)
    setPageKey(k => k + 1)
    setFavicon(FAVICON_STATIC)
  }

  const handleStreamingChange = (isStreaming) => {
    setFavicon(isStreaming ? FAVICON_LOADING : FAVICON_STATIC)
  }

  if (checking) {
    return (
      <ToastProvider>
        <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--t-bg)' }}>
          <div className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin"
            style={{ borderColor: 'var(--t-accent)', borderTopColor: 'transparent' }} />
        </div>
      </ToastProvider>
    )
  }

  return (
    <ToastProvider>
      <div key={pageKey} className="page-enter" style={{ height: '100%' }}>
        {auth
          ? <ChatPage
              auth={auth}
              onLogout={handleLogout}
              dark={dark}
              onToggleTheme={toggleTheme}
              onStreamingChange={handleStreamingChange}
            />
          : <LoginPage
              onLogin={handleLogin}
              dark={dark}
              onToggleTheme={toggleTheme}
            />
        }
      </div>
      <Toast />
    </ToastProvider>
  )
}
