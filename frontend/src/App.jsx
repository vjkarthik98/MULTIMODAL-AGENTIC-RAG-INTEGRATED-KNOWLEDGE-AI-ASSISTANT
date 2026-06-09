import { useState, useEffect, useRef } from 'react'
import LoginPage from './pages/LoginPage'
import ChatPage from './pages/ChatPage'
import { getMe, refreshAccessToken, logout as apiLogout } from './api/client'
import { ToastProvider } from './context/ToastContext'
import Toast from './components/Toast'
import ErrorBoundary from './components/ErrorBoundary'

// Brain-circuit glyph (matches the in-app BrainCircuit logo), reused across both favicons
const BRAIN_CIRCUIT_PATHS = `<path d='M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z'/><path d='M9 13a4.5 4.5 0 0 0 3-4'/><path d='M6.003 5.125A3 3 0 0 0 6.401 6.5'/><path d='M3.477 10.896a4 4 0 0 1 .585-.396'/><path d='M6 18a4 4 0 0 1-1.967-.516'/><path d='M12 13h4'/><path d='M12 18h6a2 2 0 0 1 2 2v1'/><path d='M12 8h8'/><path d='M16 8V5a2 2 0 0 1 2-2'/><circle cx='16' cy='13' r='.5'/><circle cx='18' cy='3' r='.5'/><circle cx='20' cy='21' r='.5'/><circle cx='20' cy='8' r='.5'/>`

// Two tiny inline SVG favicons — static and pulsing
const FAVICON_STATIC  = `data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%238b5cf6'/><stop offset='1' stop-color='%233b82f6'/></linearGradient></defs><rect width='32' height='32' rx='8' fill='url(%23g)'/><g transform='translate(5,5) scale(0.92)' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>${BRAIN_CIRCUIT_PATHS}</g></svg>`
const FAVICON_LOADING = `data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><defs><linearGradient id='g' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='%23a78bfa'/><stop offset='1' stop-color='%2360a5fa'/></linearGradient></defs><rect width='32' height='32' rx='8' fill='url(%23g)'/><g transform='translate(5,5) scale(0.92)' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round' opacity='0.75'>${BRAIN_CIRCUIT_PATHS}</g></svg>`

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

  // Persistent sign-in: access tokens are short-lived (30 min) by design, but the
  // user should only be signed out when THEY choose to log out — not on a timer.
  // We keep them signed in by silently exchanging the long-lived refresh token
  // (stored alongside the access token) for a fresh pair, both proactively on an
  // interval and reactively whenever the access token turns out to be stale.
  const persistAuth = ({ token, refreshToken, email }) => {
    localStorage.setItem('magik_token', token)
    if (refreshToken) localStorage.setItem('magik_refresh', refreshToken)
    localStorage.setItem('magik_email', email)
  }
  const clearAuth = () => {
    localStorage.removeItem('magik_token')
    localStorage.removeItem('magik_refresh')
    localStorage.removeItem('magik_email')
  }

  useEffect(() => {
    const params       = new URLSearchParams(window.location.search)
    const oauthToken   = params.get('magik_token')
    const oauthRefresh = params.get('magik_refresh')
    const oauthEmail   = params.get('magik_email')

    if (oauthToken && oauthEmail) {
      persistAuth({ token: oauthToken, refreshToken: oauthRefresh, email: oauthEmail })
      window.history.replaceState({}, '', '/')
      setAuth({ token: oauthToken, refreshToken: oauthRefresh, email: oauthEmail })
      setChecking(false)
      return
    }

    const storedToken   = localStorage.getItem('magik_token')
    const storedRefresh = localStorage.getItem('magik_refresh')
    const storedEmail   = localStorage.getItem('magik_email')
    if (storedToken && storedEmail) {
      // Retry getMe up to 5× with 3s back-off to survive the ~40s uvicorn model-load
      // window on instance restart.  Only treat a genuine 401/403 as "logged out" —
      // network errors (backend still booting) keep the spinner until it's ready.
      const getMeWithRetry = async (token, attempts = 5) => {
        for (let i = 0; i < attempts; i++) {
          try {
            await getMe(token)
            return 'ok'
          } catch (err) {
            const isAuthError = err?.message?.includes('Session expired') || err?.message?.includes('401') || err?.message?.includes('403')
            if (isAuthError || i === attempts - 1) throw err
            await new Promise(r => setTimeout(r, 3000))
          }
        }
      }
      getMeWithRetry(storedToken)
        .then(() => setAuth({ token: storedToken, refreshToken: storedRefresh, email: storedEmail }))
        .catch(async () => {
          // Access token expired (e.g. the browser was closed for a while) —
          // silently renew via the refresh token before forcing a re-login.
          if (storedRefresh) {
            try {
              const data = await refreshAccessToken(storedRefresh)
              persistAuth({ token: data.access_token, refreshToken: data.refresh_token, email: storedEmail })
              setAuth({ token: data.access_token, refreshToken: data.refresh_token, email: storedEmail })
              return
            } catch {
              // refresh token also expired/revoked — only real recourse is sign-in
            }
          }
          clearAuth()
        })
        .finally(() => setChecking(false))
    } else {
      setChecking(false)
    }
  }, [])

  // Keep the access token fresh while the app is open, so it never has the
  // chance to expire mid-session — and catch up immediately on tab focus in
  // case the computer was asleep longer than the access token's lifetime.
  useEffect(() => {
    if (!auth?.refreshToken) return

    const silentRefresh = async () => {
      try {
        const data = await refreshAccessToken(auth.refreshToken)
        persistAuth({ token: data.access_token, refreshToken: data.refresh_token, email: auth.email })
        setAuth(prev => prev && { ...prev, token: data.access_token, refreshToken: data.refresh_token })
      } catch {
        // Refresh token expired or was revoked (e.g. logged out elsewhere) —
        // the session is truly over; return to the login page.
        clearAuth()
        setAuth(null)
        setPageKey(k => k + 1)
      }
    }

    const interval = setInterval(silentRefresh, 20 * 60 * 1000)  // well under the 30-min access-token TTL
    const onVisible = () => { if (document.visibilityState === 'visible') silentRefresh() }
    document.addEventListener('visibilitychange', onVisible)

    return () => {
      clearInterval(interval)
      document.removeEventListener('visibilitychange', onVisible)
    }
  }, [auth?.refreshToken])

  const handleLogin = ({ token, refreshToken, email }) => {
    persistAuth({ token, refreshToken, email })
    setAuth({ token, refreshToken, email })
    setPageKey(k => k + 1)
  }

  const handleLogout = () => {
    if (auth) apiLogout(auth.token, auth.refreshToken)
    clearAuth()
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
    <ErrorBoundary>
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
    </ErrorBoundary>
  )
}
