import { useState, useEffect, useRef } from 'react'
import LoginPage from './pages/LoginPage'
import ChatPage from './pages/ChatPage'
import ForgotPasswordPage from './pages/ForgotPasswordPage'
import ResetPasswordPage from './pages/ResetPasswordPage'
import { getMe, refreshAccessToken, logout as apiLogout } from './api/client'
import { ToastProvider } from './context/ToastContext'
import Toast from './components/Toast'
import ErrorBoundary from './components/ErrorBoundary'

// Static PNG favicon — used for both idle and loading states.
// Loading state dims the icon slightly via CSS filter.
const FAVICON_STATIC  = '/logo.png'
const FAVICON_LOADING = '/logo.png'

function setFavicon(href, loading = false) {
  let link = document.querySelector("link[rel~='icon']")
  if (!link) { link = document.createElement('link'); link.rel = 'icon'; document.head.appendChild(link) }
  link.type = 'image/png'
  link.href = href
}

export default function App() {
  const [auth, setAuth]         = useState(null)
  const [checking, setChecking] = useState(
    // Skip the auth-check spinner on reset/forgot-password pages — no session needed
    () => !['/reset-password', '/forgot-password'].includes(window.location.pathname)
  )
  const [pageKey, setPageKey]   = useState(0)   // incremented to re-trigger fade-in
  // Read URL synchronously at init — before any effect or render runs
  const [resetToken] = useState(() => {
    if (window.location.pathname === '/reset-password')
      return new URLSearchParams(window.location.search).get('token') || ''
    return ''
  })
  const [page, setPage] = useState(() => {
    const p = window.location.pathname
    if (p === '/reset-password') { window.history.replaceState({}, '', '/reset-password'); return 'reset' }
    if (p === '/forgot-password') { window.history.replaceState({}, '', '/'); return 'forgot' }
    return 'main'
  })

  const [dark, setDark] = useState(
    () => !document.documentElement.classList.contains('light')
  )
  // Guards against overlapping silentRefresh() calls (the 20-min interval and
  // the visibilitychange listener can both fire within moments of each other,
  // e.g. tabbing away and back during a long upload) — refresh tokens rotate
  // single-use server-side, so a second concurrent call would legitimately be
  // rejected as reuse of an already-consumed token and wrongly end the session.
  const refreshInFlight = useRef(false)

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
    // device token intentionally kept — it lets this browser skip OTP on next login
  }
  useEffect(() => {
    // Guard against React StrictMode's dev-only double-invocation of this
    // effect: without `cancelled`, two concurrent getMeWithRetry chains can
    // race, and whichever setAuth() call lands LAST wins — non-
    // deterministically clobbering a real auth state with a stale one. Only
    // the current (non-cleaned-up) invocation may commit.
    let cancelled = false

    const params       = new URLSearchParams(window.location.search)
    const oauthToken   = params.get('magik_token')
    const oauthRefresh = params.get('magik_refresh')
    const oauthEmail   = params.get('magik_email')

    if (oauthToken && oauthEmail) {
      persistAuth({ token: oauthToken, refreshToken: oauthRefresh, email: oauthEmail })
      window.history.replaceState({}, '', '/')
      if (cancelled) return
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
        .then(async () => {
          if (cancelled) return
          // Token is valid right now but may be near its 30-min expiry.
          // Always exchange it for a fresh pair so the first upload/query
          // never hits a just-expired token before the 20-min interval fires.
          if (storedRefresh) {
            try {
              const data = await refreshAccessToken(storedRefresh)
              if (cancelled) return
              persistAuth({ token: data.access_token, refreshToken: data.refresh_token, email: storedEmail })
              setAuth({ token: data.access_token, refreshToken: data.refresh_token, email: storedEmail })
              return
            } catch {
              // Refresh failed (edge case: Redis down, multi-tab race).
              // The stored token is still valid — use it as-is.
            }
          }
          if (cancelled) return
          setAuth({ token: storedToken, refreshToken: storedRefresh, email: storedEmail })
        })
        .catch(async () => {
          if (cancelled) return
          // Access token expired (e.g. the browser was closed for a while) —
          // silently renew via the refresh token before forcing a re-login.
          if (storedRefresh) {
            try {
              const data = await refreshAccessToken(storedRefresh)
              if (cancelled) return
              persistAuth({ token: data.access_token, refreshToken: data.refresh_token, email: storedEmail })
              setAuth({ token: data.access_token, refreshToken: data.refresh_token, email: storedEmail })
              return
            } catch {
              // refresh token also expired/revoked — fall through to logout
            }
          }
          if (cancelled) return
          clearAuth()
          // Stored token invalid and refresh also failed — genuinely logged
          // out; leave auth null so LoginPage renders.
        })
        .finally(() => { if (!cancelled) setChecking(false) })
    } else {
      // No stored credentials — leave auth null so LoginPage renders.
      setChecking(false)
    }

    return () => { cancelled = true }
  }, [])

  // Keep the access token fresh while the app is open, so it never has the
  // chance to expire mid-session — and catch up immediately on tab focus in
  // case the computer was asleep longer than the access token's lifetime.
  useEffect(() => {
    if (!auth?.refreshToken) return

    const silentRefresh = async () => {
      if (refreshInFlight.current) return   // an overlapping call is already handling this
      refreshInFlight.current = true
      try {
        const data = await refreshAccessToken(auth.refreshToken)
        persistAuth({ token: data.access_token, refreshToken: data.refresh_token, email: auth.email })
        setAuth(prev => prev && { ...prev, token: data.access_token, refreshToken: data.refresh_token })
      } catch (err) {
        // Only a genuine 401 (server actively rejected the refresh token —
        // truly expired, revoked, or already consumed) means the session is
        // over. A network blip or 5xx leaves the refresh token untouched
        // server-side (rotation only happens on success), so the same token
        // is still good — just let the next interval/visibility tick retry it
        // instead of forcing the user back to the login page mid-upload.
        if (err?.status === 401) {
          clearAuth()
          setAuth(null)
          setPageKey(k => k + 1)
        }
      } finally {
        refreshInFlight.current = false
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
          {page === 'forgot' ? (
            <ForgotPasswordPage onBack={() => { setPage('main'); setPageKey(k => k + 1) }} />
          ) : page === 'reset' ? (
            <ResetPasswordPage
              token={resetToken}
              onSuccess={() => {
                setPage('main')
                window.history.replaceState({}, '', '/')
                setPageKey(k => k + 1)
              }}
            />
          ) : auth ? (
            <ChatPage
              auth={auth}
              onLogout={handleLogout}
              dark={dark}
              onToggleTheme={toggleTheme}
              onStreamingChange={handleStreamingChange}
            />
          ) : (
            <LoginPage
              onLogin={handleLogin}
              dark={dark}
              onToggleTheme={toggleTheme}
              onForgotPassword={() => { setPage('forgot'); setPageKey(k => k + 1) }}
            />
          )}
        </div>
        <Toast />
      </ToastProvider>
    </ErrorBoundary>
  )
}
