import { useState, useEffect, useRef } from 'react'
import LoginPage from './pages/LoginPage'
import ChatPage from './pages/ChatPage'
import ForgotPasswordPage from './pages/ForgotPasswordPage'
import ResetPasswordPage from './pages/ResetPasswordPage'
import { getMe, refreshAccessToken, logout as apiLogout, readCsrfCookie } from './api/client'
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
  // for a fresh pair, both proactively on an interval and reactively whenever
  // the access token turns out to be stale.
  //
  // Neither token is ever visible to this code: both live in httpOnly cookies
  // set by the server (app/auth/cookies.py) and ride along automatically on
  // every fetch via credentials:'include'. The only thing kept in React state
  // is the CSRF token (`auth.token` — see client.js), which is not a secret.
  const clearAuth = () => {
    // Nothing to remove — no auth material has ever been written to
    // localStorage. Kept as a named no-op so handleLogout reads the same as
    // it always has.
  }
  useEffect(() => {
    // Guard against React StrictMode's dev-only double-invocation of this
    // effect: without `cancelled`, two concurrent bootstrap chains can race,
    // and whichever setAuth() call lands LAST wins — non-deterministically
    // clobbering a real auth state with a stale one. Only the current
    // (non-cleaned-up) invocation may commit.
    let cancelled = false

    // Google OAuth lands back here as `/?oauth=1` — the server already set
    // the session cookies on the redirect response itself (no token material
    // ever touches this URL). Strip the marker immediately either way.
    const params = new URLSearchParams(window.location.search)
    if (params.has('oauth')) window.history.replaceState({}, '', '/')

    // Retry up to 5× with 3s back-off to survive the ~40s uvicorn model-load
    // window on instance restart. Only treat a genuine 401/403 as "logged
    // out" — network errors (backend still booting) keep the spinner until
    // it's ready.
    const getMeWithRetry = async (attempts = 5) => {
      for (let i = 0; i < attempts; i++) {
        try {
          return await getMe()
        } catch (err) {
          const isAuthError = err?.message?.includes('Session expired') || err?.message?.includes('401') || err?.message?.includes('403')
          if (isAuthError || i === attempts - 1) throw err
          await new Promise(r => setTimeout(r, 3000))
        }
      }
    }

    getMeWithRetry()
      .then((data) => {
        if (cancelled) return
        setAuth({ token: readCsrfCookie(), email: data.email })
      })
      .catch(async () => {
        if (cancelled) return
        // Access token expired (e.g. the browser was closed for a while) —
        // silently renew via the refresh cookie before forcing a re-login.
        try {
          await refreshAccessToken()
          if (cancelled) return
          const data = await getMe()
          if (cancelled) return
          setAuth({ token: readCsrfCookie(), email: data.email })
        } catch {
          // No valid session cookie at all, or refresh also failed —
          // genuinely logged out; leave auth null so LoginPage renders.
          if (!cancelled) setAuth(null)
        }
      })
      .finally(() => { if (!cancelled) setChecking(false) })

    return () => { cancelled = true }
  }, [])

  // Keep the access token fresh while the app is open, so it never has the
  // chance to expire mid-session — and catch up immediately on tab focus in
  // case the computer was asleep longer than the access token's lifetime.
  useEffect(() => {
    if (!auth) return

    const silentRefresh = async () => {
      if (refreshInFlight.current) return   // an overlapping call is already handling this
      refreshInFlight.current = true
      try {
        await refreshAccessToken()
        setAuth(prev => prev && { ...prev, token: readCsrfCookie() })
      } catch (err) {
        // Only a genuine 401 (server actively rejected the refresh cookie —
        // truly expired, revoked, or already consumed) means the session is
        // over. A network blip or 5xx leaves the refresh token untouched
        // server-side (rotation only happens on success), so the same cookie
        // is still good — just let the next interval/visibility tick retry it
        // instead of forcing the user back to the login page mid-upload.
        if (err?.status === 401) {
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
  }, [!!auth])

  const handleLogin = ({ email }) => {
    setAuth({ token: readCsrfCookie(), email })
    setPageKey(k => k + 1)
  }

  const handleLogout = () => {
    if (auth) apiLogout(auth.token)
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
