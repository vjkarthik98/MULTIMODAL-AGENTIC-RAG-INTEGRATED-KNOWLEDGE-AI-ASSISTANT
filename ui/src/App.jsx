import { useState, useEffect, useRef } from 'react'
import LoginPage from './pages/LoginPage'
import LoginModal from './components/LoginModal'
import ChatPage from './pages/ChatPage'
import ForgotPasswordPage from './pages/ForgotPasswordPage'
import ResetPasswordPage from './pages/ResetPasswordPage'
import { getMe, refreshAccessToken, logout as apiLogout, createGuestSession, migrateGuestData } from './api/client'
import { ToastProvider } from './context/ToastContext'
import Toast from './components/Toast'
import ErrorBoundary from './components/ErrorBoundary'

// Brain+neuron glyph — two lobes + three neural nodes connected by axons
const BRAIN_CIRCUIT_PATHS = `<path d='M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.98-3 2.5 2.5 0 0 1-1.32-4.24 3 3 0 0 1 .34-5.58 2.5 2.5 0 0 1 2.98-3.19A2.5 2.5 0 0 1 9.5 2Z'/><path d='M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 1.98-3 2.5 2.5 0 0 0 1.32-4.24 3 3 0 0 0-.34-5.58 2.5 2.5 0 0 0-2.98-3.19A2.5 2.5 0 0 0 14.5 2Z'/><circle cx='9' cy='8.5' r='1.2' fill='white' stroke='none'/><circle cx='15' cy='8.5' r='1.2' fill='white' stroke='none'/><circle cx='12' cy='13.5' r='1.2' fill='white' stroke='none'/><line x1='9' y1='8.5' x2='15' y2='8.5'/><line x1='9' y1='8.5' x2='12' y2='13.5'/><line x1='15' y1='8.5' x2='12' y2='13.5'/>`

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
  const [showLoginModal, setShowLoginModal] = useState(false)

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
  const clearGuestAuth = () => {
    sessionStorage.removeItem('magik_guest_token')
    sessionStorage.removeItem('magik_guest_id')
    sessionStorage.removeItem('magik_guest_queries')
    sessionStorage.removeItem('magik_guest_uploads')
    sessionStorage.removeItem('magik_pending_guest_token')
  }

  useEffect(() => {
    const params       = new URLSearchParams(window.location.search)
    const oauthToken   = params.get('magik_token')
    const oauthRefresh = params.get('magik_refresh')
    const oauthEmail   = params.get('magik_email')

    if (oauthToken && oauthEmail) {
      persistAuth({ token: oauthToken, refreshToken: oauthRefresh, email: oauthEmail })
      window.history.replaceState({}, '', '/')
      // Google OAuth conversion path: if a pending guest token exists, migrate data
      const pendingGuestToken = sessionStorage.getItem('magik_pending_guest_token')
      if (pendingGuestToken) {
        clearGuestAuth()
        migrateGuestData(oauthToken, pendingGuestToken).catch(() => {
          // Migration failure is non-fatal — user still gets their real account
        })
      }
      setAuth({ token: oauthToken, refreshToken: oauthRefresh, email: oauthEmail, isGuest: false })
      setChecking(false)
      return
    }

    // Guest session recovery from sessionStorage (tab-isolated, expires on close)
    const guestToken   = sessionStorage.getItem('magik_guest_token')
    const guestId      = sessionStorage.getItem('magik_guest_id')
    const guestQueries = parseInt(sessionStorage.getItem('magik_guest_queries') || '5', 10)
    const guestUploads = parseInt(sessionStorage.getItem('magik_guest_uploads') || '2', 10)
    if (guestToken && guestId) {
      setAuth({
        token: guestToken, refreshToken: null, email: '',
        isGuest: true, guestUserId: guestId,
        queriesLeft: guestQueries, uploadsLeft: guestUploads,
      })
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
              // refresh token also expired/revoked — fall through to guest creation
            }
          }
          clearAuth()
          // Stored token invalid and refresh failed — silently become a guest
          _autoGuest()
        })
        .finally(() => setChecking(false))
    } else {
      // No stored credentials — chat IS the landing page; create a silent guest session
      _autoGuest().finally(() => setChecking(false))
    }
  }, [])

  // Creates a silent guest session and sets auth. Used on initial load when no
  // credentials are found, so the chat UI is shown immediately (ChatGPT-style).
  const _autoGuest = async () => {
    try {
      const data = await createGuestSession()
      sessionStorage.setItem('magik_guest_token',   data.access_token)
      sessionStorage.setItem('magik_guest_id',      data.guest_user_id)
      sessionStorage.setItem('magik_guest_queries', String(data.queries_left))
      sessionStorage.setItem('magik_guest_uploads', String(data.uploads_left))
      setAuth({
        token: data.access_token, refreshToken: null, email: '',
        isGuest: true, guestUserId: data.guest_user_id,
        queriesLeft: data.queries_left, uploadsLeft: data.uploads_left,
      })
    } catch {
      // Backend unreachable — leave auth null so LoginPage renders as fallback
    }
  }

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
    clearGuestAuth()
    persistAuth({ token, refreshToken, email })
    setAuth({ token, refreshToken, email, isGuest: false })
    setShowLoginModal(false)
    setPageKey(k => k + 1)
  }

  const handleGuestMode = async () => {
    try {
      const data = await createGuestSession()
      sessionStorage.setItem('magik_guest_token',   data.access_token)
      sessionStorage.setItem('magik_guest_id',      data.guest_user_id)
      sessionStorage.setItem('magik_guest_queries', String(data.queries_left))
      sessionStorage.setItem('magik_guest_uploads', String(data.uploads_left))
      setAuth({
        token: data.access_token, refreshToken: null, email: '',
        isGuest: true, guestUserId: data.guest_user_id,
        queriesLeft: data.queries_left, uploadsLeft: data.uploads_left,
      })
      setPageKey(k => k + 1)
    } catch (err) {
      console.error('Guest session creation failed:', err)
    }
  }

  // Called from ConversionModal after successful email/password conversion
  const handleGuestConvert = (realTokenData) => {
    clearGuestAuth()
    persistAuth({ token: realTokenData.access_token, refreshToken: realTokenData.refresh_token, email: realTokenData.email })
    setAuth({ token: realTokenData.access_token, refreshToken: realTokenData.refresh_token, email: realTokenData.email, isGuest: false })
    setPageKey(k => k + 1)
  }

  // Called from ConversionModal "Continue with Google" in guest mode
  const handleGuestGoogleConvert = () => {
    // Store guest token before redirect so OAuth callback can pick it up
    if (auth?.isGuest && auth?.token) {
      sessionStorage.setItem('magik_pending_guest_token', auth.token)
    }
    window.location.href = '/auth/google'
  }

  const handleLogout = () => {
    if (auth && !auth.isGuest) apiLogout(auth.token, auth.refreshToken)
    clearAuth()
    clearGuestAuth()
    setAuth(null)
    setPageKey(k => k + 1)
    setFavicon(FAVICON_STATIC)
  }

  // Update guest limits in auth state (called by ChatPage after each query/upload)
  const handleGuestLimitsUpdate = (updates) => {
    setAuth(prev => prev?.isGuest ? { ...prev, ...updates } : prev)
    if (updates.queriesLeft !== undefined)
      sessionStorage.setItem('magik_guest_queries', String(updates.queriesLeft))
    if (updates.uploadsLeft !== undefined)
      sessionStorage.setItem('magik_guest_uploads', String(updates.uploadsLeft))
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
            <>
              <ChatPage
                auth={auth}
                onLogout={handleLogout}
                dark={dark}
                onToggleTheme={toggleTheme}
                onStreamingChange={handleStreamingChange}
                onGuestConvert={handleGuestConvert}
                onGuestGoogleConvert={handleGuestGoogleConvert}
                onGuestLimitsUpdate={handleGuestLimitsUpdate}
                onShowLogin={() => setShowLoginModal(true)}
              />
              {showLoginModal && auth?.isGuest && (
                <LoginModal
                  onLogin={handleLogin}
                  onClose={() => setShowLoginModal(false)}
                  onForgotPassword={() => {
                    setShowLoginModal(false)
                    setPage('forgot')
                    setPageKey(k => k + 1)
                  }}
                />
              )}
            </>
          ) : (
            <LoginPage
              onLogin={handleLogin}
              onGuestMode={handleGuestMode}
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
