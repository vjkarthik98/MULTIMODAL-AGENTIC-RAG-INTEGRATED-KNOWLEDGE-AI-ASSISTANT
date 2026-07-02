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
    // Guard against React StrictMode's dev-only double-invocation of this
    // effect: without `cancelled`, two concurrent getMeWithRetry/_autoGuest
    // chains can race, and whichever setAuth() call lands LAST wins — non-
    // deterministically producing a guest session even though the other,
    // now-stale invocation already authenticated as the real stored user
    // (and its Sidebar/ChatPage mount already fetched that user's real KB
    // files and chat history, which then linger on screen under the newer,
    // guest `auth`). Only the current (non-cleaned-up) invocation may commit.
    let cancelled = false

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
      if (cancelled) return
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
      if (cancelled) return
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
              // refresh token also expired/revoked — fall through to guest creation
            }
          }
          if (cancelled) return
          clearAuth()
          // Stored token invalid and refresh failed — silently become a guest
          _autoGuest(() => cancelled)
        })
        .finally(() => { if (!cancelled) setChecking(false) })
    } else {
      // No stored credentials — chat IS the landing page; create a silent guest session
      _autoGuest(() => cancelled).finally(() => { if (!cancelled) setChecking(false) })
    }

    return () => { cancelled = true }
  }, [])

  // Creates a silent guest session and sets auth. Used on initial load when no
  // credentials are found, so the chat UI is shown immediately (ChatGPT-style).
  // `isCancelled` (optional) lets the caller's effect veto a stale result —
  // see the StrictMode race-guard comment on the auth-check effect above.
  const _autoGuest = async (isCancelled) => {
    try {
      const data = await createGuestSession()
      if (isCancelled?.()) return
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
