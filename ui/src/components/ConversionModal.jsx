import { useState } from 'react'
import { convertGuestToUser } from '../api/client'

const HEADLINES = {
  query_limit:  "You've used all 5 free queries",
  upload_limit: "You've used your 2 free uploads",
  voluntary:    'Save your work forever',
}

const SUBTEXTS = {
  query_limit:  "Create a free account to keep chatting with your documents — your uploaded files carry over.",
  upload_limit: "Create a free account to upload more files and continue your research.",
  voluntary:    "Sign up to save your documents, chat history, and get unlimited queries.",
}

export default function ConversionModal({ guestToken, trigger = 'voluntary', onConvert, onGoogleConvert, onClose }) {
  const [email, setEmail]             = useState('')
  const [password, setPassword]       = useState('')
  const [confirmPass, setConfirmPass] = useState('')
  const [showPass, setShowPass]       = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState('')

  const headline = HEADLINES[trigger] || HEADLINES.voluntary
  const subtext  = SUBTEXTS[trigger]  || SUBTEXTS.voluntary

  const handleGoogle = () => {
    if (onGoogleConvert) onGoogleConvert()
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!email || !password || !confirmPass) return
    if (password !== confirmPass) {
      setError('Passwords do not match.')
      return
    }
    setLoading(true)
    setError('')
    try {
      const data = await convertGuestToUser(guestToken, { email, password })
      onConvert(data)
    } catch (err) {
      if (err.status === 409) {
        setError('An account with this email already exists. Sign in instead.')
      } else {
        setError(err.message || 'Something went wrong. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    /* Backdrop */
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      style={{ background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)' }}
      onClick={e => { if (e.target === e.currentTarget && onClose) onClose() }}
    >
      {/* Modal card */}
      <div
        className="w-full max-w-sm rounded-2xl shadow-2xl"
        style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}
      >
        {/* Header */}
        <div className="px-6 pt-6 pb-4">
          {/* Zap icon */}
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center mb-4"
            style={{ background: 'color-mix(in srgb, var(--t-accent) 18%, transparent)' }}
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--t-accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
          </div>

          <h2 className="text-lg font-bold mb-1" style={{ color: 'var(--t-tx1)' }}>
            {headline}
          </h2>
          <p className="text-sm" style={{ color: 'var(--t-tx3)' }}>
            {subtext}
          </p>
        </div>

        {/* Body */}
        <div className="px-6 pb-6 space-y-3">
          {/* Google CTA */}
          <button
            type="button"
            onClick={handleGoogle}
            className="w-full flex items-center justify-center gap-3 rounded-xl py-3 px-4 text-sm font-medium transition-colors"
            style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd4)', color: 'var(--t-tx1)' }}
            onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
            onMouseLeave={e => e.currentTarget.style.background = 'var(--t-inp)'}
          >
            <GoogleG />
            Continue with Google
          </button>

          {/* Divider */}
          <div className="flex items-center gap-3">
            <div className="flex-1 h-px" style={{ background: 'var(--t-bd2)' }} />
            <span className="text-xs font-medium" style={{ color: 'var(--t-ph)' }}>OR</span>
            <div className="flex-1 h-px" style={{ background: 'var(--t-bd2)' }} />
          </div>

          {/* Email / password form */}
          <form onSubmit={handleSubmit} className="space-y-3">
            <input
              type="email"
              placeholder="Email address"
              value={email}
              onChange={e => setEmail(e.target.value)}
              required
              autoComplete="email"
              className="w-full rounded-xl px-4 py-3 text-sm outline-none transition-colors"
              style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
            />

            {/* Password */}
            <div className="relative">
              <input
                type={showPass ? 'text' : 'password'}
                placeholder="Password (8+ characters)"
                value={password}
                onChange={e => setPassword(e.target.value)}
                required
                autoComplete="new-password"
                className="w-full rounded-xl px-4 py-3 pr-10 text-sm outline-none transition-colors"
                style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
              />
              <button
                type="button"
                onClick={() => setShowPass(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2"
                style={{ color: 'var(--t-ph)' }}
                tabIndex={-1}
              >
                {/* eye-open = currently visible; eye-slash = currently hidden */}
                {showPass
                  ? <EyeOpen />
                  : <EyeSlash />
                }
              </button>
            </div>

            {/* Confirm password */}
            <div className="relative">
              <input
                type={showConfirm ? 'text' : 'password'}
                placeholder="Confirm password"
                value={confirmPass}
                onChange={e => setConfirmPass(e.target.value)}
                required
                autoComplete="new-password"
                className="w-full rounded-xl px-4 py-3 pr-10 text-sm outline-none transition-colors"
                style={{
                  background: 'var(--t-inp)',
                  border: `1px solid ${confirmPass && password !== confirmPass ? '#ef4444' : 'var(--t-bd2)'}`,
                  color: 'var(--t-tx1)',
                }}
              />
              <button
                type="button"
                onClick={() => setShowConfirm(v => !v)}
                className="absolute right-3 top-1/2 -translate-y-1/2"
                style={{ color: 'var(--t-ph)' }}
                tabIndex={-1}
              >
                {showConfirm ? <EyeOpen /> : <EyeSlash />}
              </button>
            </div>

            {/* Inline mismatch hint */}
            {confirmPass && password !== confirmPass && (
              <p className="text-xs -mt-1 px-1" style={{ color: '#ef4444' }}>
                Passwords do not match
              </p>
            )}

            {error && (
              <p className="text-xs px-3 py-2 rounded-lg" style={{ color: '#ef4444', background: 'rgba(239,68,68,0.08)' }}>
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={loading || !email || !password || !confirmPass || password !== confirmPass}
              className="w-full rounded-xl py-3 text-sm font-semibold transition-colors disabled:opacity-50"
              style={{ background: 'var(--t-accent)', color: 'white' }}
            >
              {loading ? 'Creating account…' : 'Create free account'}
            </button>
          </form>

          {/* Dismiss link */}
          {onClose && (
            <p className="text-center text-xs" style={{ color: 'var(--t-tx4)' }}>
              <button
                type="button"
                onClick={onClose}
                className="underline hover:no-underline transition-colors"
                style={{ color: 'var(--t-tx4)' }}
              >
                Continue as guest
              </button>
              {' '}— {trigger === 'voluntary' ? 'you have' : 'no'} {trigger === 'voluntary' ? `${window._guestQueriesLeft ?? '?'} queries left` : 'remaining queries'}
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function EyeOpen() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
      <circle cx="12" cy="12" r="3"/>
    </svg>
  )
}

function EyeSlash() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/>
      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/>
      <line x1="1" y1="1" x2="23" y2="23"/>
    </svg>
  )
}

function GoogleG() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">
      <path d="M17.64 9.205c0-.639-.057-1.252-.164-1.841H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>
      <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>
      <path d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
      <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z" fill="#EA4335"/>
    </svg>
  )
}
