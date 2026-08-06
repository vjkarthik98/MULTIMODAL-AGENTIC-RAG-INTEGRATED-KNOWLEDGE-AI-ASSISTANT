import { useState, useEffect, useRef } from 'react'
import { Eye, EyeOff, Loader2, Sun, Moon, ShieldCheck, RotateCcw } from 'lucide-react'
import { login, register, verifyOtp, resendOtp } from '../api/client'

const GoogleG = () => (
  <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
    <path d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 0 0 2.38-5.88c0-.57-.05-.66-.15-1.18" fill="#4285F4"/>
    <path d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2a4.8 4.8 0 0 1-7.18-2.54H1.83v2.07A8 8 0 0 0 8.98 17" fill="#34A853"/>
    <path d="M4.5 10.52a4.8 4.8 0 0 1 0-3.04V5.41H1.83a8 8 0 0 0 0 7.18z" fill="#FBBC05"/>
    <path d="M8.98 4.18c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 0 0 1.83 5.4L4.5 7.49a4.77 4.77 0 0 1 4.48-3.3" fill="#EA4335"/>
  </svg>
)

export default function LoginPage({ onLogin, dark, onToggleTheme, onForgotPassword }) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => { requestAnimationFrame(() => setMounted(true)) }, [])

  const [mode, setMode]         = useState('login')
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm]   = useState('')
  const [showPass, setShowPass]       = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [error, setError]       = useState('')
  const [success, setSuccess]   = useState('')
  const [loading, setLoading]   = useState(false)

  // OTP step state
  const [otpStep, setOtpStep]     = useState(false)   // true = show OTP form
  const [otpToken, setOtpToken]   = useState('')       // short-lived challenge token
  const [otpCode, setOtpCode]     = useState(['', '', '', '', '', ''])
  const [resendCooldown, setResendCooldown] = useState(0)  // seconds remaining
  const [resending, setResending] = useState(false)
  const otpRefs = useRef([])
  const cooldownRef = useRef(null)

  const startCooldown = (seconds = 60) => {
    setResendCooldown(seconds)
    clearInterval(cooldownRef.current)
    cooldownRef.current = setInterval(() => {
      setResendCooldown(prev => {
        if (prev <= 1) { clearInterval(cooldownRef.current); return 0 }
        return prev - 1
      })
    }, 1000)
  }
  useEffect(() => () => clearInterval(cooldownRef.current), [])

  const switchMode = (m) => { setMode(m); setError(''); setSuccess(''); setOtpStep(false); setShowPass(false); setShowConfirm(false) }

  // Handle 6 individual OTP digit boxes
  const handleOtpChange = (idx, val) => {
    const digit = val.replace(/\D/g, '').slice(-1)
    const next = [...otpCode]
    next[idx] = digit
    setOtpCode(next)
    if (digit && idx < 5) otpRefs.current[idx + 1]?.focus()
  }
  const handleOtpKeyDown = (idx, e) => {
    if (e.key === 'Backspace' && !otpCode[idx] && idx > 0) {
      otpRefs.current[idx - 1]?.focus()
    }
    if (e.key === 'ArrowLeft' && idx > 0) otpRefs.current[idx - 1]?.focus()
    if (e.key === 'ArrowRight' && idx < 5) otpRefs.current[idx + 1]?.focus()
  }
  const handleOtpPaste = (e) => {
    const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6)
    if (pasted.length === 6) {
      setOtpCode(pasted.split(''))
      otpRefs.current[5]?.focus()
    }
    e.preventDefault()
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(''); setSuccess('')

    if (otpStep) {
      // Step 2: verify OTP
      const code = otpCode.join('')
      if (code.length !== 6) { setError('Enter all 6 digits'); return }
      setLoading(true)
      try {
        // The trusted-device token (if issued) comes back as an httpOnly
        // cookie set on this response — nothing for the client to store.
        await verifyOtp(otpToken, code)
        onLogin({ email })
      } catch (err) {
        setError(err.message)
        setOtpCode(['', '', '', '', '', ''])
        otpRefs.current[0]?.focus()
      } finally {
        setLoading(false)
      }
      return
    }

    if (mode === 'register' && password !== confirm) {
      setError('Passwords do not match')
      return
    }
    setLoading(true)
    try {
      let data
      if (mode === 'register') {
        // register() creates account as inactive + sends OTP → returns {otp_required, otp_token}
        data = await register(email, password)
      } else {
        data = await login(email, password)
      }
      if (data.otp_required) {
        setOtpToken(data.otp_token)
        setOtpStep(true)
        setError('')
        setSuccess('')
        startCooldown()
        setTimeout(() => otpRefs.current[0]?.focus(), 100)
      } else {
        // Trusted device — server skipped OTP; session cookies are already set.
        onLogin({ email })
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleResend = async () => {
    if (resendCooldown > 0 || resending) return
    setResending(true); setError('')
    try {
      // Dedicated resend endpoint — reuses the existing otp_token (works
      // identically whether the challenge came from login or register)
      // instead of re-running the original login/register call, which broke
      // resend during registration (the account is still inactive at that
      // point, and login on an inactive account is correctly rejected).
      const data = await resendOtp(otpToken)
      setOtpCode(['', '', '', '', '', ''])
      otpRefs.current[0]?.focus()
      startCooldown(data.cooldown_seconds || 60)
      setSuccess('New code sent!')
      setTimeout(() => setSuccess(''), 3000)
    } catch (err) {
      setError(err.message)
      if (err.status === 429) startCooldown(60)
    } finally {
      setResending(false)
    }
  }

  return (
    <div
      className="relative min-h-screen flex flex-col items-center justify-center px-6 transition-opacity duration-300"
      style={{ background: 'var(--t-bg)', opacity: mounted ? 1 : 0 }}
    >
      {/* Animated ambient background glow */}
      <div aria-hidden style={{
        position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none', zIndex: 0,
      }}>
        <div style={{
          position: 'absolute',
          top: '20%', left: '50%',
          width: 600, height: 600,
          marginLeft: -300, marginTop: -300,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(139,92,246,0.12) 0%, rgba(59,130,246,0.06) 50%, transparent 70%)',
          animation: 'glow-rotate 12s linear infinite',
          transformOrigin: 'center center',
        }} />
        <div style={{
          position: 'absolute',
          top: '55%', left: '30%',
          width: 400, height: 400,
          marginLeft: -200, marginTop: -200,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 65%)',
          animation: 'glow-rotate 18s linear infinite reverse',
          transformOrigin: 'center center',
        }} />
      </div>

      {/* Theme toggle — top right */}
      <button
        onClick={onToggleTheme}
        className="absolute top-5 right-5 z-20 w-9 h-9 flex items-center justify-center rounded-xl transition-colors"
        style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx4)' }}
        onMouseEnter={e => e.currentTarget.style.color = 'var(--t-accent)'}
        onMouseLeave={e => e.currentTarget.style.color = 'var(--t-tx4)'}
        title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {dark ? <Sun size={16} /> : <Moon size={16} />}
      </button>

      {/* Content — above the glow layer */}
      <div className="relative z-10 w-full flex flex-col items-center">
      {/* Brand */}
      <div className="flex flex-col items-center mb-6 sm:mb-10 gap-3 sm:gap-4">
        <img src="/logo.png" alt="MAGIK" className="w-14 h-14 sm:w-16 sm:h-16 rounded-2xl object-cover shadow-lg" />
        <div className="text-center">
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight" style={{
            background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}>MAGIK</h1>
          <p className="text-sm sm:text-base mt-2" style={{ color: 'var(--t-tx4)' }}>
            Multimodal · Agentic · RAG · Integrated · Knowledge
          </p>
        </div>
      </div>

      {/* Card */}
      <div className="w-full max-w-[440px] rounded-2xl p-6 sm:p-10"
        style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}>

        {otpStep ? (
          /* ── OTP Step ─────────────────────────────────── */
          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="text-center mb-2">
              <div className="w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-4"
                style={{ background: 'var(--t-accent-soft)', border: '1px solid var(--t-accent)' }}>
                <ShieldCheck size={26} style={{ color: 'var(--t-accent)' }} />
              </div>
              <h2 className="text-xl font-bold mb-1" style={{ color: 'var(--t-tx1)' }}>Check your email</h2>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--t-tx4)' }}>
                We sent a 6-digit code to<br/>
                <strong style={{ color: 'var(--t-tx2)' }}>{email}</strong>
              </p>
            </div>

            {/* 6-box OTP input */}
            <div className="flex gap-2 justify-center" onPaste={handleOtpPaste}>
              {otpCode.map((digit, idx) => (
                <input
                  key={idx}
                  ref={el => otpRefs.current[idx] = el}
                  type="text"
                  inputMode="numeric"
                  maxLength={1}
                  value={digit}
                  onChange={e => handleOtpChange(idx, e.target.value)}
                  onKeyDown={e => handleOtpKeyDown(idx, e)}
                  className="w-12 h-14 text-center text-xl font-bold rounded-xl outline-none transition-colors t-focus"
                  style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd4)', color: 'var(--t-tx1)' }}
                />
              ))}
            </div>

            {/* Spam folder hint */}
            <div className="flex items-start gap-2 rounded-xl px-3.5 py-2.5 text-xs leading-relaxed"
              style={{ background: 'rgba(234,179,8,0.07)', border: '1px solid rgba(234,179,8,0.18)', color: '#a38b2a' }}>
              <span style={{ fontSize: 14, lineHeight: 1.2, flexShrink: 0 }}>💡</span>
              <span style={{ color: 'var(--t-tx4)' }}>
                Didn't receive the code? Check your <strong style={{ color: 'var(--t-tx3)' }}>Spam</strong> or <strong style={{ color: 'var(--t-tx3)' }}>Junk</strong> folder — verification emails sometimes land there on first send.
              </span>
            </div>

            {error && <p className="text-sm text-center" style={{ color: 'var(--t-danger)' }}>{error}</p>}
            {success && <p className="text-sm text-center" style={{ color: 'var(--t-success)' }}>{success}</p>}

            <button
              type="submit"
              disabled={loading || otpCode.join('').length !== 6}
              className="w-full flex items-center justify-center gap-2 font-semibold rounded-xl py-3.5 text-base transition-colors disabled:opacity-50"
              style={{ background: 'var(--t-tx1)', color: 'var(--t-bg)' }}
              onMouseEnter={e => { if (!loading) e.currentTarget.style.background = 'var(--t-tx2)' }}
              onMouseLeave={e => e.currentTarget.style.background = 'var(--t-tx1)'}
            >
              {loading && <Loader2 size={16} className="animate-spin" />}
              Verify code
            </button>

            <div className="flex items-center justify-between text-sm" style={{ color: 'var(--t-tx5)' }}>
              <button type="button" onClick={() => { setOtpStep(false); setOtpCode(['','','','','','']); setError('') }}
                className="hover:opacity-80 transition-opacity">
                ← Back to sign in
              </button>
              <button
                type="button"
                onClick={handleResend}
                disabled={resendCooldown > 0 || resending}
                className="flex items-center gap-1 transition-opacity disabled:opacity-40 hover:opacity-70"
                style={{ color: 'var(--t-accent)' }}
              >
                <RotateCcw size={13} />
                {resendCooldown > 0 ? `Resend in ${resendCooldown}s` : 'Resend code'}
              </button>
            </div>
          </form>
        ) : (
          /* ── Login / Register Step ──────────────────────── */
          <>
            {/* Google */}
            <button
              type="button"
              onClick={() => { window.location.href = '/auth/google' }}
              className="w-full flex items-center justify-center gap-3 rounded-xl py-3.5 px-4 text-base font-medium transition-colors"
              style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd4)', color: 'var(--t-tx1)' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
              onMouseLeave={e => e.currentTarget.style.background = 'var(--t-inp)'}
            >
              <GoogleG />
              Continue with Google
            </button>

            {/* Divider */}
            <div className="flex items-center gap-3 my-7">
              <div className="flex-1 h-px" style={{ background: 'var(--t-bd2)' }} />
              <span className="text-sm font-medium" style={{ color: 'var(--t-ph)' }}>OR</span>
              <div className="flex-1 h-px" style={{ background: 'var(--t-bd2)' }} />
            </div>

            {/* Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              <input
                type="email"
                placeholder="Email address"
                value={email}
                onChange={e => setEmail(e.target.value)}
                required
                autoComplete="email"
                className="w-full rounded-xl px-5 py-3.5 text-base outline-none transition-colors t-focus"
                style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
              />

              <div className="relative">
                <input
                  type={showPass ? 'text' : 'password'}
                  placeholder="Password"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  required
                  autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                  className="w-full rounded-xl px-5 py-3.5 pr-12 text-base outline-none transition-colors t-focus"
                  style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
                />
                <button
                  type="button"
                  onClick={() => setShowPass(v => !v)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 transition-colors"
                  style={{ color: 'var(--t-ph)' }}
                >
                  {showPass ? <EyeOff size={18} /> : <Eye size={18} />}
                </button>
              </div>

              {mode === 'register' && (
                <div className="relative">
                  <input
                    type={showConfirm ? 'text' : 'password'}
                    placeholder="Confirm password"
                    value={confirm}
                    onChange={e => setConfirm(e.target.value)}
                    required
                    autoComplete="new-password"
                    className="w-full rounded-xl px-5 py-3.5 pr-12 text-base outline-none transition-colors t-focus"
                    style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirm(v => !v)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 transition-colors"
                    style={{ color: 'var(--t-ph)' }}
                  >
                    {showConfirm ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>
              )}

              {/* Forgot password link — only on login mode */}
              {mode === 'login' && (
                <div className="text-right">
                  <button
                    type="button"
                    onClick={onForgotPassword}
                    className="text-sm transition-opacity hover:opacity-70"
                    style={{ color: 'var(--t-tx5)' }}
                  >
                    Forgot password?
                  </button>
                </div>
              )}

              {error && (
                error.toLowerCase().includes('google sign-in') ? (
                  <div className="flex items-center gap-2.5 rounded-xl px-4 py-3 text-sm"
                    style={{ background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.25)', color: 'var(--t-tx3)' }}>
                    <GoogleG />
                    <span>{error}</span>
                  </div>
                ) : (
                  <p className="text-sm py-1" style={{ color: 'var(--t-danger)' }}>{error}</p>
                )
              )}
              {success && <p className="text-sm py-1" style={{ color: 'var(--t-success)' }}>{success}</p>}

              <button
                type="submit"
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 font-semibold rounded-xl py-3.5 text-base transition-colors disabled:opacity-60 mt-1"
                style={{ background: 'var(--t-tx1)', color: 'var(--t-bg)' }}
                onMouseEnter={e => { if (!loading) e.currentTarget.style.background = 'var(--t-tx2)' }}
                onMouseLeave={e => e.currentTarget.style.background = 'var(--t-tx1)'}
              >
                {loading && <Loader2 size={16} className="animate-spin" />}
                {mode === 'login' ? 'Sign in' : 'Create account'}
              </button>
            </form>

            {/* Mode toggle */}
            <p className="text-center text-base mt-6" style={{ color: 'var(--t-tx4)' }}>
              {mode === 'login' ? (
                <>New here?{' '}
                  <button onClick={() => switchMode('register')} className="transition-colors hover:opacity-80" style={{ color: 'var(--t-tx3)' }}>
                    Create an account →
                  </button>
                </>
              ) : (
                <>Already have an account?{' '}
                  <button onClick={() => switchMode('login')} className="transition-colors hover:opacity-80" style={{ color: 'var(--t-tx3)' }}>
                    Sign in →
                  </button>
                </>
              )}
            </p>
          </>
        )}
      </div>

      {/* Footer */}
      <p className="text-sm mt-7 text-center" style={{ color: 'var(--t-ph)' }}>
        By continuing you agree to our{' '}
        <span className="underline cursor-pointer hover:opacity-70">Terms of Service</span>
        {' '}and{' '}
        <span className="underline cursor-pointer hover:opacity-70">Privacy Policy</span>
      </p>
      </div> {/* end z-10 wrapper */}
    </div>
  )
}
