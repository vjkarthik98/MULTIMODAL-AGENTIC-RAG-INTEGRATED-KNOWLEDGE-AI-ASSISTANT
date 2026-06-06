import { useState, useEffect } from 'react'
import { Eye, EyeOff, Loader2, Sun, Moon, BrainCircuit } from 'lucide-react'
import { login, register } from '../api/client'

const GoogleG = () => (
  <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
    <path d="M16.51 8H8.98v3h4.3c-.18 1-.74 1.48-1.6 2.04v2.01h2.6a7.8 7.8 0 0 0 2.38-5.88c0-.57-.05-.66-.15-1.18" fill="#4285F4"/>
    <path d="M8.98 17c2.16 0 3.97-.72 5.3-1.94l-2.6-2a4.8 4.8 0 0 1-7.18-2.54H1.83v2.07A8 8 0 0 0 8.98 17" fill="#34A853"/>
    <path d="M4.5 10.52a4.8 4.8 0 0 1 0-3.04V5.41H1.83a8 8 0 0 0 0 7.18z" fill="#FBBC05"/>
    <path d="M8.98 4.18c1.17 0 2.23.4 3.06 1.2l2.3-2.3A8 8 0 0 0 1.83 5.4L4.5 7.49a4.77 4.77 0 0 1 4.48-3.3" fill="#EA4335"/>
  </svg>
)

export default function LoginPage({ onLogin, dark, onToggleTheme }) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => { requestAnimationFrame(() => setMounted(true)) }, [])

  const [mode, setMode]         = useState('login')
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm]   = useState('')
  const [showPass, setShowPass] = useState(false)
  const [error, setError]       = useState('')
  const [success, setSuccess]   = useState('')
  const [loading, setLoading]   = useState(false)

  const switchMode = (m) => { setMode(m); setError(''); setSuccess('') }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(''); setSuccess('')
    if (mode === 'register' && password !== confirm) {
      setError('Passwords do not match')
      return
    }
    setLoading(true)
    try {
      if (mode === 'register') {
        await register(email, password)
        setSuccess('Account created! Signing you in…')
        const data = await login(email, password)
        onLogin({ token: data.access_token, refreshToken: data.refresh_token, email })
      } else {
        const data = await login(email, password)
        onLogin({ token: data.access_token, refreshToken: data.refresh_token, email })
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
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
      <div className="flex flex-col items-center mb-10 gap-4">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center shadow-lg">
          <BrainCircuit size={32} strokeWidth={1.7} className="text-white" />
        </div>
        <div className="text-center">
          <h1 className="text-4xl font-bold tracking-tight" style={{
            background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}>MAGIK</h1>
          <p className="text-base mt-2" style={{ color: 'var(--t-tx4)' }}>
            Multimodal · Agentic · RAG · Integrated · Knowledge
          </p>
        </div>
      </div>

      {/* Card */}
      <div className="w-full max-w-[440px] rounded-2xl p-10"
        style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}>

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
            <input
              type="password"
              placeholder="Confirm password"
              value={confirm}
              onChange={e => setConfirm(e.target.value)}
              required
              autoComplete="new-password"
              className="w-full rounded-xl px-5 py-3.5 text-base outline-none transition-colors t-focus"
              style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
            />
          )}

          {error   && <p className="text-sm py-1" style={{ color: 'var(--t-danger)' }}>{error}</p>}
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
