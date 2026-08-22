import { useState } from 'react'
import { Eye, EyeOff, Loader2, CheckCircle } from 'lucide-react'
import { resetPassword } from '../api/client'

export default function ResetPasswordPage({ token, onSuccess }) {
  // Starts visible — see LoginPage.jsx for the full Lighthouse CI NO_FCP
  // root-cause writeup on why a deferred `useState(false)` + rAF fade breaks
  // First Contentful Paint under headless/CDP-traced rendering.
  const mounted = true

  const [password, setPassword]   = useState('')
  const [confirm, setConfirm]     = useState('')
  const [showPass, setShowPass]       = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [loading, setLoading]     = useState(false)
  const [done, setDone]           = useState(false)
  const [error, setError]         = useState('')

  if (!token) {
    return (
      <div className="min-h-dvh-screen flex items-center justify-center px-4" style={{ background: 'var(--t-bg)' }}>
        <p style={{ color: 'var(--t-danger)' }}>Invalid or missing reset token.</p>
      </div>
    )
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (password !== confirm) { setError('Passwords do not match'); return }
    if (password.length < 8)  { setError('Password must be at least 8 characters'); return }
    setLoading(true)
    try {
      await resetPassword(token, password)
      setDone(true)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      className="relative h-dvh-screen overflow-y-auto flex flex-col items-center px-4 roomy:px-6 transition-opacity duration-300"
      style={{ background: 'var(--t-bg)', opacity: mounted ? 1 : 0 }}
    >
      {/* Ambient glow */}
      <div aria-hidden style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none', zIndex: 0 }}>
        <div style={{
          position: 'absolute', top: '20%', left: '50%',
          width: 600, height: 600, marginLeft: -300, marginTop: -300,
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(139,92,246,0.10) 0%, rgba(59,130,246,0.05) 50%, transparent 70%)',
          animation: 'glow-rotate 12s linear infinite',
        }} />
      </div>

      <div className="relative z-10 w-full flex flex-col items-center my-auto py-4">
        {/* Brand */}
        <div className="flex flex-col items-center mb-3 roomy:mb-8 gap-1.5 roomy:gap-3">
          <img src="/logo.png" alt="MAGIK" className="w-10 h-10 roomy:w-14 roomy:h-14 rounded-2xl object-cover shadow-lg" />
          <h1 className="text-xl roomy:text-3xl font-bold tracking-tight" style={{
            background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
          }}>MAGIK</h1>
        </div>

        <div className="w-full max-w-[440px] rounded-2xl p-4 roomy:p-10"
          style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}>

          {done ? (
            /* Success state */
            <div className="text-center space-y-3 roomy:space-y-4">
              <div className="w-12 h-12 roomy:w-16 roomy:h-16 rounded-2xl flex items-center justify-center mx-auto"
                style={{ background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)' }}>
                <CheckCircle size={24} style={{ color: '#22c55e' }} />
              </div>
              <h2 className="text-lg roomy:text-xl font-bold" style={{ color: 'var(--t-tx1)' }}>Password updated</h2>
              <p className="text-sm" style={{ color: 'var(--t-tx4)' }}>
                Your password has been changed. All previous sessions have been signed out.
              </p>
              <button
                onClick={onSuccess}
                className="w-full font-semibold rounded-xl py-2.5 roomy:py-3.5 text-sm roomy:text-base transition-colors mt-2"
                style={{ background: 'var(--t-tx1)', color: 'var(--t-bg)' }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--t-tx2)'}
                onMouseLeave={e => e.currentTarget.style.background = 'var(--t-tx1)'}
              >
                Sign in with new password
              </button>
            </div>
          ) : (
            /* Reset form */
            <>
              <h2 className="text-lg roomy:text-xl font-bold mb-1" style={{ color: 'var(--t-tx1)' }}>Set new password</h2>
              <p className="text-xs roomy:text-sm mb-4 roomy:mb-7" style={{ color: 'var(--t-tx4)' }}>
                Choose a strong password — at least 8 characters.
              </p>

              <form onSubmit={handleSubmit} className="space-y-2.5 roomy:space-y-4">
                <div className="relative">
                  <input
                    type={showPass ? 'text' : 'password'}
                    placeholder="New password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                    autoComplete="new-password"
                    className="w-full rounded-xl px-4 roomy:px-5 py-2.5 roomy:py-3.5 pr-11 roomy:pr-12 text-sm roomy:text-base outline-none transition-colors t-focus"
                    style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPass(v => !v)}
                    className="absolute right-3.5 roomy:right-4 top-1/2 -translate-y-1/2 transition-colors"
                    style={{ color: 'var(--t-ph)' }}
                    aria-label={showPass ? 'Hide password' : 'Show password'}
                  >
                    {showPass ? <EyeOff size={17} /> : <Eye size={17} />}
                  </button>
                </div>

                <div className="relative">
                  <input
                    type={showConfirm ? 'text' : 'password'}
                    placeholder="Confirm new password"
                    value={confirm}
                    onChange={e => setConfirm(e.target.value)}
                    required
                    autoComplete="new-password"
                    className="w-full rounded-xl px-4 roomy:px-5 py-2.5 roomy:py-3.5 pr-11 roomy:pr-12 text-sm roomy:text-base outline-none transition-colors t-focus"
                    style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
                  />
                  <button
                    type="button"
                    onClick={() => setShowConfirm(v => !v)}
                    className="absolute right-3.5 roomy:right-4 top-1/2 -translate-y-1/2 transition-colors"
                    style={{ color: 'var(--t-ph)' }}
                    aria-label={showConfirm ? 'Hide password' : 'Show password'}
                  >
                    {showConfirm ? <EyeOff size={17} /> : <Eye size={17} />}
                  </button>
                </div>

                {error && <p className="text-sm" style={{ color: 'var(--t-danger)' }}>{error}</p>}

                <button
                  type="submit"
                  disabled={loading || !password || !confirm}
                  className="w-full flex items-center justify-center gap-2 font-semibold rounded-xl py-2.5 roomy:py-3.5 text-sm roomy:text-base transition-colors disabled:opacity-60"
                  style={{ background: 'var(--t-tx1)', color: 'var(--t-bg)' }}
                  onMouseEnter={e => { if (!loading) e.currentTarget.style.background = 'var(--t-tx2)' }}
                  onMouseLeave={e => e.currentTarget.style.background = 'var(--t-tx1)'}
                >
                  {loading && <Loader2 size={16} className="animate-spin" />}
                  Update password
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
