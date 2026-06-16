import { useState, useEffect } from 'react'
import { Eye, EyeOff, Loader2, BrainCircuit, CheckCircle } from 'lucide-react'
import { resetPassword } from '../api/client'

export default function ResetPasswordPage({ token, onSuccess }) {
  const [mounted, setMounted] = useState(false)
  useEffect(() => { requestAnimationFrame(() => setMounted(true)) }, [])

  const [password, setPassword]   = useState('')
  const [confirm, setConfirm]     = useState('')
  const [showPass, setShowPass]       = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [loading, setLoading]     = useState(false)
  const [done, setDone]           = useState(false)
  const [error, setError]         = useState('')

  if (!token) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: 'var(--t-bg)' }}>
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
      className="relative min-h-screen flex flex-col items-center justify-center px-6 transition-opacity duration-300"
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

      <div className="relative z-10 w-full flex flex-col items-center">
        {/* Brand */}
        <div className="flex flex-col items-center mb-8 gap-3">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center shadow-lg">
            <BrainCircuit size={28} strokeWidth={1.7} className="text-white" />
          </div>
          <h1 className="text-3xl font-bold tracking-tight" style={{
            background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text',
          }}>MAGIK</h1>
        </div>

        <div className="w-full max-w-[440px] rounded-2xl p-6 sm:p-10"
          style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}>

          {done ? (
            /* Success state */
            <div className="text-center space-y-4">
              <div className="w-16 h-16 rounded-2xl flex items-center justify-center mx-auto"
                style={{ background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)' }}>
                <CheckCircle size={28} style={{ color: '#22c55e' }} />
              </div>
              <h2 className="text-xl font-bold" style={{ color: 'var(--t-tx1)' }}>Password updated</h2>
              <p className="text-sm" style={{ color: 'var(--t-tx4)' }}>
                Your password has been changed. All previous sessions have been signed out.
              </p>
              <button
                onClick={onSuccess}
                className="w-full font-semibold rounded-xl py-3.5 text-base transition-colors mt-2"
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
              <h2 className="text-xl font-bold mb-1" style={{ color: 'var(--t-tx1)' }}>Set new password</h2>
              <p className="text-sm mb-7" style={{ color: 'var(--t-tx4)' }}>
                Choose a strong password — at least 8 characters.
              </p>

              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="relative">
                  <input
                    type={showPass ? 'text' : 'password'}
                    placeholder="New password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                    autoComplete="new-password"
                    className="w-full rounded-xl px-5 py-3.5 pr-12 text-base outline-none transition-colors t-focus"
                    style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
                  />
                  <button type="button" onClick={() => setShowPass(v => !v)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 transition-colors"
                    style={{ color: 'var(--t-ph)' }}>
                    {showPass ? <EyeOff size={18} /> : <Eye size={18} />}
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
                    className="w-full rounded-xl px-5 py-3.5 pr-12 text-base outline-none transition-colors t-focus"
                    style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
                  />
                  <button type="button" onClick={() => setShowConfirm(v => !v)}
                    className="absolute right-4 top-1/2 -translate-y-1/2 transition-colors"
                    style={{ color: 'var(--t-ph)' }}>
                    {showConfirm ? <EyeOff size={18} /> : <Eye size={18} />}
                  </button>
                </div>

                {error && <p className="text-sm" style={{ color: 'var(--t-danger)' }}>{error}</p>}

                <button
                  type="submit"
                  disabled={loading || !password || !confirm}
                  className="w-full flex items-center justify-center gap-2 font-semibold rounded-xl py-3.5 text-base transition-colors disabled:opacity-60"
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
