import { useState } from 'react'
import { Loader2, Mail, ArrowLeft } from 'lucide-react'
import { forgotPassword } from '../api/client'

export default function ForgotPasswordPage({ onBack }) {
  // Starts visible — see LoginPage.jsx for the full Lighthouse CI NO_FCP
  // root-cause writeup on why a deferred `useState(false)` + rAF fade breaks
  // First Contentful Paint under headless/CDP-traced rendering.
  const mounted = true

  const [email, setEmail]     = useState('')
  const [loading, setLoading] = useState(false)
  const [sent, setSent]       = useState(false)
  const [error, setError]     = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(''); setLoading(true)
    try {
      await forgotPassword(email)
      setSent(true)
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

          {sent ? (
            /* Success state */
            <div className="text-center space-y-3 roomy:space-y-4">
              <div className="w-12 h-12 roomy:w-16 roomy:h-16 rounded-2xl flex items-center justify-center mx-auto"
                style={{ background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)' }}>
                <Mail size={24} style={{ color: '#22c55e' }} />
              </div>
              <h2 className="text-lg roomy:text-xl font-bold" style={{ color: 'var(--t-tx1)' }}>Check your inbox</h2>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--t-tx4)' }}>
                If <strong style={{ color: 'var(--t-tx2)' }}>{email}</strong> is registered,
                you'll receive a password reset link shortly. The link expires in 1 hour.
              </p>
              <p className="text-xs" style={{ color: 'var(--t-tx5)' }}>
                Didn't get it? Check your spam folder.
              </p>
              <button
                onClick={onBack}
                className="w-full flex items-center justify-center gap-2 font-semibold rounded-xl py-2.5 roomy:py-3.5 text-sm roomy:text-base transition-colors mt-2"
                style={{ background: 'var(--t-inp)', color: 'var(--t-tx1)', border: '1px solid var(--t-bd3)' }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
                onMouseLeave={e => e.currentTarget.style.background = 'var(--t-inp)'}
              >
                <ArrowLeft size={16} /> Back to sign in
              </button>
            </div>
          ) : (
            /* Email form */
            <>
              <button
                onClick={onBack}
                className="flex items-center gap-1.5 text-sm mb-3 roomy:mb-6 transition-opacity hover:opacity-70"
                style={{ color: 'var(--t-tx5)' }}
              >
                <ArrowLeft size={15} /> Back to sign in
              </button>

              <h2 className="text-lg roomy:text-xl font-bold mb-1" style={{ color: 'var(--t-tx1)' }}>Forgot your password?</h2>
              <p className="text-xs roomy:text-sm mb-4 roomy:mb-7 leading-relaxed" style={{ color: 'var(--t-tx4)' }}>
                Enter your email and we'll send you a link to reset your password.
              </p>

              <form onSubmit={handleSubmit} className="space-y-2.5 roomy:space-y-4">
                <input
                  type="email"
                  placeholder="Email address"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  required
                  autoComplete="email"
                  className="w-full rounded-xl px-4 roomy:px-5 py-2.5 roomy:py-3.5 text-sm roomy:text-base outline-none transition-colors t-focus"
                  style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }}
                />

                {error && <p className="text-sm" style={{ color: 'var(--t-danger)' }}>{error}</p>}

                <button
                  type="submit"
                  disabled={loading || !email}
                  className="w-full flex items-center justify-center gap-2 font-semibold rounded-xl py-2.5 roomy:py-3.5 text-sm roomy:text-base transition-colors disabled:opacity-60"
                  style={{ background: 'var(--t-tx1)', color: 'var(--t-bg)' }}
                  onMouseEnter={e => { if (!loading) e.currentTarget.style.background = 'var(--t-tx2)' }}
                  onMouseLeave={e => e.currentTarget.style.background = 'var(--t-tx1)'}
                >
                  {loading && <Loader2 size={16} className="animate-spin" />}
                  Send reset link
                </button>
              </form>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
