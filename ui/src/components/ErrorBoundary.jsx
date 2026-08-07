import { Component } from 'react'
import { RefreshCw } from 'lucide-react'

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }

  componentDidCatch(error, info) {
    // Log only the message + component stack (component names, not props/state
    // or full error objects) — never risk printing something that embeds a
    // URL or response body a caught error happened to carry.
    console.error('[MAGIK] Unhandled render error:', error?.message || String(error), info.componentStack)
  }

  render() {
    if (!this.state.hasError) return this.props.children

    return (
      <div
        className="min-h-screen flex flex-col items-center justify-center px-6 text-center"
        style={{ background: 'var(--t-bg, #0a0a0a)' }}
      >
        <img src="/logo.png" alt="MAGIK" className="w-16 h-16 rounded-2xl object-cover shadow-lg mb-6" />

        <h1
          className="text-2xl font-bold mb-2"
          style={{
            background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}
        >
          Something went wrong
        </h1>

        <p className="text-sm mb-8 max-w-xs leading-relaxed" style={{ color: '#6b7280' }}>
          An unexpected error occurred. Your data is safe — refreshing the page will restore the app.
        </p>

        <button
          onClick={() => window.location.reload()}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-white transition-opacity hover:opacity-85"
          style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)' }}
        >
          <RefreshCw size={15} />
          Reload page
        </button>

        {process.env.NODE_ENV !== 'production' && this.state.error && (
          <pre className="mt-8 text-left text-xs p-4 rounded-xl max-w-lg overflow-auto"
            style={{ background: '#1a1a1a', color: '#f87171', border: '1px solid #2a2a2a' }}>
            {this.state.error.toString()}
          </pre>
        )}
      </div>
    )
  }
}
