import { GUEST_QUERY_LIMIT } from '../utils/guestConstants'

export default function GuestBanner({ queriesLeft, uploadsLeft, onSignUp }) {
  const queriesUsed = Math.max(0, GUEST_QUERY_LIMIT - queriesLeft)
  const pct = Math.round((queriesUsed / GUEST_QUERY_LIMIT) * 100)

  return (
    <div
      className="flex items-center justify-between px-4 py-2 text-sm flex-shrink-0"
      style={{
        background: 'color-mix(in srgb, var(--t-accent) 12%, var(--t-bg))',
        borderBottom: '1px solid color-mix(in srgb, var(--t-accent) 30%, transparent)',
      }}
    >
      <div className="flex items-center gap-3 min-w-0">
        {/* Zap icon */}
        <svg
          width="14" height="14" viewBox="0 0 24 24" fill="none"
          stroke="var(--t-accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
          className="flex-shrink-0"
        >
          <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
        </svg>

        <span style={{ color: 'var(--t-tx2)' }} className="whitespace-nowrap">
          <strong style={{ color: 'var(--t-tx1)' }}>{queriesLeft}</strong>
          {' '}free {queriesLeft === 1 ? 'query' : 'queries'} left
        </span>

        {/* Progress bar */}
        <div
          className="w-16 h-1 rounded-full flex-shrink-0 hidden sm:block"
          style={{ background: 'var(--t-bd2)' }}
        >
          <div
            className="h-full rounded-full transition-all duration-500"
            style={{ width: `${pct}%`, background: 'var(--t-accent)' }}
          />
        </div>

        {uploadsLeft <= 0 && (
          <span
            className="text-xs px-2 py-0.5 rounded-full flex-shrink-0 hidden sm:inline-flex"
            style={{
              background: 'color-mix(in srgb, var(--t-accent) 15%, transparent)',
              color: 'var(--t-accent)',
            }}
          >
            uploads full
          </span>
        )}
      </div>

      <button
        onClick={onSignUp}
        className="ml-3 text-xs font-semibold px-3 py-1.5 rounded-lg transition-colors flex-shrink-0"
        style={{ background: 'var(--t-accent)', color: 'white' }}
        onMouseEnter={e => e.currentTarget.style.opacity = '0.88'}
        onMouseLeave={e => e.currentTarget.style.opacity = '1'}
      >
        Sign up free →
      </button>
    </div>
  )
}
