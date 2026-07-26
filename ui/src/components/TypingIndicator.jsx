export default function TypingIndicator() {
  return (
    <div className="flex justify-start">
      <div
        className="rounded-2xl rounded-tl-sm px-4 py-3"
        style={{ background: 'var(--t-card)', border: '1px solid var(--t-hov2)' }}
      >
        <div className="flex gap-1.5 items-center h-4">
          {[0, 1, 2].map(i => (
            <span
              key={i}
              className="w-1.5 h-1.5 rounded-full bounce-dot"
              style={{ background: 'var(--t-tx4)', animationDelay: `${i * 0.18}s` }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
