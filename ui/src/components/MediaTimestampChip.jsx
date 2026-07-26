/* Clickable audio/video timestamp chip — renders as [🎙 CFO 28:30] style.
   Props:
     startSeconds  – number, required
     endSeconds    – number, optional
     speakerName   – string, optional
     speakerRole   – string, optional (CEO / CFO / Analyst)
     modality      – "audio" | "video" (determines emoji)
     onClick       – (seconds: number) => void, optional
     compact       – boolean (omit speaker name label)
*/

function fmtSec(sec) {
  const t = parseFloat(sec)
  if (isNaN(t)) return '0:00'
  const h = Math.floor(t / 3600)
  const m = Math.floor((t % 3600) / 60)
  const s = Math.floor(t % 60).toString().padStart(2, '0')
  return h > 0 ? `${h}:${m.toString().padStart(2, '0')}:${s}` : `${m}:${s}`
}

const _ROLE_COLORS = {
  ceo:      '#4ade80',
  cfo:      '#60a5fa',
  analyst:  '#fb923c',
  operator: '#a78bfa',
}
function roleColor(role) {
  if (!role) return '#22d3ee'
  return _ROLE_COLORS[role.toLowerCase()] || '#22d3ee'
}

export default function MediaTimestampChip({
  startSeconds, endSeconds, speakerName, speakerRole,
  modality = 'audio', onClick, compact = false,
}) {
  const icon  = modality === 'video' ? '🎬' : '🎙'
  const ts    = fmtSec(startSeconds)
  const color = roleColor(speakerRole)
  const label = compact ? null : (speakerName || speakerRole || null)

  const handleClick = () => {
    if (onClick) onClick(parseFloat(startSeconds) || 0)
  }

  return (
    <button
      onClick={onClick ? handleClick : undefined}
      title={`${speakerRole || 'Speaker'} at ${ts}${endSeconds != null ? ' – ' + fmtSec(endSeconds) : ''}`}
      className="inline-flex items-center gap-1 text-[11px] leading-4 rounded-2xl px-2.5 py-1 transition-colors select-none"
      style={{
        background: 'var(--t-chp)',
        border: `1px solid ${color}55`,
        color,
        cursor: onClick ? 'pointer' : 'default',
        fontFamily: 'ui-monospace, Consolas, monospace',
      }}
      onMouseEnter={e => { if (onClick) e.currentTarget.style.borderColor = color }}
      onMouseLeave={e => e.currentTarget.style.borderColor = `${color}55`}
    >
      <span>{icon}</span>
      {label && <span style={{ color: 'var(--t-tx3)', fontFamily: 'inherit' }}>{label}</span>}
      <span style={{ fontWeight: 600 }}>{ts}</span>
    </button>
  )
}
