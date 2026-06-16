import { useState, useEffect } from 'react'
import { ChevronDown, ChevronRight, Mic, MessageSquare } from 'lucide-react'
import { getTranscript } from '../api/client'
import MediaTimestampChip from './MediaTimestampChip'

/* ── Section labels and ordering ── */
const SECTION_ORDER = ['operator_intro', 'prepared_remarks', 'qa_session', 'closing_remarks']
const SECTION_LABELS = {
  operator_intro:     'Operator Intro',
  prepared_remarks:   'Prepared Remarks',
  qa_session:         'Q&A Session',
  closing_remarks:    'Closing Remarks',
}

/* ── Speaker role → display color ── */
const _ROLE_COLOR = { ceo: '#4ade80', cfo: '#60a5fa', analyst: '#fb923c', operator: '#a78bfa' }
const speakerColor = (role) => _ROLE_COLOR[(role || '').toLowerCase()] || '#22d3ee'

/* Group flat transcript chunks by call_section */
function groupBySections(chunks) {
  const map = {}
  for (const chunk of chunks) {
    const sec = chunk.call_section || 'prepared_remarks'
    if (!map[sec]) map[sec] = []
    map[sec].push(chunk)
  }
  const ordered = []
  for (const sec of SECTION_ORDER) {
    if (map[sec]) ordered.push({ section: sec, chunks: map[sec] })
  }
  for (const [sec, chunks] of Object.entries(map)) {
    if (!SECTION_ORDER.includes(sec)) ordered.push({ section: sec, chunks })
  }
  return ordered
}

/* ── Props:
     auth         – { token }
     fileHash     – string (audio/video KB file hash)
     filename     – string (display name)
     onSelectTurn – (query: string) => void — pre-fills chat input
     onClose      – () => void
*/
export default function EarningsCallBrowser({ auth, fileHash, filename, onSelectTurn, onClose }) {
  const [loading, setLoading]         = useState(true)
  const [error, setError]             = useState(null)
  const [sections, setSections]       = useState([])
  const [collapsed, setCollapsed]     = useState({})

  useEffect(() => {
    if (!fileHash) return
    setLoading(true); setError(null)
    getTranscript(auth.token, fileHash)
      .then(data => {
        const grouped = groupBySections(data.chunks || [])
        setSections(grouped)
        // Collapse all except prepared_remarks by default
        const init = {}
        grouped.forEach(g => { if (g.section !== 'prepared_remarks') init[g.section] = true })
        setCollapsed(init)
      })
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [fileHash])

  const toggle = (sec) => setCollapsed(prev => ({ ...prev, [sec]: !prev[sec] }))

  const handleTurnClick = (chunk) => {
    if (!onSelectTurn) return
    const who = chunk.speaker_name || chunk.speaker_role || 'Speaker'
    const ts  = chunk.start_timestamp != null ? ` at ${Math.round(chunk.start_timestamp)}s` : ''
    onSelectTurn(`What did ${who} say${ts}?`)
  }

  return (
    <div
      className="flex flex-col h-full overflow-hidden"
      style={{ background: 'var(--t-card)', borderLeft: '1px solid var(--t-bd2)' }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 flex-shrink-0"
        style={{ borderBottom: '1px solid var(--t-bd2)', background: 'var(--t-inp)' }}
      >
        <div>
          <div className="flex items-center gap-2">
            <Mic size={14} style={{ color: '#60a5fa' }} />
            <span className="text-sm font-semibold" style={{ color: 'var(--t-tx2)' }}>
              Earnings Call
            </span>
          </div>
          {filename && (
            <div className="text-[11px] mt-0.5 truncate max-w-[200px]" style={{ color: 'var(--t-tx5)' }}>
              {filename}
            </div>
          )}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-[11px] px-2 py-1 rounded-lg transition-colors"
            style={{ color: 'var(--t-tx5)' }}
            onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            ✕
          </button>
        )}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto px-2 py-2 space-y-1">
        {loading && (
          <div className="text-center py-8 text-[12px]" style={{ color: 'var(--t-tx5)' }}>
            Loading transcript…
          </div>
        )}
        {error && (
          <div className="text-center py-8 text-[12px]" style={{ color: 'var(--t-danger)' }}>
            {error}
          </div>
        )}

        {!loading && !error && sections.map(({ section, chunks }) => (
          <div key={section}>
            {/* Section header */}
            <button
              onClick={() => toggle(section)}
              className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-left transition-colors"
              style={{ color: 'var(--t-tx3)' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov1)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              {collapsed[section]
                ? <ChevronRight size={12} className="flex-shrink-0" />
                : <ChevronDown size={12} className="flex-shrink-0" />}
              <span className="text-[12px] font-semibold uppercase tracking-wider">
                {SECTION_LABELS[section] || section} ({chunks.length})
              </span>
            </button>

            {/* Speaker turns */}
            {!collapsed[section] && (
              <div className="ml-4 space-y-0.5 mt-0.5">
                {chunks.map((chunk, i) => {
                  const color = speakerColor(chunk.speaker_role)
                  const name  = chunk.speaker_name || chunk.speaker_role || 'Unknown'
                  const preview = (chunk.transcript || '').slice(0, 80).trim()
                  return (
                    <button
                      key={i}
                      onClick={() => handleTurnClick(chunk)}
                      className="w-full text-left px-2 py-1.5 rounded-lg transition-colors group"
                      style={{ cursor: onSelectTurn ? 'pointer' : 'default' }}
                      onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov1)'}
                      onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                    >
                      <div className="flex items-center gap-2 mb-0.5">
                        <span className="text-[11px] font-semibold" style={{ color }}>{name}</span>
                        <MediaTimestampChip
                          startSeconds={chunk.start_timestamp}
                          endSeconds={chunk.end_timestamp}
                          speakerRole={chunk.speaker_role}
                          modality="audio"
                          compact
                        />
                        {onSelectTurn && (
                          <MessageSquare
                            size={10}
                            className="ml-auto opacity-0 group-hover:opacity-60 flex-shrink-0 transition-opacity"
                            style={{ color: 'var(--t-tx5)' }}
                          />
                        )}
                      </div>
                      {preview && (
                        <div
                          className="text-[11px] leading-snug line-clamp-2"
                          style={{ color: 'var(--t-tx5)' }}
                        >
                          {preview}{chunk.transcript?.length > 80 ? '…' : ''}
                        </div>
                      )}
                    </button>
                  )
                })}
              </div>
            )}
          </div>
        ))}

        {!loading && !error && sections.length === 0 && (
          <div className="text-center py-8 text-[12px]" style={{ color: 'var(--t-tx5)' }}>
            No transcript available for this file.
          </div>
        )}
      </div>
    </div>
  )
}
