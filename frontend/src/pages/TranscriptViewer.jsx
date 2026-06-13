import { useState, useEffect, useRef, useMemo } from 'react'
import { Search, X, Mic, ChevronLeft } from 'lucide-react'
import { getTranscript } from '../api/client'
import MediaTimestampChip from '../components/MediaTimestampChip'

/* ── Speaker role → display color ── */
const _ROLE_COLORS = {
  ceo:      { bg: '#14532d', border: '#4ade80', text: '#4ade80' },
  cfo:      { bg: '#1e3a5f', border: '#60a5fa', text: '#60a5fa' },
  analyst:  { bg: '#431407', border: '#fb923c', text: '#fb923c' },
  operator: { bg: '#2e1065', border: '#a78bfa', text: '#a78bfa' },
}
const _DEFAULT_COLOR = { bg: '#134e4a', border: '#22d3ee', text: '#22d3ee' }

function speakerStyle(role) {
  return _ROLE_COLORS[(role || '').toLowerCase()] || _DEFAULT_COLOR
}

/* ── Section labels ── */
const SECTION_LABELS = {
  operator_intro:   'Operator Introduction',
  prepared_remarks: 'Prepared Remarks',
  qa_session:       'Q&A Session',
  closing_remarks:  'Closing Remarks',
}

/* ── Highlight a search term in text ── */
function Highlighted({ text, term }) {
  if (!term) return <>{text}</>
  const re  = new RegExp(`(${term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi')
  const parts = text.split(re)
  return (
    <>
      {parts.map((part, i) =>
        re.test(part)
          ? <mark key={i} style={{ background: '#854d0e', color: '#fde68a', borderRadius: 2 }}>{part}</mark>
          : part
      )}
    </>
  )
}

/* ── Props:
     auth      – { token }
     fileHash  – string
     filename  – string (display)
     onBack    – () => void, optional
*/
export default function TranscriptViewer({ auth, fileHash, filename, onBack }) {
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [chunks, setChunks]     = useState([])
  const [search, setSearch]     = useState('')
  const [activeSec, setActiveSec] = useState(null)
  const searchRef               = useRef(null)

  useEffect(() => {
    if (!fileHash) return
    setLoading(true); setError(null)
    getTranscript(auth.token, fileHash)
      .then(data => setChunks(data.chunks || []))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [fileHash])

  /* ── Derive section boundaries ── */
  const sections = useMemo(() => {
    const map = {}
    for (const chunk of chunks) {
      const sec = chunk.call_section || 'prepared_remarks'
      if (!map[sec]) map[sec] = []
      map[sec].push(chunk)
    }
    return map
  }, [chunks])

  /* ── Filter chunks by search term ── */
  const termLower = search.trim().toLowerCase()
  const filtered = useMemo(() => {
    if (!termLower) return chunks
    return chunks.filter(c => (c.transcript || '').toLowerCase().includes(termLower))
  }, [chunks, termLower])

  const matchCount = termLower ? filtered.length : null

  return (
    <div
      className="flex flex-col h-full overflow-hidden"
      style={{ background: 'var(--t-bg)', color: 'var(--t-tx2)' }}
    >
      {/* ── Top bar ── */}
      <div
        className="flex items-center gap-3 px-4 py-3 flex-shrink-0"
        style={{ borderBottom: '1px solid var(--t-bd2)', background: 'var(--t-card)' }}
      >
        {onBack && (
          <button
            onClick={onBack}
            className="p-1 rounded-lg transition-colors"
            style={{ color: 'var(--t-tx5)' }}
            onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <ChevronLeft size={18} />
          </button>
        )}
        <Mic size={16} style={{ color: '#60a5fa', flexShrink: 0 }} />
        <div className="flex-1 min-w-0">
          <div className="text-[13px] font-semibold truncate" style={{ color: 'var(--t-tx2)' }}>
            {filename || 'Transcript'}
          </div>
          <div className="text-[11px]" style={{ color: 'var(--t-tx5)' }}>
            {chunks.length} segments
            {Object.keys(sections).map(sec => (
              <span key={sec} className="ml-2">· {SECTION_LABELS[sec] || sec}</span>
            ))}
          </div>
        </div>

        {/* Speaker legend */}
        <div className="hidden sm:flex items-center gap-2">
          {Object.entries(_ROLE_COLORS).map(([role, c]) => (
            <span key={role} className="flex items-center gap-1 text-[10px] capitalize" style={{ color: c.text }}>
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: c.border }} />
              {role}
            </span>
          ))}
        </div>
      </div>

      {/* ── Search bar ── */}
      <div className="px-4 py-2 flex-shrink-0" style={{ borderBottom: '1px solid var(--t-bd1)' }}>
        <div
          className="flex items-center gap-2 rounded-lg px-3 py-2"
          style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)' }}
        >
          <Search size={13} style={{ color: 'var(--t-tx5)', flexShrink: 0 }} />
          <input
            ref={searchRef}
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search within transcript…"
            className="flex-1 bg-transparent outline-none text-[13px]"
            style={{ color: 'var(--t-tx2)' }}
          />
          {search && (
            <>
              {matchCount != null && (
                <span className="text-[11px]" style={{ color: 'var(--t-tx5)' }}>
                  {matchCount} match{matchCount !== 1 ? 'es' : ''}
                </span>
              )}
              <button onClick={() => setSearch('')} style={{ color: 'var(--t-tx5)' }}>
                <X size={13} />
              </button>
            </>
          )}
        </div>
      </div>

      {/* ── Body ── */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-6">
        {loading && (
          <div className="text-center py-12 text-[13px]" style={{ color: 'var(--t-tx5)' }}>
            Loading transcript…
          </div>
        )}
        {error && (
          <div className="text-center py-12 text-[13px]" style={{ color: 'var(--t-danger)' }}>
            {error}
          </div>
        )}

        {!loading && !error && (() => {
          if (termLower) {
            // Flat list of matching segments
            return filtered.length === 0
              ? (
                <div className="text-center py-12 text-[13px]" style={{ color: 'var(--t-tx5)' }}>
                  No matches for "{search}"
                </div>
              )
              : filtered.map((chunk, i) => (
                  <TranscriptSegment key={i} chunk={chunk} term={search} />
                ))
          }

          // Group by section
          const rendered = []
          let prevSection = null
          for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i]
            const sec = chunk.call_section || 'prepared_remarks'
            if (sec !== prevSection) {
              rendered.push(
                <SectionHeader key={`sec-${sec}`} section={sec} />
              )
              prevSection = sec
            }
            rendered.push(<TranscriptSegment key={i} chunk={chunk} term="" />)
          }
          return rendered
        })()}
      </div>
    </div>
  )
}

function SectionHeader({ section }) {
  return (
    <div
      className="flex items-center gap-3 py-2"
      style={{ color: 'var(--t-tx5)' }}
    >
      <div className="flex-1 h-px" style={{ background: 'var(--t-bd2)' }} />
      <span className="text-[11px] font-semibold uppercase tracking-widest whitespace-nowrap">
        {SECTION_LABELS[section] || section}
      </span>
      <div className="flex-1 h-px" style={{ background: 'var(--t-bd2)' }} />
    </div>
  )
}

function TranscriptSegment({ chunk, term }) {
  const [expanded, setExpanded] = useState(false)
  const colors = speakerStyle(chunk.speaker_role)
  const name   = chunk.speaker_name || chunk.speaker_role || 'Unknown'
  const text   = chunk.transcript || ''
  const PREVIEW = 200
  const isLong = text.length > PREVIEW

  return (
    <div
      className="rounded-xl px-4 py-3"
      style={{
        background: colors.bg + '33',
        border: `1px solid ${colors.border}44`,
      }}
    >
      {/* Speaker header */}
      <div className="flex items-center gap-2 mb-2">
        <span className="text-[12px] font-bold" style={{ color: colors.text }}>
          {name}
        </span>
        {chunk.speaker_role && chunk.speaker_role !== chunk.speaker_name && (
          <span className="text-[11px] capitalize" style={{ color: colors.text, opacity: 0.7 }}>
            {chunk.speaker_role}
          </span>
        )}
        {chunk.start_timestamp != null && (
          <MediaTimestampChip
            startSeconds={chunk.start_timestamp}
            endSeconds={chunk.end_timestamp}
            speakerRole={chunk.speaker_role}
            modality="audio"
            compact
          />
        )}
        {chunk.call_section && (
          <span
            className="ml-auto text-[10px] px-2 py-0.5 rounded-full capitalize"
            style={{ background: 'var(--t-chp)', color: 'var(--t-tx5)', border: '1px solid var(--t-chpb)' }}
          >
            {SECTION_LABELS[chunk.call_section] || chunk.call_section}
          </span>
        )}
      </div>

      {/* Transcript text */}
      <div className="text-[13px] leading-relaxed" style={{ color: 'var(--t-tx2)' }}>
        <Highlighted
          text={isLong && !expanded ? text.slice(0, PREVIEW) + '…' : text}
          term={term}
        />
        {isLong && (
          <button
            onClick={() => setExpanded(e => !e)}
            className="ml-1 text-[11px] underline"
            style={{ color: 'var(--t-tx5)' }}
          >
            {expanded ? 'show less' : 'show more'}
          </button>
        )}
      </div>

      {/* Finance entities */}
      {chunk.finance_entities && Object.keys(chunk.finance_entities).length > 0 && (
        <div className="flex flex-wrap gap-1 mt-2">
          {Object.entries(chunk.finance_entities).flatMap(([type, values]) =>
            (Array.isArray(values) ? values : [values]).map((v, i) => (
              <span
                key={`${type}-${i}`}
                className="text-[10px] rounded-lg px-1.5 py-0.5"
                style={{ background: 'var(--t-chp)', color: colors.text, border: `1px solid ${colors.border}44` }}
              >
                {v}
              </span>
            ))
          )}
        </div>
      )}
    </div>
  )
}
