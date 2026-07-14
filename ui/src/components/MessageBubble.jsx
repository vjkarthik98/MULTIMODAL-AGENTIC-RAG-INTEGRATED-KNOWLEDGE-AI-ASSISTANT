import { useState, Children } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { FileText, Globe, Copy, ThumbsUp, ThumbsDown, Check, RotateCcw, Pencil,
         Image, Sheet, FileVideo, FileAudio, FileType, LetterText, File } from 'lucide-react'
import { useToast } from '../context/ToastContext'
import useIsMobile from '../hooks/useIsMobile'
import { submitFeedback } from '../api/client'
import FinanceTable from './FinanceTable'

// Extract inline citations from LLM output. Handles both the bare form
// [filename.pdf] and the page-tagged forms the CoT path emits:
//   [filename.pdf p.6]  [report.pdf, p.12]  [data.csv page 3]
// The page number is captured so the rendered chip can show "· p.6".
const CITATION_RE = /\[\s*([^\]\n]{1,120}?\.(?:pdf|txt|docx|xlsx|csv|png|jpg|jpeg|mp4|mp3|wav|pptx|md))(?:[\s,]+(?:pp?\.?|pg\.?|page)\s*(\d+))?\s*\]/gi
// Numeric citations like [1], [2,3], [1, 2, 4] — carry no filename, so they
// are simply removed from the text (the source chips come from message.sources
// or the filename citations above). Without this, reopened chats show stray
// "[1]" markers inline.
const NUMERIC_CITATION_RE = /\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]/g
// Mangled cite tags where the PII/URL scrubber ate part of the filename,
// e.g. [aapl_def14a_<URL>cx] (from aapl_def14a_2023.docx). The real source
// still comes through cleanly in message.sources, so just remove the tag.
const MANGLED_CITATION_RE = /\s*\[[^\]\n]*<[A-Z_]{2,20}>[^\]\n]*\]/g
// Bare PII placeholder tags that leaked into the prose (e.g. a webcast "<URL>").
const PII_TAG_RE = /<(?:PERSON|LOCATION|ORG|NRP|GPE|DATE_TIME|AGE|ID|URL|IP_ADDRESS|US_SSN|CREDIT_CARD|PHONE_NUMBER|EMAIL_ADDRESS)>/gi

// DOCX section citations (rag_pipeline._attach_section_citations): the backend
// inserts inline "[1.2]" markers after cited sentences PLUS a trailing
// "Sources: [1.2 Investment Thesis], [5.1.1 China Revenue...]" footer with the
// full heading names. Per product decision these no longer render inline in
// the prose (kept minimal like PDF's "[p.N]") — pull the full heading names
// out of the footer (falling back to the bare inline number if no footer is
// present) and render them once at the end of the answer, no "Sources:"
// label (DocxCitations component, same treatment as XLSX's end-of-answer
// citation).
const DOCX_FOOTER_RE = /\n*Sources?:\s*((?:\[\d+(?:\.\d+)*[^\]\n]*\](?:,\s*)?)+)\s*$/i
const DOCX_HEADING_RE = /\[(\d+(?:\.\d+)*[^\]\n]*)\]/g
const DOCX_INLINE_SECTION_RE = /\s*\[(\d+(?:\.\d+)+)[^\]\n]*\]/g

function parseInlineCitations(content) {
  if (!content) return { cleanContent: content, inlineSources: [], docxSections: [] }
  const found = []
  let clean = content.replace(CITATION_RE, (_, name, page) => {
    const src = { source: name }
    if (page) src.page_number = parseInt(page, 10)
    found.push(src)
    return ''
  })

  // Pull DOCX section citations out of the text entirely (footer first, since
  // it carries the full heading name; then any remaining bare inline
  // markers as a fallback), deduped, in first-seen order.
  const docxSections = []
  const seenSections = new Set()
  const footerMatch = clean.match(DOCX_FOOTER_RE)
  if (footerMatch) {
    let hm
    DOCX_HEADING_RE.lastIndex = 0
    while ((hm = DOCX_HEADING_RE.exec(footerMatch[1])) !== null) {
      const h = hm[1].trim()
      if (!seenSections.has(h)) { seenSections.add(h); docxSections.push(h) }
    }
    clean = clean.slice(0, footerMatch.index)
  }
  clean = clean.replace(DOCX_INLINE_SECTION_RE, (_m, sid) => {
    if (docxSections.length === 0 && !seenSections.has(sid)) {
      seenSections.add(sid)
      docxSections.push(sid)
    }
    return ''
  })

  // Strip mangled cite tags, bare numeric citations, and leftover PII tags so
  // no citation/placeholder marker remains in the response text — all sources
  // live only in the chips below.
  clean = clean
    .replace(MANGLED_CITATION_RE, '')
    .replace(NUMERIC_CITATION_RE, '')
    .replace(PII_TAG_RE, '')
  // Safety net: the "§" section symbol must NEVER reach the screen (older
  // cached messages may still contain "[§4.1]").
  clean = clean.replace(/§\s*/g, '')
  // Tidy whitespace left where markers were removed: " ." → "." and
  // collapse any double spaces, then trim the trailing edge.
  clean = clean.replace(/\s+([.,;!?])/g, '$1').replace(/[ \t]{2,}/g, ' ').trimEnd()
  return { cleanContent: clean, inlineSources: found, docxSections }
}

// PDF page citation, e.g. "[p.38]" (rag_pipeline._attach_page_citations). This
// is the only inline citation kept mid-prose — DOCX's "[4.1]" markers are
// extracted out entirely by parseInlineCitations and rendered once at the end
// of the answer instead (DocxCitations), so CITE_TOKEN_RE only ever needs to
// match the page-number shape here.
const CITE_TOKEN_RE = /\[p\.(\d+)\]/g

function CitePill({ page }) {
  // Rendered inline at the SAME font size as the answer prose (no superscript,
  // no pill chip) — just an accent-coloured "[p.26]" reference so it reads as a
  // natural part of the sentence.
  return (
    <span
      title={`Source: page ${page}`}
      style={{ color: 'var(--t-accent)', fontWeight: 500, whiteSpace: 'nowrap' }}
    >
      {' '}[p.{page}]
    </span>
  )
}

function SectionCitePill({ label }) {
  // Accent-coloured DOCX section reference, e.g. "[1.2 Investment Thesis]" —
  // shown once at the end of the answer (DocxCitations), never inline, and
  // never prefixed with a "Sources:" label.
  return (
    <span
      title={`Source: ${label}`}
      style={{ color: 'var(--t-accent)', fontWeight: 500 }}
    >
      {' '}[{label}]
    </span>
  )
}

function DocxCitations({ sections }) {
  if (!sections || sections.length === 0) return null
  return (
    <div className="mt-1 text-[15px] leading-relaxed">
      {sections.map((label, i) => <SectionCitePill key={i} label={label} />)}
    </div>
  )
}

// Walk a ReactMarkdown node's children and replace [p.N] substrings inside
// text nodes with coloured PDF page-citation elements; non-strings pass
// through. DOCX section citations never reach here — parseInlineCitations
// strips them from cleanContent before it hits ReactMarkdown.
function injectPageCites(children) {
  return Children.map(children, (child) => {
    if (typeof child !== 'string') return child
    const parts = []
    let last = 0
    let m
    CITE_TOKEN_RE.lastIndex = 0
    while ((m = CITE_TOKEN_RE.exec(child)) !== null) {
      if (m.index > last) parts.push(child.slice(last, m.index))
      parts.push(<CitePill key={`${m.index}-p${m[1]}`} page={m[1]} />)
      last = m.index + m[0].length
    }
    if (last < child.length) parts.push(child.slice(last))
    return parts.length ? parts : child
  })
}

// Deduplicate sources: web sources by hostname, file sources by basename.
// When a key is seen again, keep whichever entry has more locator info
// (section_title / page_number / start_time) so the richer chip wins.
function sourceRichness(src) {
  if (typeof src !== 'object') return 0
  return (src.section_title ? 2 : 0) + (src.page_number != null ? 2 : 0) + (src.start_time != null ? 1 : 0)
}
function deduplicateSources(messageSources, inlineSources) {
  const seen = new Map()   // key → index in result
  const result = []
  const add = (src) => {
    const raw = typeof src === 'string' ? src : (src.source || src.filename || src.file || '')
    const isWeb = typeof src === 'object' && src.modality === 'web'
    let key
    if (isWeb) {
      try { key = new URL(raw).hostname.toLowerCase().replace(/^www\./, '') } catch { key = raw.toLowerCase() }
    } else {
      key = raw.toLowerCase().replace(/^.*[/\\]/, '')  // basename
    }
    if (!key) return
    if (seen.has(key)) {
      // Replace if newcomer is richer
      const idx = seen.get(key)
      if (sourceRichness(src) > sourceRichness(result[idx])) result[idx] = src
    } else {
      seen.set(key, result.length)
      result.push(src)
    }
  }
  ;(messageSources || []).forEach(add)
  inlineSources.forEach(add)
  return result
}

/* ── Detect "no info" responses — sources should never appear with these ── */
const NO_INFO_PATTERNS = [
  'no relevant information was found',
  'no relevant documents found',
  'could not find any relevant documents',
  'i could not find any relevant',
  'no information found in your knowledge base',
  'please ingest documents first',
  'no documents found',
  'nothing relevant was found',
  'no relevant information',
]
function isNoInfoResponse(content) {
  if (!content) return false
  const lc = content.toLowerCase()
  return NO_INFO_PATTERNS.some(p => lc.includes(p))
}

/* ── Modality colour + icon — unified teal for all source types ── */
function getModalityColor(_source, _isWeb) {
  return '#22d3ee'
}

function SourceIcon({ source, isWeb }) {
  const cls   = "flex-shrink-0 mt-[3px]"
  const color = getModalityColor(source, isWeb)
  const s     = { color }
  if (isWeb) return <Globe size={10} className={cls} style={s} />

  const modality = typeof source === 'object' ? (source.modality || '') : ''
  const raw = typeof source === 'string' ? source : (source.source || source.filename || source.file || '')
  const ext = (raw.split('.').pop() || '').toUpperCase()

  if (modality === 'audio' || ['MP3','WAV','M4A','OGG','FLAC','OPUS','AIFF','WMA'].includes(ext))
    return <FileAudio size={10} className={cls} style={s} />
  if (modality === 'video' || ['MP4','MOV','AVI','MKV','WEBM'].includes(ext))
    return <FileVideo size={10} className={cls} style={s} />
  if (modality === 'image' || ['PNG','JPG','JPEG','GIF','WEBP'].includes(ext))
    return <Image size={10} className={cls} style={s} />
  if (['table','excel'].includes(modality) || ['XLS','XLSX','CSV'].includes(ext))
    return <Sheet size={10} className={cls} style={s} />
  if (ext === 'PDF')
    return <FileText size={10} className={cls} style={s} />
  if (['DOC','DOCX'].includes(ext))
    return <LetterText size={10} className={cls} style={s} />
  if (ext === 'TXT')
    return <FileType size={10} className={cls} style={s} />
  return <File size={10} className={cls} style={s} />
}

/* ── Seconds → M:SS string ── */
function fmtTimestamp(sec) {
  const t = parseFloat(sec)
  if (isNaN(t)) return null
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

/* ── Source chip ──
   Full Phase 6.3 citation shape:
   { source/filename, modality, page, section_title, sheet_name, heading,
     timestamp_start, timestamp_end, speaker_role, speaker_name,
     call_section, row_range, chunk_type, image_title, slide_numbers, snippet }
*/
/* Shared label/suffix derivation used by both the web source card and the
   inline end-of-answer citation text — single source of truth so the two
   renderers never drift (e.g. one showing a page number the other omits). */
function getSourceLabelParts(source) {
  const isWeb  = typeof source === 'object' && source.modality === 'web'
  const raw    = typeof source === 'string' ? source : (source.source || source.filename || source.file || String(source))

  let label  = raw.replace(/^.*[/\\]/, '').replace(/<[A-Z_]{2,20}>/g, '')   // basename, strip PII tags
  let suffix = ''

  if (isWeb) {
    try { label = new URL(raw).hostname.replace(/^www\./, '') } catch { /* keep raw */ }
  } else if (typeof source === 'object') {
    const mod    = source.modality || ''
    const isTxt  = mod === 'text' || mod === 'txt'
    const isAudio = ['audio', 'mp3'].includes(mod)
    const isVideo = ['video', 'mp4'].includes(mod)
    // DOCX sections and XLSX sheet+row are cited via a colored inline pill at
    // the end of the answer (XlsxCitePill below, same pattern as CitePill's
    // "[p.N]" for PDF), so the chip stays the clean document identity
    // (filename only) — no "· 4.1 DCF Model Key Assumptions" / "· Country
    // Lookup row [1, 23]" suffix duplicated here.
    const isDocx  = ['docx', 'doc', 'word'].includes(mod)

    // Priority order: (paged docs) short clean section → timestamp → sheet+row → ...
    // The PAGE NUMBER is NOT shown on the chip — it lives inline in the answer as
    // a [p.N] reference, so the chip stays the clean document identity
    // ("apple_10k.pdf"), optionally + a short clean section heading. Long strings
    // or chunks full of digits are raw chunk text (bad section_title metadata) and
    // are skipped so the chip never shows a giant data dump.
    if (source.page != null || source.page_number != null) {
      const stRaw = String(source.section_title || source.heading || '').trim()
      if (stRaw && stRaw.length <= 45 && !/\d{3,}/.test(stRaw)) {
        suffix = ` · ${stRaw}`
      }
    } else if (isAudio || isVideo) {
      // Speaker + timestamp is rendered as an accent-colored pill at the END of
      // the answer (AudioCitations below), same pattern as XLSX sheet+row and
      // Image title — so the chip stays the clean file identity (filename only).
      suffix = ''
    } else if (source.heading && !isDocx) {
      suffix = ` · ${String(source.heading)}`
    } else if (source.section_title && !isTxt && !isDocx) {
      const st = String(source.section_title).trim()
      if (st) suffix = ` · ${st}`
    } else if (source.image_title) {
      // Image chip stays the clean file identity ("aapl-20240928_g2.jpg") —
      // NO title suffix here. The chart title is shown as an accent-colored
      // citation pill at the END of the answer (ImageCitePill), exactly like
      // the XLSX sheet+row pill and the filename-only XLSX chip below it.
    } else if (source.start_time != null) {
      // legacy field
      const ts = fmtTimestamp(source.start_time)
      if (ts) suffix = ` · ${ts}`
    }
  }

  return { isWeb, raw, label, suffix, color: getModalityColor(source, isWeb) }
}

/* XLSX sheet+row citation — same visual treatment as CitePill's "[p.26]" for
   PDF and SectionCitePill's "[4.1]" for DOCX: plain accent-colored text, no
   chip background/border, appended at the end of the answer. Shows ONLY the
   locator ("Country Lookup row [1, 23]") — the filename lives in the
   separate Sources chip area below, exactly like PDF/DOCX. */
function XlsxCitePill({ source }) {
  const sheet = source.sheet_name
  if (!sheet) return null
  const rowRange = source.row_range
  // row_range already carries its own brackets ("[1, 23]") — do NOT also wrap
  // the whole label in an outer "[...]" (that produced "[Sheet row [1, 23]]",
  // a double-bracket bug). Unlike CitePill's bare "[p.26]", the label here is
  // shown unwrapped: "Sheet row [1, 23]".
  const rr = Array.isArray(rowRange) ? `[${rowRange.join(', ')}]` : rowRange
  const label = rr ? `${sheet} row ${rr}` : sheet
  return (
    <span
      title={`Source: ${label}`}
      style={{ color: 'var(--t-accent)', fontWeight: 500 }}
    >
      {' '}{label}
    </span>
  )
}

function XlsxCitations({ sources }) {
  const xlsxSrcs = (sources || []).filter(
    s => typeof s === 'object' && ['excel', 'xlsx'].includes(s.modality) && s.sheet_name
  )
  if (xlsxSrcs.length === 0) return null
  return (
    <div className="mt-1 text-[15px] leading-relaxed">
      {xlsxSrcs.map((s, i) => <XlsxCitePill key={i} source={s} />)}
    </div>
  )
}

/* Image/chart citation — same visual treatment as XlsxCitePill (accent-colored
   text at the end of the answer). Shows the chart's own TITLE as the locator
   (e.g. "Comparison of 5-Year Cumulative Total Return"); the filename lives
   separately in the Sources chip area below, exactly like XLSX/PDF/DOCX. */
function ImageCitePill({ source }) {
  let title = String(source.image_title || '').trim()
  if (!title) return null
  // Prefer the main title line (before a "… Among <series> …" subtitle).
  const amongIdx = title.search(/\s+Among\s+/i)
  if (amongIdx > 20) title = title.slice(0, amongIdx).trim()
  if (title.length > 90) {
    const cut = title.lastIndexOf(' ', 90)
    title = (cut > 40 ? title.slice(0, cut) : title.slice(0, 90)).trim() + '…'
  }
  return (
    <span title={`Source: ${title}`} style={{ color: 'var(--t-accent)', fontWeight: 500 }}>
      {' '}{title}
    </span>
  )
}

function ImageCitations({ sources }) {
  const imgSrcs = (sources || []).filter(
    s => typeof s === 'object' && s.modality === 'image' && s.image_title
  )
  if (imgSrcs.length === 0) return null
  // De-dup by title so one chart cited by both its text+vision chunks shows once.
  const seen = new Set()
  const unique = imgSrcs.filter(s => {
    const k = String(s.image_title || '').trim()
    if (seen.has(k)) return false
    seen.add(k); return true
  })
  return (
    <div className="mt-1 text-[15px] leading-relaxed">
      {unique.map((s, i) => <ImageCitePill key={i} source={s} />)}
    </div>
  )
}

/* Compact "moment" chip for an audio/video citation — an icon, an optional
   short label, and the timestamp (accent-colored). One clean chip per cited
   moment, industry-standard (ChatGPT/Gemini/NotebookLM) style: no raw caption
   or OCR dumps. */
function MomentChip({ icon, label, ts, title }) {
  const t = fmtTimestamp(ts)
  if (!label && !t) return null
  return (
    <span
      title={title || undefined}
      className="inline-flex items-center gap-1 text-[11px] leading-4 rounded-full px-2 py-0.5 mr-1.5 mb-1 select-none max-w-full"
      style={{ background: 'var(--t-chp)', border: '1px solid var(--t-chpb)', color: 'var(--t-tx4)' }}
    >
      <span aria-hidden style={{ flexShrink: 0 }}>{icon}</span>
      {label ? <span className="truncate">{label}</span> : null}
      {t ? <span style={{ color: 'var(--t-accent)', fontWeight: 600 }}>{label ? '· ' : ''}{t}</span> : null}
    </span>
  )
}

/* Audio/video citation row — one compact chip per cited moment: a speaker chip
   for spoken content ("🗣 Tim Cook (CEO) · 3:49") and a frame chip for on-screen
   evidence ("🖼 EPS $1.85 beats $1.76 · 2:43"), falling back to "🖼 On-screen
   chart · 2:43" when no clean metric distils out. The filename lives in the
   Sources chip area below (like PDF/DOCX/XLSX/Image). */
function AudioCitations({ sources }) {
  const avSrcs = (sources || []).filter(
    s => typeof s === 'object'
      && ['audio', 'mp3', 'video', 'mp4'].includes(s.modality)
      && (s.timestamp_start != null || s.start_time != null || s.frame_timestamp != null)
  )
  if (avSrcs.length === 0) return null

  const spokenSrcs = avSrcs.filter(s => !s.is_frame)
  const frameSrcs  = avSrcs.filter(s => s.is_frame)

  // De-dup spoken by speaker+timestamp.
  const seen = new Set()
  const uniqueSpoken = spokenSrcs.filter(s => {
    const ts = s.timestamp_start != null ? s.timestamp_start : s.start_time
    const k = `${s.speaker_name || s.speaker_role || ''}|${ts}`
    if (seen.has(k)) return false
    seen.add(k); return true
  })
  // De-dup frames by timestamp.
  const fseen = new Set()
  const uniqueFrames = frameSrcs.filter(s => {
    const k = `${s.frame_timestamp != null ? s.frame_timestamp : (s.timestamp_start ?? s.start_time)}`
    if (fseen.has(k)) return false
    fseen.add(k); return true
  })

  return (
    <div className="mt-1.5 flex flex-wrap items-center">
      {uniqueSpoken.map((s, i) => {
        const name = String(s.speaker_name || '').trim()
        const role = String(s.speaker_role || '').trim()
        const speaker = (name && role && name.toLowerCase() !== role.toLowerCase())
          ? `${name} (${role})`
          : (name || role || '')
        const ts = s.timestamp_start != null ? s.timestamp_start : s.start_time
        return <MomentChip key={`s${i}`} icon="🗣" label={speaker || null} ts={ts} title="Spoken source" />
      })}
      {uniqueFrames.map((s, i) => {
        const ts = s.frame_timestamp != null ? s.frame_timestamp : (s.timestamp_start ?? s.start_time)
        const label = String(s.frame_label || '').trim() || 'On-screen chart'
        return <MomentChip key={`f${i}`} icon="🖼" label={label} ts={ts} title={s.frame_caption || 'On-screen frame'} />
      })}
    </div>
  )
}

function SourceChip({ source }) {
  const { isWeb, raw, label, suffix, color } = getSourceLabelParts(source)
  const chipClass   = "inline-flex items-start gap-1.5 text-[11px] leading-4 rounded-2xl px-2.5 py-1 mr-1.5 mb-1 select-none transition-colors max-w-full"
  const chipStyle   = { background: 'var(--t-chp)', border: '1px solid var(--t-chpb)', color: 'var(--t-tx4)' }
  const chipIcon    = <SourceIcon source={source} isWeb={isWeb} />
  const chipText    = (
    <span className="break-words">
      <span style={{ color }}>{label}</span>
      {suffix && <span style={{ color, opacity: 0.85 }}>{suffix}</span>}
    </span>
  )

  if (isWeb) {
    // Title + domain card (Perplexity/ChatGPT "Sources" style). Falls back to a
    // domain-only chip when the article title isn't available.
    const webTitle = typeof source === 'object' ? String(source.title || '').trim() : ''
    return (
      <a
        href={raw}
        target="_blank"
        rel="noopener noreferrer"
        title={webTitle || raw}
        className="flex items-start gap-2 rounded-xl px-2.5 py-1.5 w-full h-full transition-colors cursor-pointer"
        style={{ background: 'var(--t-chp)', border: '1px solid var(--t-chpb)', textDecoration: 'none' }}
        onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--t-accent)'}
        onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--t-chpb)'}
      >
        {chipIcon}
        <span className="flex flex-col min-w-0">
          {webTitle && (
            <span className="text-[11px] leading-snug font-medium break-words"
              style={{ color: 'var(--t-tx3)' }}>
              {webTitle}
            </span>
          )}
          <span className="text-[10px] leading-tight truncate" style={{ color }}>
            {label}
          </span>
        </span>
      </a>
    )
  }

  return (
    <span className={`${chipClass} cursor-default`} style={chipStyle}>
      {chipIcon}{chipText}
    </span>
  )
}

/* ── Timestamp ── */
function formatTime(ts) {
  if (!ts) return ''
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/* ── Code renderer for ReactMarkdown ── */
function CodeBlock({ inline, children }) {
  const style = {
    background: 'var(--t-inp)',
    border: '1px solid var(--t-bd4)',
    borderRadius: inline ? 4 : 8,
    padding: inline ? '1px 5px' : '10px 14px',
    fontSize: '0.85em',
    fontFamily: 'ui-monospace, Consolas, monospace',
    display: inline ? 'inline' : 'block',
    overflowX: inline ? undefined : 'auto',
    margin: inline ? 0 : '0.6em 0',
  }
  return inline
    ? <code style={style}>{children}</code>
    : <pre style={style}><code>{children}</code></pre>
}

/* ── Main component ── */
export default function MessageBubble({ message, isStreaming, dark, onRegenerate, onEdit, showSources = true, authToken, sessionId, precedingQuery }) {
  const isUser = message.role === 'user'
  const { addToast } = useToast()
  const isMobile = useIsMobile()
  const [copied, setCopied]   = useState(false)
  const [vote, setVote]       = useState(message.vote ?? null)  // 'up' | 'down' | null — restored from session
  const [isEditing, setIsEditing] = useState(false)
  const [editText, setEditText]   = useState('')

  function handleVote(newVote) {
    const next = vote === newVote ? null : newVote
    setVote(next)
    if (authToken && sessionId) {
      submitFeedback(
        authToken,
        sessionId,
        next ?? 'none',
        message.id || null,
        precedingQuery || null,
        typeof message.content === 'string' ? message.content.slice(0, 500) : null,
      )
    }
  }

  // Parse inline citations from LLM content and merge with structured sources, deduped
  const { cleanContent, inlineSources, docxSections } = parseInlineCitations(message.content)
  const allSources = deduplicateSources(message.sources, inlineSources)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content || '')
    setCopied(true)
    addToast('Copied to clipboard', 'success')
    setTimeout(() => setCopied(false), 2000)
  }

  const startEdit = () => { setEditText(message.content || ''); setIsEditing(true) }
  const cancelEdit = () => setIsEditing(false)
  const saveEdit = () => {
    const text = editText.trim()
    if (!text) return
    setIsEditing(false)
    onEdit?.(text)
  }

  /* ── User bubble ── */
  if (isUser) {
    if (isEditing) {
      return (
        <div className="flex justify-end">
          <div className="flex flex-col items-end gap-2 max-w-[72%] w-full">
            <div className="w-full rounded-2xl rounded-tr-sm px-5 py-3.5"
              style={{ background: 'var(--t-ubg)', border: '1px solid var(--t-accent)' }}>
              <textarea
                value={editText}
                onChange={e => setEditText(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); saveEdit() }
                  if (e.key === 'Escape') cancelEdit()
                }}
                autoFocus
                rows={Math.min(8, Math.max(2, editText.split('\n').length))}
                className="w-full bg-transparent outline-none resize-none text-[16px] leading-relaxed"
                style={{ color: 'var(--t-tx1)' }}
              />
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={cancelEdit}
                className="text-sm rounded-lg px-3.5 py-1.5 transition-colors"
                style={{ color: 'var(--t-tx4)' }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                Cancel
              </button>
              <button
                onClick={saveEdit}
                className="text-sm font-medium rounded-lg px-4 py-1.5 text-white transition-opacity"
                style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)' }}
                onMouseEnter={e => e.currentTarget.style.opacity = '0.85'}
                onMouseLeave={e => e.currentTarget.style.opacity = '1'}
              >
                Save & submit
              </button>
            </div>
          </div>
        </div>
      )
    }

    return (
      <div className="flex justify-end group">
        <div className="flex flex-col items-end gap-1 max-w-[72%]">
          <div className="rounded-2xl rounded-tr-sm px-5 py-3.5 text-[16px] leading-relaxed"
            style={{ background: 'var(--t-ubg)', border: '1px solid var(--t-ubd)', color: 'var(--t-tx1)' }}>
            {cleanContent}
          </div>
          <div className={`flex items-center gap-1 transition-opacity pr-0.5 ${isMobile ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}>
            {onEdit && (
              <button
                onClick={startEdit}
                className="w-6 h-6 rounded-md flex items-center justify-center transition-colors"
                style={{ color: 'var(--t-tx5)' }}
                onMouseEnter={e => { e.currentTarget.style.background = 'var(--t-hov2)'; e.currentTarget.style.color = 'var(--t-tx2)' }}
                onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--t-tx5)' }}
                title="Edit message"
              >
                <Pencil size={12} />
              </button>
            )}
            <button
              onClick={handleCopy}
              className="w-6 h-6 rounded-md flex items-center justify-center transition-colors"
              style={{ color: copied ? 'var(--t-success)' : 'var(--t-tx5)' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              title="Copy"
            >
              {copied ? <Check size={12} /> : <Copy size={12} />}
            </button>
            {message.ts && (
              <span className="text-[10px] ml-1" style={{ color: 'var(--t-tx5)' }}>{formatTime(message.ts)}</span>
            )}
          </div>
        </div>
      </div>
    )
  }

  const isEmpty = !message.content && message.pending

  /* ── Bot bubble ── */
  return (
    <div className="flex justify-start group">

      <div className="flex-1 min-w-0 max-w-[84%]">
        {/* Bubble */}
        <div className="rounded-2xl rounded-tl-sm px-5 py-3.5 text-[16px] leading-relaxed"
          style={
            message.error
              ? { background: 'var(--t-err-bg)', border: '1px solid var(--t-err-bd)', color: 'var(--t-err-tx)' }
              : { background: 'var(--t-bbg)', border: '1px solid var(--t-bbd)', color: 'var(--t-tx2)' }
          }
        >
          {isEmpty ? (
            <span className="italic text-sm" style={{ color: 'var(--t-tx6)' }}>Thinking…</span>
          ) : (
            <>
              <div className={`prose-chat ${isStreaming ? 'streaming-cursor' : ''}`}>
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code: ({ inline, children }) => (
                      <CodeBlock inline={inline}>{children}</CodeBlock>
                    ),
                    p: ({ children }) => <p>{injectPageCites(children)}</p>,
                    li: ({ children }) => <li>{injectPageCites(children)}</li>,
                  }}
                >
                  {cleanContent}
                </ReactMarkdown>
              </div>

              {/* XLSX sheet+row / DOCX section citation — colored text at the end
                  of the answer, no "Sources:" label. The filename itself
                  renders in the Sources chip area below. */}
              {showSources && !isStreaming && !isNoInfoResponse(cleanContent) && (
                <>
                  <XlsxCitations sources={allSources} />
                  <DocxCitations sections={docxSections} />
                  <ImageCitations sources={allSources} />
                  {/* Raw (un-deduped) sources: audio/video citations need the
                      frame source, which shares the video's filename and would
                      otherwise be collapsed by the basename dedup used for chips. */}
                  <AudioCitations sources={message.sources && message.sources.length ? message.sources : allSources} />
                </>
              )}

              {/* Numeric verification badges — rendered after stream completes */}
              {!isStreaming && message.verification_results && message.verification_results.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {message.verification_results.map((v, i) => {
                    const badge = v.grounded ? '✅' : v.approximate ? '⚠' : '🚩'
                    const title = v.grounded ? 'Grounded in source' : v.approximate ? 'Approximate match' : 'Unverified — not found in sources'
                    return (
                      <span
                        key={i}
                        title={title}
                        className="inline-flex items-center gap-1 text-[11px] leading-4 rounded-xl px-2 py-0.5 font-mono select-none"
                        style={{ background: 'var(--t-chp)', border: '1px solid var(--t-chpb)', color: 'var(--t-tx4)', cursor: 'help' }}
                      >
                        {badge} {v.number}
                      </span>
                    )
                  })}
                </div>
              )}

              {/* Inline FinanceTable for XLSX sources with markdown_repr */}
              {!isStreaming && allSources.some(s => typeof s === 'object' && ['excel','xlsx'].includes(s.modality) && s.markdown_repr) && (
                <div className="mt-3 space-y-2">
                  {allSources
                    .filter(s => typeof s === 'object' && ['excel','xlsx'].includes(s.modality) && s.markdown_repr)
                    .map((s, i) => (
                      <FinanceTable
                        key={i}
                        markdown={s.markdown_repr}
                        title={s.sheet_name || s.source || ''}
                        rowRange={s.row_range}
                      />
                    ))
                  }
                </div>
              )}

              {/* Sources inside the bubble — hidden when answer says no info was found.
                  Web sources render in an aligned 2-column grid (Perplexity/ChatGPT
                  "Sources" style) so card edges line up; other chips keep the
                  compact flex-wrap row. Chips show the clean document identity
                  (filename, +short section for paged docs) — page/sheet+row/
                  section locators live inline in the answer instead (see
                  injectPageCites / XlsxCitations above), never duplicated here. */}
              {showSources && allSources.length > 0 && !isStreaming && !isNoInfoResponse(cleanContent) && (() => {
                const webSrcs   = allSources.filter(s => typeof s === 'object' && s.modality === 'web')
                const otherSrcs = allSources.filter(s => !(typeof s === 'object' && s.modality === 'web'))
                return (
                  <div className="mt-3 pt-2.5" style={{ borderTop: '1px solid var(--t-bbd)' }}>
                    {webSrcs.length > 0 && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 items-stretch">
                        {webSrcs.map((src, i) => <SourceChip key={i} source={src} />)}
                      </div>
                    )}
                    {otherSrcs.length > 0 && (
                      <div className={`flex flex-wrap ${webSrcs.length > 0 ? 'mt-2' : ''}`}>
                        {otherSrcs.map((src, i) => <SourceChip key={i} source={src} />)}
                      </div>
                    )}
                  </div>
                )
              })()}
            </>
          )}
        </div>

        {/* Action row — always visible on mobile, hover-reveal on desktop */}
        {!isEmpty && !isStreaming && (
          <div className={`flex items-center gap-1 mt-1.5 transition-opacity ${isMobile ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}>
            {/* Copy */}
            <button
              onClick={handleCopy}
              className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
              style={{ color: copied ? 'var(--t-success)' : 'var(--t-tx5)' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              title="Copy"
            >
              {copied ? <Check size={13} /> : <Copy size={13} />}
            </button>

            {/* Thumbs up */}
            <button
              onClick={() => handleVote('up')}
              className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
              style={{ color: vote === 'up' ? 'var(--t-accent)' : 'var(--t-tx5)' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              title="Good response"
            >
              <ThumbsUp size={13} fill={vote === 'up' ? 'currentColor' : 'none'} />
            </button>

            {/* Thumbs down */}
            <button
              onClick={() => handleVote('down')}
              className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
              style={{ color: vote === 'down' ? 'var(--t-danger)' : 'var(--t-tx5)' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              title="Bad response"
            >
              <ThumbsDown size={13} fill={vote === 'down' ? 'currentColor' : 'none'} />
            </button>

            {/* Regenerate */}
            {onRegenerate && (
              <button
                onClick={onRegenerate}
                className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
                style={{ color: 'var(--t-tx5)' }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                title="Regenerate response"
              >
                <RotateCcw size={13} />
              </button>
            )}

            {/* Timestamp */}
            {message.ts && (
              <span className="ml-1 text-[10px] font-mono" style={{ color: 'var(--t-tx6)' }}>
                {formatTime(message.ts)}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
