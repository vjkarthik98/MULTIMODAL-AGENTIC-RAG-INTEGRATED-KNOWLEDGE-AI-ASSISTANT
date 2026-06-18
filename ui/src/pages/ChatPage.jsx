import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Square, Sparkles, ChevronDown, Menu, Upload, WifiOff, MoreVertical,
         FileText, Image, Sheet, FileVideo, FileAudio, FileType, LetterText, File as FileIcon } from 'lucide-react'
import Sidebar from '../components/Sidebar'
import SettingsModal from '../components/SettingsModal'
import MessageBubble from '../components/MessageBubble'
import TypingIndicator from '../components/TypingIndicator'
import GuestBanner from '../components/GuestBanner'
import ConversionModal from '../components/ConversionModal'
import { streamQuery, queryMeta, getChatSession, patchLastMessage, updateChatSession } from '../api/client'
import { useToast } from '../context/ToastContext'
import useIsMobile from '../hooks/useIsMobile'

const CHAT_ICON = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>
)

const _REFUSALS = [
  'could not find', 'cannot find', 'no relevant information',
  'not provided in', 'not mentioned in', "i don't know",
  'i do not know', "couldn't find", 'not available in',
  'not in the provided', 'not found in',
]
// Kept only as an end-of-stream safety net: if the model ever ignores the
// extractive prompt and still refuses on a doc-present query, the parallel
// meta answer replaces it after the stream ends. This is the rare path now,
// not the common one — the backend handles refusal deterministically.
const isRefusal = (t) => !t || _REFUSALS.some(p => t.toLowerCase().includes(p))

// Mirror the backend's _EXPLICIT_WEB_PHRASES hard rule so we can suppress
// file source chips from the stream before meta confirms the decision.
const _WEB_PHRASES = [
  'from web', 'from the web', 'search web', 'search the web',
  'get from web', 'get it from web', 'web search', 'find online',
  'search online', 'look online', 'from internet', 'from the internet',
  'find on the internet', 'look it up',
]
const isExplicitWebQuery = (q) => {
  const lc = (q || '').toLowerCase()
  return _WEB_PHRASES.some(p => lc.includes(p))
}

const PLACEHOLDERS = [
  'Ask anything about your files…',
]

function fileModalityIcon(filename) {
  const ext = (filename || '').split('.').pop().toUpperCase()
  if (['PNG','JPG','JPEG','GIF','WEBP'].includes(ext))      return <Image    size={13} style={{ color: '#2dd4bf', flexShrink: 0 }} />
  if (['XLS','XLSX','CSV'].includes(ext))                   return <Sheet    size={13} style={{ color: '#4ade80', flexShrink: 0 }} />
  if (ext === 'PDF')                                        return <FileText size={13} style={{ color: '#f87171', flexShrink: 0 }} />
  if (['MP4','MOV','AVI','MKV','WEBM'].includes(ext))       return <FileVideo  size={13} style={{ color: '#60a5fa', flexShrink: 0 }} />
  if (['MP3','WAV','M4A','OGG','FLAC'].includes(ext))       return <FileAudio  size={13} style={{ color: '#c084fc', flexShrink: 0 }} />
  if (ext === 'TXT')                                        return <FileType   size={13} style={{ color: '#94a3b8', flexShrink: 0 }} />
  if (['DOC','DOCX'].includes(ext))                         return <LetterText size={13} style={{ color: '#60a5fa', flexShrink: 0 }} />
  return <FileIcon size={13} style={{ color: 'var(--t-tx5)', flexShrink: 0 }} />
}

function buildSuggestions(kbFiles) {
  if (!kbFiles.length) return []
  return kbFiles.slice(0, 3).map((f, i) => {
    const name = f.filename.replace(/\.[^.]+$/, '')
    if (i === 0) return `Summarise ${name}`
    if (i === 1) return `What does the ${name} show?`
    return `Key trends in ${name}?`
  })
}

export default function ChatPage({ auth, onLogout, dark, onToggleTheme, onStreamingChange, onGuestConvert, onGuestGoogleConvert, onGuestLimitsUpdate, onShowLogin }) {
  const [messages, setMessages]           = useState([])
  const [input, setInput]                 = useState('')
  const [kbFiles, setKbFiles]             = useState([])
  const [streaming, setStreaming]         = useState(false)
  const [showConversionModal, setShowConversionModal] = useState(false)
  const [conversionTrigger, setConversionTrigger]     = useState('voluntary')
  const [streamingId, setStreamingId]     = useState(null)
  const [autoScroll, setAutoScroll]       = useState(true)
  const [showScrollBtn, setShowScrollBtn] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => typeof window !== 'undefined' && window.innerWidth < 960
  )
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const isMobile = useIsMobile()

  // Auto-collapse/expand sidebar as viewport crosses 960px
  useEffect(() => {
    const mq = window.matchMedia('(max-width: 959px)')
    const handler = e => { if (!isMobile) setSidebarCollapsed(e.matches) }
    mq.addEventListener('change', handler)
    return () => mq.removeEventListener('change', handler)
  }, [isMobile])
  const [sessionId, setSessionId]         = useState(() => crypto.randomUUID())
  const [selectedFile, setSelectedFile]   = useState(null)
  const [showFilePicker, setShowFilePicker] = useState(false)
  const filePickerRef                     = useRef(null)
  const [loadingSession, setLoadingSession] = useState(false)
  const [inputFocused, setInputFocused]   = useState(false)
  const [placeholderIdx, setPlaceholderIdx] = useState(0)
  const [settingsOpen, setSettingsOpen]   = useState(false)
  const [showSources, setShowSources]     = useState(() => localStorage.getItem('magik_show_sources') !== 'false')
  const [chatDragOver, setChatDragOver]   = useState(false)
  const [isOnline, setIsOnline]           = useState(() => navigator.onLine)
  const [settingsSection, setSettingsSection] = useState('account')
  const chatUploadRef                     = useRef(null)
  const [chatMenuOpen, setChatMenuOpen]   = useState(false)
  const chatMenuRef                       = useRef(null)
  const [historyClearedAt, setHistoryClearedAt] = useState(0)
  const [staleSessionId, setStaleSessionId]     = useState(null)

  useEffect(() => { localStorage.setItem('magik_show_sources', String(showSources)) }, [showSources])

  const { addToast } = useToast()

  const scrollAreaRef  = useRef(null)
  const messagesEndRef = useRef(null)
  const inputRef       = useRef(null)
  const abortRef       = useRef(null)

  // Rotate placeholder every 3s when input is empty and not focused
  useEffect(() => {
    if (input || inputFocused) return
    const t = setInterval(() => setPlaceholderIdx(i => (i + 1) % PLACEHOLDERS.length), 3000)
    return () => clearInterval(t)
  }, [input, inputFocused])

  // Reset textarea height when input is cleared (after send / new chat)
  useEffect(() => {
    if (!input && inputRef.current) {
      inputRef.current.style.height = 'auto'
    }
  }, [input])

  // Online / offline banner
  useEffect(() => {
    const handleOnline  = () => { setIsOnline(true);  addToast('Connection restored', 'success') }
    const handleOffline = () => { setIsOnline(false); addToast('You are offline', 'error') }
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  // Favicon pulse while streaming
  useEffect(() => {
    const link = document.querySelector("link[rel~='icon']")
    if (!link) return
    if (streaming) {
      const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#863bff" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M9 13a4.5 4.5 0 0 0 3-4"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/><path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M12 13h4"/><path d="M12 18h6a2 2 0 0 1 2 2v1"/><path d="M12 8h8"/><path d="M16 8V5a2 2 0 0 1 2-2"/><circle cx="16" cy="13" r=".5" fill="#863bff"/><circle cx="18" cy="3" r=".5" fill="#863bff"/><circle cx="20" cy="21" r=".5" fill="#863bff"/><circle cx="20" cy="8" r=".5" fill="#863bff"/><circle cx="20" cy="8" r="3" fill="none" stroke="#a78bfa" opacity="0.8"><animate attributeName="r" values="3;5;3" dur="1.2s" repeatCount="indefinite"/><animate attributeName="opacity" values="0.8;0;0.8" dur="1.2s" repeatCount="indefinite"/></circle></svg>`
      link.href = `data:image/svg+xml;base64,${btoa(svg)}`
    } else {
      link.href = '/favicon.svg'
    }
  }, [streaming])

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      const mod = e.metaKey || e.ctrlKey
      if (mod && e.key === 'k') { e.preventDefault(); inputRef.current?.focus() }
      if (mod && e.shiftKey && e.key === 'N') { e.preventDefault(); handleNewChat() }
      if (e.key === 'Escape') { setMobileSidebarOpen(false); setSidebarCollapsed(true) }
      if (e.key === '?' && !e.target.closest('input, textarea')) {
        e.preventDefault()
        setSettingsSection('shortcuts')
        setSettingsOpen(true)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  // Close file picker when clicking outside it
  useEffect(() => {
    if (!showFilePicker) return
    const handler = (e) => {
      if (filePickerRef.current && !filePickerRef.current.contains(e.target)) {
        setShowFilePicker(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [showFilePicker])

  // Close three-dot chat menu on outside click
  useEffect(() => {
    if (!chatMenuOpen) return
    const handler = (e) => {
      if (chatMenuRef.current && !chatMenuRef.current.contains(e.target)) {
        setChatMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [chatMenuOpen])

  const handleScroll = () => {
    const el = scrollAreaRef.current
    if (!el) return
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 120
    setAutoScroll(nearBottom)
    setShowScrollBtn(!nearBottom)
  }

  useEffect(() => {
    if (autoScroll) messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, autoScroll])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    setAutoScroll(true)
    setShowScrollBtn(false)
  }

  // Remove duplicate sentences/paragraphs the LLM sometimes repeats verbatim.
  const dedupText = (text) => {
    const sentences = text.split(/(?<=[.!?])\s+/)
    const seen = new Set()
    const out = []
    for (const s of sentences) {
      const key = s.trim().toLowerCase().replace(/\s+/g, ' ')
      if (key.length < 20 || !seen.has(key)) {
        seen.add(key)
        out.push(s)
      }
    }
    return out.join(' ')
  }

  // Strip LLM format artifacts from the streamed text so users never see
  // raw numeric citation indices, format fields, or summary epilogues.
  const cleanStreamText = (raw) => dedupText(raw
    // "<end of output>" and everything after it — model echoed a prompt terminator
    .replace(/<end\s+of\s+output>[\s\S]*/gi, '')
    // "1. (apple_q4_2023_earnings_3) Revenue..." — model copied numbered context chunk labels verbatim
    .replace(/\b\d+\.\s+\(\w[\w._-]{2,80}\)\s*/g, '')
    .replace(/\(?\s*[\d.]+\s+is\s+(?:the\s+)?(?:most|least)\s+confident\s*\)?/gi, '')   // "(1.0 is most confident)" / "0 is least confident)"
    .replace(/\(?\s*\d+\s*=\s*(?:most|least)\s+confident\s*\)?/gi, '')                 // "(1 = most confident)" / "0 = least confident)"
    .replace(/\(\s*confidence\s*[:=]?\s*[\d.]+\s*\)/gi, '')                            // "(confidence: 0.9)"
    .replace(/^\s*(?:Answer|Response)\s*:\s*/i, '')
    .replace(/^Write\s+your\s+(?:complete\s+)?answer\s+below[^:\n]*:\s*\n?/i, '') // echoed prompt instruction
    .replace(/^Begin\s+with\s+a\s+(?:full|complete)\s+sentence\s*:?\s*\n?/i, '')  // echoed prompt instruction
    .replace(/^\s*(?:Timeline|Overview|Summary|Context)\s*:\s*\n?/i, '')   // "Timeline:" header label
    .replace(/^\s*Source\s*:\s*<source[^>]*>\s*\n?/i, '')                  // "Source: <source number>"
    .replace(/\s*Source\s*:\s*$/im, '')                                     // Trailing bare "Source:" label
    .replace(/^\s*\[?\s*Answer\s*\]?\s*:\s*/i, '')                         // Leading "Answer:" or "[Answer:"
    .replace(/\n\s*\[?\s*Answer\s*\]?\s*:\s*/gi, '\n')                     // Mid-text "Answer:" on its own line
    .replace(/\n?Sources?\s+differ\s*:[^\n]*/gi, '')                       // "Sources differ: ..." leak
    .replace(/\n?The\s+(?:graph|chart|image)\s+is\s+identical\s+to[^\n]*/gi, '')  // "The graph is identical to..." meta-leak
    .replace(/\n?Therefore,?\s+the\s+information\s+from\s+is[^\n]*/gi, '')  // "Therefore, the information from is..." partial-strip artifact
    .replace(/^\s*,\s+/, '')                                                // leading ", " left after prefix stripped
    .replace(/\s*\[\d+(?:,\s*\d+)*\]/g, '')                               // [1,3] [1,2,4]
    .replace(/\s+\(\d+\)(?=[\s.,;!?]|$)/g, '')                           // "(1)" footnote markers after words
    .replace(/\s*\[[^\]]{3,120}\.(txt|pdf|docx?|xlsx?|csv|mp3|mp4|wav|png|jpg|jpeg)\s*[^\]]{0,60}\]/gi, '')  // inline [filename.ext] cite tags
    .replace(/\s*\[[^\]\n]*<[A-Z_]{2,20}>[^\]\n]*\]/g, '')                // mangled cite tag, e.g. [aapl_def14a_<URL>cx] (scrubber ate the .docx)
    .replace(/\n?(Answer Tags|Confidence|Sources Used|Reasoning)\s*:\s*[^\n]*/gi, '')
    .replace(/\n?Sources?\s*:\s*\([^)]{1,40}\)[^\n]*/gi, '')             // "Sources: (tag) description" cite block
    .replace(/\n?Sources?\s*:\s*\[[^\]]{1,80}\][^\n]*/gi, '')            // "Sources: [tag] description" cite block
    .replace(/\n?\s*Sources?\s*:\s*$/im, '')                              // trailing bare "Sources:" the LLM appended
    .replace(/\n?Therefore,?\s+the\s+answer\s+is\s*:?\s*<text>[\s\S]*?<\/text>/gi, '')
    .replace(/\n?Therefore,?\s+the\s+answer\s+is\s*:?[^\n]*/gi, '')
    .replace(/<(?:PERSON|LOCATION|ORG|NRP|GPE|DATE_TIME|AGE|ID|URL|IP_ADDRESS|US_SSN|CREDIT_CARD|PHONE_NUMBER|EMAIL_ADDRESS)>/gi, '')
    .replace(/\s*\[No source[^\]]{0,120}\]/gi, '')                         // "[No source for X]" cite-miss artifacts
    .replace(/\s*\(\s*Item\s+\d+[^()]{0,80}\)/g, '')                      // "(Item 1. Business. Overview)"
    .replace(/\s*\(\s*[A-Z][a-zA-Z ]{1,50}\)\s*(?=[.,;!?]|$)/g, '')      // "(Overview)" "(Human capital)" as sentence-end section refs
    .replace(/\s*\(\s*[,;]?\s*\)\s*/g, ' ')   // empty parens "(, )" or "( )" left after inner content stripped
    .replace(/ +\.(?=[\s,;!?]|$)/g, '.')      // "legal risks ." → "legal risks."
    .replace(/([.!?])\s*[,;]\s*$/g, '$1')   // "Management. ," → "Management."
    .replace(/[,;]\s*$/, '')                  // bare trailing comma/semicolon
    .trim())

  const handleSend = useCallback(async (overrideText, { skipUserMessage = false, noCache = false } = {}) => {
    const text = (overrideText ?? input).trim()
    if (!text || streaming) return
    setInput('')
    setAutoScroll(true)

    const botId = crypto.randomUUID()
    const ts    = Date.now()
    setMessages(prev => skipUserMessage
      ? [...prev, { role: 'assistant', content: '', id: botId, pending: true, ts }]
      : [
          ...prev,
          { role: 'user', content: text, ts },
          { role: 'assistant', content: '', id: botId, pending: true, ts: ts + 1 },
        ]
    )
    setStreaming(true)
    setStreamingId(botId)
    onStreamingChange?.(true)

    // Auto-name the chat from the first user message
    if (!skipUserMessage && messages.filter(m => m.role === 'user').length === 0) {
      const title = text.length > 40 ? text.slice(0, 38).trimEnd() + '…' : text
      updateChatSession(auth.token, sessionId, { title }).catch(() => {})
    }

    // KB-empty guard — respond immediately without a backend call to prevent
    // hallucinated answers when the user hasn't uploaded any files yet.
    if (kbFiles.length === 0) {
      const _GREETING = /^(hi+|hello+|hey+|howdy|hiya|greetings|good\s+(morning|afternoon|evening|day))[\s!.,?]*$/i
      const reply = _GREETING.test(text)
        ? `Hello! I'm your AI knowledge assistant.\n\nTo get started, please **upload your documents** using the Files panel in the left sidebar. I support:\n- PDFs, Word documents, Excel spreadsheets\n- Images, audio recordings, videos\n\nOnce your files are uploaded, I'll be ready to answer any questions about your content.`
        : `Your knowledge base is currently **empty**. Please upload documents using the **Files** panel in the left sidebar before asking questions.\n\nI can work with PDFs, Word files, Excel spreadsheets, images, audio, and video. Once your files are uploaded, I'll be able to provide accurate, source-backed answers.`
      setMessages(prev => prev.map(m =>
        m.id === botId ? { ...m, content: reply, pending: false, streaming: false } : m
      ))
      setStreaming(false)
      setStreamingId(null)
      onStreamingChange?.(false)
      return
    }

    const controller = new AbortController()
    abortRef.current = controller
    let fullText = ''
    let streamedSources = null
    let refused = false

    // File-scope filter: if user selected a specific file, restrict retrieval to it.
    const fileSources = selectedFile ? [selectedFile.filename] : null
    // Cache key is query-only — bypass it whenever a file scope is active so a
    // scoped "explain" never returns a cached answer from an unscoped "explain".
    const effectiveNoCache = noCache || !!fileSources

    // The stream is the single answer path: it emits tokens live, then a
    // "replace" event with the canonical guarded answer and a "sources" event.
    // queryMeta (the full non-streaming pipeline — a second LLM generation) is
    // only called lazily on the refusal-fallback path below, so the GPU does
    // one generation per message instead of two.

    // Animate a completed string into the bubble letter-by-letter (8ms/char,
    // matching the backend's re-chunk pacing). Used for the meta answer when
    // the streaming model refused — so the fallback still feels like live
    // streaming instead of appearing all at once.
    const streamTextIntoBubble = async (full) => {
      const clean = cleanStreamText(full)
      for (let i = 1; i <= clean.length; i++) {
        if (controller.signal.aborted) break
        const shown = clean.slice(0, i)
        setMessages(prev => prev.map(m =>
          m.id === botId ? { ...m, content: shown, pending: false } : m
        ))
        await new Promise(r => setTimeout(r, 8))
      }
      setMessages(prev => prev.map(m =>
        m.id === botId ? { ...m, content: clean, pending: false } : m
      ))
    }

    try {
      const res = await streamQuery(auth.token, text, sessionId, controller.signal, noCache, fileSources)
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.detail || `Server error ${res.status}`)
      }

      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = '', done = false

      while (!done) {
        const { done: readerDone, value } = await reader.read()
        if (readerDone) break
        buf += decoder.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop()

        for (const block of parts) {
          for (const line of block.split('\n')) {
            if (!line.startsWith('data: ')) continue
            const raw = line.slice(6)
            if (raw === '[DONE]' || raw === '[Stream interrupted]') { done = true; break }
            let parsed
            try { parsed = JSON.parse(raw) } catch { parsed = raw }
            // Refusal signal: the streaming model declined despite relevant
            // docs. Keep the typing indicator (don't render the refusal text)
            // and stream the accurate meta answer once the stream ends.
            if (parsed && typeof parsed === 'object' && parsed.__type__ === 'refusal') {
              refused = true
              continue
            }
            // Canonical guarded answer — swap the streamed buffer for it. The
            // visible text rarely changes (client-side cleaning already strips
            // citation tags); this guarantees guard repairs always land.
            if (parsed && typeof parsed === 'object' && parsed.__type__ === 'replace') {
              fullText = parsed.data || fullText
              if (!refused) {
                setMessages(prev => prev.map(m =>
                  m.id === botId ? { ...m, content: cleanStreamText(fullText), pending: false } : m
                ))
              }
              continue
            }
            // Sources event emitted by the stream after text is complete.
            // For explicit web queries keep only web chips — file sources
            // must never show when the answer came from the internet.
            if (parsed && typeof parsed === 'object' && parsed.__type__ === 'sources') {
              streamedSources = parsed.data || []
              const visibleSources = isExplicitWebQuery(text)
                ? streamedSources.filter(s => s && s.modality === 'web')
                : streamedSources
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, sources: visibleSources } : m
              ))
              continue
            }
            // Guest query limit reached — show conversion modal
            if (parsed && typeof parsed === 'object' && parsed.__type__ === 'guest_limit') {
              done = true
              setMessages(prev => prev.filter(m => m.id !== botId))  // remove empty bot bubble
              setConversionTrigger(parsed.limit_type === 'uploads' ? 'upload_limit' : 'query_limit')
              setShowConversionModal(true)
              // Sync zero count to parent so banner updates immediately
              if (onGuestLimitsUpdate) onGuestLimitsUpdate({ queriesLeft: 0 })
              break
            }
            const token = typeof parsed === 'string' ? parsed : raw
            if (token && !refused) {
              fullText += token
              // Letter-by-letter from char 1. A genuine "no documents" case is
              // the correct canonical message and streams normally; a wrong
              // refusal never reaches here (backend sends a refusal signal
              // instead), so there is no flash to hide.
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, content: cleanStreamText(fullText), pending: false } : m
              ))
            }
          }
          if (done) break
        }
      }

      const streamedAnswer = cleanStreamText(fullText)

      // Fall back to the meta answer when the backend signalled a refusal OR the
      // streamed answer cleaned down to empty/refusal. The empty case is the
      // bug behind the blank bubble: the model sometimes emits only a citation
      // tag or a bare "Answer:" label, which passes the backend's non-empty
      // check but strips to "" here. isRefusal("") is true, so this catches it.
      // The meta answer is the same one persisted to redis/mongo, so the user
      // sees the real response instead of an empty bubble.
      if (refused || isRefusal(streamedAnswer)) {
        // Reset to the typing indicator while we wait for the meta answer. In
        // the refused case the bubble was already showing dots; in the empty
        // case it briefly showed a blank bubble, so this removes that flash.
        setMessages(prev => prev.map(m =>
          m.id === botId ? { ...m, content: '', pending: true } : m
        ))
        // Lazily run the full meta pipeline (CrossEncoder + CoT) — only on this
        // fallback path, so the common case never pays a second LLM generation.
        const meta = await queryMeta(auth.token, text, sessionId, effectiveNoCache, fileSources).catch(() => null)
        const metaAnswer = cleanStreamText(meta?.answer || '')
        const finalText = metaAnswer ||
          'I could not find relevant information in your knowledge base to answer this question.'
        const metaSources = (meta?.sources?.length > 0) ? meta.sources : (streamedSources || [])
        setMessages(prev => prev.map(m =>
          m.id === botId ? { ...m, sources: metaSources } : m
        ))
        await streamTextIntoBubble(finalText)
        // Stamp botId onto the DB message so votes on refusal-path answers persist.
        patchLastMessage(auth.token, sessionId, finalText, metaSources, botId)
      } else {
        // Lock in the streamed answer immediately.
        setMessages(prev => prev.map(m =>
          m.id === botId ? {
            ...m,
            content: streamedAnswer,
            sources: streamedSources || [],
            pending: false, streaming: false,
          } : m
        ))

        // The backend persisted the canonical turn (Mongo + memory) before
        // sending [DONE]. Patch it with the client-cleaned text and stamp
        // botId so votes persist and reload shows exactly what the user saw.
        if (streamedAnswer && !isRefusal(streamedAnswer)) {
          patchLastMessage(auth.token, sessionId, streamedAnswer, streamedSources || [], botId)
        }
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        setMessages(prev => prev.map(m =>
          m.id === botId
            ? { ...m, content: fullText || '_Generation stopped._', pending: false, streaming: false }
            : m
        ))
      } else {
        setMessages(prev => prev.map(m =>
          m.id === botId
            ? { ...m, content: `Something went wrong: ${err.message}`, error: true, pending: false, streaming: false }
            : m
        ))
      }
    } finally {
      abortRef.current = null
      setStreaming(false)
      setStreamingId(null)
      onStreamingChange?.(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [input, streaming, auth.token, sessionId, selectedFile])

  const handleStop = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  // Regenerate: re-send the last user message
  const handleRegenerate = useCallback(() => {
    const lastUser = [...messages].reverse().find(m => m.role === 'user')
    if (!lastUser || streaming) return
    // Remove the last assistant message before regenerating
    setMessages(prev => {
      const lastBotIdx = [...prev].map((m,i) => m.role === 'assistant' ? i : -1).filter(i => i >= 0).pop()
      return lastBotIdx != null ? prev.filter((_, i) => i !== lastBotIdx) : prev
    })
    handleSend(lastUser.content, { skipUserMessage: true, noCache: true })
  }, [messages, streaming, handleSend])

  // Edit a past user message: drop everything from that point on, then resend the edited text
  const handleEditMessage = useCallback((index, newText) => {
    if (streaming) return
    setMessages(prev => prev.slice(0, index))
    handleSend(newText)
  }, [streaming, handleSend])

const handleNewChat = () => {
    setMessages([])
    setInput('')
    setStreamingId(null)
    setAutoScroll(true)
    setShowScrollBtn(false)
    setSessionId(crypto.randomUUID())
    setSelectedFile(null)
    setShowFilePicker(false)
    setTimeout(() => inputRef.current?.focus(), 50)
  }


  // Open a chat from Recents — fetches its saved transcript and switches to it
  const handleLoadSession = useCallback(async (targetId) => {
    if (streaming || loadingSession || !targetId || targetId === sessionId) return
    setLoadingSession(true)
    try {
      const session = await getChatSession(auth.token, targetId)
      if (!session) {
        addToast('That chat is no longer available', 'error')
        setStaleSessionId(targetId)
        return
      }
      const loaded = (session.messages || []).map((m, i) => ({
        role:    m.role,
        content: m.content,
        ts:      m.timestamp ? new Date(m.timestamp).getTime() : Date.now(),
        id:      m.msg_id || `${targetId}-${i}`,
        sources: Array.isArray(m.sources) ? m.sources : [],
        vote:    m.vote ?? null,
      }))
      setMessages(loaded)
      setSessionId(targetId)
      setInput('')
      setStreamingId(null)
      setAutoScroll(true)
      setShowScrollBtn(false)
      setSelectedFile(null)
      setShowFilePicker(false)
      setTimeout(() => inputRef.current?.focus(), 50)
    } catch (err) {
      addToast(err.message || 'Failed to load chat', 'error')
    } finally {
      setLoadingSession(false)
    }
  }, [streaming, loadingSession, sessionId, auth.token, addToast])

  const suggestions    = buildSuggestions(kbFiles)

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--t-bg)' }}>

      {/* Mobile backdrop — tap anywhere outside sidebar to close */}
      {isMobile && mobileSidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/60"
          onClick={() => setMobileSidebarOpen(false)}
        />
      )}

      {/* Sidebar — inline on desktop, slide-in overlay on mobile */}
      <div className={
        isMobile
          ? `fixed inset-y-0 left-0 z-40 transition-transform duration-300 ease-in-out ${mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`
          : `flex-shrink-0 transition-all duration-300 ease-in-out overflow-hidden ${sidebarCollapsed ? 'w-[72px]' : 'w-80'}`
      }>
        <Sidebar
          auth={auth}
          kbFiles={kbFiles}
          setKbFiles={setKbFiles}
          onLogout={onLogout}
          onNewChat={isMobile ? () => { handleNewChat(); setMobileSidebarOpen(false) } : handleNewChat}
          currentSessionId={sessionId}
          onSelectSession={isMobile ? (id) => { handleLoadSession(id); setMobileSidebarOpen(false) } : handleLoadSession}
          streaming={streaming}
          collapsed={isMobile ? false : sidebarCollapsed}
          onToggleCollapse={isMobile ? () => setMobileSidebarOpen(false) : () => setSidebarCollapsed(c => !c)}
          dark={dark}
          onToggleTheme={onToggleTheme}
          onOpenSettings={() => { setSettingsOpen(true); if (isMobile) setMobileSidebarOpen(false) }}
          onSetUploadHandler={chatUploadRef}
          historyClearedAt={historyClearedAt}
          staleSessionId={staleSessionId}
          onGuestUploadLimit={auth?.isGuest ? () => { setConversionTrigger('upload_limit'); setShowConversionModal(true) } : undefined}
          onShowLogin={onShowLogin}
        />
      </div>

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => { setSettingsOpen(false); setSettingsSection('account') }}
        initialSection={settingsSection}
        auth={auth}
        onLogout={onLogout}
        dark={dark}
        onToggleTheme={onToggleTheme}
        kbFiles={kbFiles}
        setKbFiles={setKbFiles}
        sessionId={sessionId}
        showSources={showSources}
        setShowSources={setShowSources}
        onClearConversation={() => { handleNewChat(); setSettingsOpen(false) }}
        onClearAllHistory={() => { handleNewChat(); setHistoryClearedAt(Date.now()); setSettingsOpen(false) }}
      />

      {/* Main column */}
      <div
        className="flex-1 flex flex-col min-w-0 relative"
        style={{ background: 'var(--t-sur)' }}
        onDragOver={e => { e.preventDefault(); setChatDragOver(true) }}
        onDragLeave={e => { if (!e.currentTarget.contains(e.relatedTarget)) setChatDragOver(false) }}
        onDrop={e => {
          e.preventDefault(); setChatDragOver(false)
          const files = Array.from(e.dataTransfer.files)
          if (files.length) chatUploadRef.current?.(files)
        }}
      >
        {/* Drag-over overlay */}
        {chatDragOver && (
          <div className="absolute inset-0 z-50 flex flex-col items-center justify-center pointer-events-none rounded-none"
            style={{ background: 'rgba(139,92,246,0.10)', border: '2px dashed #8b5cf6' }}>
            <Upload size={36} style={{ color: '#8b5cf6', marginBottom: 12 }} />
            <p className="text-lg font-semibold" style={{ color: '#8b5cf6' }}>Drop files to upload</p>
          </div>
        )}

        {/* Offline banner */}
        {!isOnline && (
          <div className="flex-shrink-0 flex items-center gap-2 px-4 py-2 text-sm font-medium"
            style={{ background: '#b91c1c', color: '#fff' }}>
            <WifiOff size={14} />
            No internet connection — responses may be unavailable
          </div>
        )}

        {/* Top bar — hamburger (mobile) + guest auth buttons + three-dot menu */}
        {(isMobile || messages.length > 0 || auth?.isGuest) && (
          <div className="flex items-center justify-between px-3 pt-3 pb-1 flex-shrink-0">
            {isMobile ? (
              <button
                onClick={() => setMobileSidebarOpen(true)}
                className="w-9 h-9 flex items-center justify-center rounded-xl"
                style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx4)' }}
                aria-label="Open sidebar"
              >
                <Menu size={18} />
              </button>
            ) : <span />}

            <div className="flex items-center gap-2">
              {/* Guest auth buttons — Log in (outline pill) + Sign up for free (filled pill) */}
              {auth?.isGuest && (
                <>
                  <button
                    type="button"
                    onClick={onShowLogin}
                    className="text-sm font-semibold px-5 py-1.5 rounded-full transition-colors"
                    style={{ border: '1px solid var(--t-bd4)', color: 'var(--t-tx1)', background: 'transparent' }}
                    onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
                    onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
                  >
                    Log in
                  </button>
                  <button
                    type="button"
                    onClick={() => { setConversionTrigger('voluntary'); setShowConversionModal(true) }}
                    className="text-sm font-semibold px-5 py-1.5 rounded-full transition-opacity"
                    style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', color: 'white' }}
                    onMouseEnter={e => e.currentTarget.style.opacity = '0.88'}
                    onMouseLeave={e => e.currentTarget.style.opacity = '1'}
                  >
                    Sign up for free
                  </button>
                </>
              )}

              {messages.length > 0 && (
              <div className="relative" ref={chatMenuRef}>
                <button
                  type="button"
                  onClick={() => setChatMenuOpen(v => !v)}
                  aria-label="Chat options"
                  title="Options"
                  className="w-8 h-8 flex items-center justify-center rounded-lg transition-all"
                  style={{ color: chatMenuOpen ? 'var(--t-tx2)' : 'var(--t-tx5)', background: chatMenuOpen ? 'var(--t-hov)' : 'var(--t-card)', border: '1px solid var(--t-bd2)' }}
                  onMouseEnter={e => { e.currentTarget.style.color = 'var(--t-tx2)'; e.currentTarget.style.background = 'var(--t-hov)' }}
                  onMouseLeave={e => { if (!chatMenuOpen) { e.currentTarget.style.color = 'var(--t-tx5)'; e.currentTarget.style.background = 'var(--t-card)' } }}
                >
                  <MoreVertical size={15} />
                </button>

                {chatMenuOpen && (
                  <div className="absolute right-0 top-full mt-1.5 w-52 rounded-xl overflow-hidden z-50 shadow-xl"
                    style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd3)' }}>

                    {/* Files in this chat */}
                    {(() => {
                      const files = [...new Map(
                        messages.flatMap(m => Array.isArray(m.sources) ? m.sources : [])
                          .filter(s => {
                            const name = typeof s === 'string' ? s : (s.source || s.filename || s.file || '')
                            return name && !(typeof s === 'object' && s.modality === 'web')
                          })
                          .map(s => {
                            const name = typeof s === 'string' ? s : (s.source || s.filename || s.file || '')
                            return [name, name]
                          })
                      ).values()]
                      return (
                        <>
                          <div style={{ height: '1px', background: 'var(--t-bd2)', margin: '2px 0' }} />
                          <div className="px-4 py-2">
                            <p className="text-[11px] font-semibold uppercase tracking-wide mb-1.5"
                              style={{ color: 'var(--t-tx5)' }}>
                              Files in this chat
                            </p>
                            {files.length === 0 ? (
                              <p className="text-[12px]" style={{ color: 'var(--t-tx6)' }}>No files referenced yet</p>
                            ) : (
                              <ul className="flex flex-col gap-1">
                                {files.map(name => (
                                  <li key={name} className="flex items-center gap-2 text-[12px] truncate"
                                    style={{ color: 'var(--t-tx3)' }}>
                                    {fileModalityIcon(name)}
                                    <span className="truncate">{name}</span>
                                  </li>
                                ))}
                              </ul>
                            )}
                          </div>
                        </>
                      )
                    })()}
                  </div>
                )}
              </div>
            )}
            </div>
          </div>
        )}

        {/* Message area */}
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center px-6 text-center select-none">
            <div className="mb-5" style={{ color: 'var(--t-accent)' }}>
              <Sparkles size={40} strokeWidth={1.3} />
            </div>
            <h2 className="text-2xl sm:text-4xl font-bold mb-3 leading-tight" style={{ color: 'var(--t-tx1)' }}>
              {kbFiles.length > 0
                ? `${kbFiles.length} file${kbFiles.length !== 1 ? 's' : ''} loaded. Ask anything.`
                : 'Turn your files into answers.'}
            </h2>
            <p className="text-[15px] sm:text-[17px] mb-7 max-w-md leading-relaxed" style={{ color: 'var(--t-tx5)' }}>
              {kbFiles.length > 0
                ? 'Your knowledge base is ready.'
                : 'Drop files into the sidebar to build your knowledge base, then ask questions.'}
            </p>

            {kbFiles.length === 0 && (
              <div className="grid grid-cols-4 sm:grid-cols-7 gap-2 mb-8 max-w-lg w-full">
                {[
                  { label: 'Text',  color: '#94a3b8', icon: (
                    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                      <rect width="28" height="28" rx="5" fill="#334155"/>
                      <rect x="7" y="8"  width="14" height="2" rx="1" fill="#94a3b8"/>
                      <rect x="7" y="13" width="14" height="2" rx="1" fill="#94a3b8"/>
                      <rect x="7" y="18" width="9"  height="2" rx="1" fill="#94a3b8"/>
                    </svg>
                  )},
                  { label: 'PDF',   color: '#f87171', icon: (
                    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                      <rect width="28" height="28" rx="5" fill="#dc2626"/>
                      <path d="M7 5h10l5 5v13a1 1 0 01-1 1H7a1 1 0 01-1-1V6a1 1 0 011-1z" fill="rgba(255,255,255,0.15)"/>
                      <path d="M17 5l5 5h-4a1 1 0 01-1-1V5z" fill="rgba(255,255,255,0.3)"/>
                      <text x="14" y="21.5" textAnchor="middle" fill="white" fontSize="6" fontWeight="800" fontFamily="Arial,sans-serif" letterSpacing="0.3">PDF</text>
                    </svg>
                  )},
                  { label: 'Word',  color: '#93c5fd', icon: (
                    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                      <rect width="28" height="28" rx="5" fill="#2b579a"/>
                      <text x="14" y="20" textAnchor="middle" fill="white" fontSize="15" fontWeight="900" fontFamily="Arial,sans-serif">W</text>
                    </svg>
                  )},
                  { label: 'Excel', color: '#4ade80', icon: (
                    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                      <rect width="28" height="28" rx="5" fill="#217346"/>
                      <rect x="6" y="6"  width="7" height="7" rx="1" fill="rgba(255,255,255,0.25)"/>
                      <rect x="15" y="6" width="7" height="7" rx="1" fill="rgba(255,255,255,0.15)"/>
                      <rect x="6" y="15" width="7" height="7" rx="1" fill="rgba(255,255,255,0.15)"/>
                      <rect x="15" y="15" width="7" height="7" rx="1" fill="rgba(255,255,255,0.25)"/>
                    </svg>
                  )},
                  { label: 'Image', color: '#34d399', icon: (
                    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                      <rect width="28" height="28" rx="5" fill="#0f766e"/>
                      <rect x="5" y="8" width="18" height="13" rx="1.5" fill="rgba(255,255,255,0.2)"/>
                      <circle cx="10.5" cy="13" r="2" fill="rgba(255,255,255,0.55)"/>
                      <path d="M5 19l6-6 3.5 3.5 2.5-2.5 6 5.5H5z" fill="rgba(255,255,255,0.55)"/>
                    </svg>
                  )},
                  { label: 'Audio', color: '#a78bfa', icon: (
                    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                      <rect width="28" height="28" rx="5" fill="#5b21b6"/>
                      <rect x="4.5"  y="12.5" width="2.5" height="3.5" rx="1.25" fill="rgba(255,255,255,0.6)"/>
                      <rect x="8.5"  y="10"   width="2.5" height="8.5" rx="1.25" fill="rgba(255,255,255,0.6)"/>
                      <rect x="12.5" y="7.5"  width="2.5" height="13" rx="1.25" fill="rgba(255,255,255,0.6)"/>
                      <rect x="16.5" y="10"   width="2.5" height="8.5" rx="1.25" fill="rgba(255,255,255,0.6)"/>
                      <rect x="20.5" y="12.5" width="2.5" height="3.5" rx="1.25" fill="rgba(255,255,255,0.6)"/>
                    </svg>
                  )},
                  { label: 'Video', color: '#60a5fa', icon: (
                    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
                      <rect width="28" height="28" rx="5" fill="#1d4ed8"/>
                      <rect x="4" y="9" width="14" height="10" rx="1.5" fill="rgba(255,255,255,0.2)"/>
                      <path d="M18 11.5l6-2.5v10l-6-2.5v-5z" fill="rgba(255,255,255,0.3)"/>
                      <path d="M10 12v5l4.5-2.5L10 12z" fill="rgba(255,255,255,0.75)"/>
                    </svg>
                  )},
                ].map(({ label, color, icon }) => (
                  <div
                    key={label}
                    className="flex flex-col items-center gap-1.5 rounded-xl px-2 py-3"
                    style={{ background: 'var(--t-card)', border: `1px solid ${color}30` }}
                  >
                    <span className="leading-none">{icon}</span>
                    <span className="text-[10px] font-semibold tracking-wide uppercase" style={{ color }}>{label}</span>
                  </div>
                ))}
              </div>
            )}

          </div>
        ) : (
          <div ref={scrollAreaRef} onScroll={handleScroll} className="flex-1 overflow-y-auto px-6 py-6">
            <div className="max-w-3xl mx-auto space-y-5">
              {messages.map((msg, i) => {
                if (msg.pending && msg.content === '') return <TypingIndicator key={msg.id || i} />
                const isLastAssistant = i === messages.length - 1 && msg.role === 'assistant' && !streaming
                const precedingQuery = msg.role === 'assistant' && i > 0 && messages[i - 1]?.role === 'user'
                  ? messages[i - 1].content
                  : null
                return (
                  <MessageBubble
                    key={msg.id || i}
                    message={msg}
                    isStreaming={msg.id === streamingId && streaming}
                    dark={dark}
                    onRegenerate={isLastAssistant ? handleRegenerate : null}
                    onEdit={msg.role === 'user' && !streaming ? (text) => handleEditMessage(i, text) : null}
                    showSources={showSources}
                    authToken={auth.token}
                    sessionId={sessionId}
                    precedingQuery={precedingQuery}
                  />
                )
              })}

              <div ref={messagesEndRef} />
            </div>
          </div>
        )}

        {/* Scroll to bottom */}
        {showScrollBtn && (
          <div className="absolute bottom-36 left-1/2 -translate-x-1/2 z-10">
            <button
              onClick={scrollToBottom}
              aria-label="Scroll to bottom"
              title="Scroll to bottom"
              className="w-9 h-9 rounded-full flex items-center justify-center shadow-lg transition-all"
              style={{ background: 'var(--t-hov2)', border: '1px solid var(--t-bd4)', color: 'var(--t-tx4)' }}
            >
              <ChevronDown size={16} />
            </button>
          </div>
        )}

        {/* Input bar */}
        <div className="flex-shrink-0 px-6 pb-5 pt-2">
          <div className="max-w-3xl mx-auto relative" ref={filePickerRef}>

            {/* @ file picker dropdown — opens upward */}
            {showFilePicker && (
              <div
                className="absolute bottom-full mb-2 left-0 right-0 z-50 rounded-xl shadow-xl overflow-hidden"
                style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd3)' }}
              >
                <div className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider"
                  style={{ color: 'var(--t-tx5)', borderBottom: '1px solid var(--t-bd2)' }}>
                  Scope to a file
                </div>
                {kbFiles.length === 0 ? (
                  <div className="px-4 py-4 text-center text-[13px]" style={{ color: 'var(--t-tx5)' }}>
                    No files uploaded yet — use the <strong style={{ color: 'var(--t-tx3)' }}>+</strong> button to add files
                  </div>
                ) : (
                  <div className="overflow-y-auto" style={{ maxHeight: 220 }}>
                    {[...kbFiles]
                      .sort((a, b) => a.filename.localeCompare(b.filename))
                      .map(f => {
                        const ext    = f.filename.split('.').pop()?.toUpperCase() || 'FILE'
                        const active = selectedFile?.filename === f.filename
                        return (
                          <button
                            key={f.filename}
                            onClick={() => { setSelectedFile(active ? null : f); setShowFilePicker(false) }}
                            className="w-full flex items-center gap-2.5 px-3 py-2 text-left transition-colors"
                            style={{ background: active ? 'rgba(139,92,246,0.12)' : 'transparent' }}
                            onMouseEnter={e => { if (!active) e.currentTarget.style.background = 'var(--t-hov)' }}
                            onMouseLeave={e => { if (!active) e.currentTarget.style.background = 'transparent' }}
                          >
                            <span className="flex-shrink-0 text-[9px] font-bold px-1.5 py-0.5 rounded"
                              style={{ background: 'var(--t-hov3)', color: 'var(--t-tx4)', letterSpacing: '0.05em' }}>
                              {ext}
                            </span>
                            <span className="text-[13px] truncate" style={{ color: active ? 'var(--t-accent)' : 'var(--t-tx2)' }}>
                              {f.filename}
                            </span>
                            {active && (
                              <span className="ml-auto text-[11px] flex-shrink-0" style={{ color: 'var(--t-accent)' }}>✓</span>
                            )}
                          </button>
                        )
                      })
                    }
                  </div>
                )}
              </div>
            )}

            {/* Two-section composer card */}
            <div
              className={`rounded-2xl overflow-hidden transition-all duration-200 ${inputFocused ? 'glow-input' : ''}`}
              style={{
                background: 'var(--t-card)',
                border: `1px solid ${streaming ? 'var(--t-bd3)' : 'var(--t-bd2)'}`,
              }}
            >
              {/* TOP — text input */}
              <div className="px-4 pt-3 pb-1">
                <textarea
                  ref={inputRef}
                  rows={1}
                  placeholder={PLACEHOLDERS[placeholderIdx]}
                  value={input}
                  onChange={e => {
                    setInput(e.target.value)
                    e.target.style.height = 'auto'
                    e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`
                  }}
                  onFocus={() => setInputFocused(true)}
                  onBlur={() => setInputFocused(false)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
                    if (e.key === 'Escape') setShowFilePicker(false)
                  }}
                  disabled={streaming}
                  className="w-full bg-transparent text-[15px] outline-none disabled:opacity-60 resize-none leading-6"
                  style={{ color: 'var(--t-tx1)', overflowY: 'auto', maxHeight: 120 }}
                />
              </div>

              {/* BOTTOM — controls row */}
              <div className="flex items-center gap-2 px-3 pb-2.5 pt-0">

                {/* @ file scope button — always visible and clickable */}
                <button
                  type="button"
                  onClick={() => setShowFilePicker(p => !p)}
                  title="Scope to a file"
                  aria-label="Scope query to a specific file"
                  className="flex-shrink-0 w-7 h-7 flex items-center justify-center rounded-lg text-[13px] font-semibold transition-all"
                  style={selectedFile || showFilePicker
                    ? { background: 'rgba(139,92,246,0.18)', color: 'var(--t-accent)' }
                    : { background: 'transparent', color: 'var(--t-tx4)' }
                  }
                  onMouseEnter={e => { if (!selectedFile && !showFilePicker) { e.currentTarget.style.background = 'var(--t-hov)'; e.currentTarget.style.color = 'var(--t-tx2)' } }}
                  onMouseLeave={e => { if (!selectedFile && !showFilePicker) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--t-tx4)' } }}
                >
                  @
                </button>

                {/* Selected-file chip */}
                {selectedFile && (
                  <span
                    className="flex-shrink-0 inline-flex items-center gap-1 text-[11px] font-medium pl-2 pr-1 py-0.5 rounded-md"
                    style={{ background: 'rgba(139,92,246,0.15)', color: 'var(--t-accent)', border: '1px solid rgba(139,92,246,0.3)', maxWidth: 160 }}
                  >
                    <span className="truncate" style={{ maxWidth: 110 }}>{selectedFile.filename}</span>
                    <button
                      type="button"
                      onClick={() => setSelectedFile(null)}
                      aria-label="Clear file scope"
                      className="flex-shrink-0 flex items-center justify-center w-3.5 h-3.5 rounded transition-opacity hover:opacity-60"
                      style={{ color: 'var(--t-accent)' }}
                    >
                      ×
                    </button>
                  </span>
                )}

                {/* Spacer */}
                <div className="flex-1" />


                {/* Send / Stop */}
                <button
                  type="button"
                  onClick={() => streaming ? handleStop() : handleSend()}
                  disabled={!streaming && !input.trim()}
                  title={streaming ? 'Stop generating' : 'Send'}
                  aria-label={streaming ? 'Stop generating' : 'Send message'}
                  className="w-8 h-8 flex items-center justify-center rounded-full flex-shrink-0 transition-all"
                  style={
                    streaming
                      ? { background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', color: '#fff' }
                      : input.trim()
                        ? { background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', color: '#fff' }
                        : { background: 'var(--t-hov3)', color: 'var(--t-tx5)' }
                  }
                >
                  {streaming ? <Square size={12} fill="currentColor" /> : <Send size={14} />}
                </button>
              </div>
            </div>

            <div className="mt-2 px-4 py-1.5 text-center text-[10.5px]"
              style={{ color: 'var(--t-tx6)' }}>
              AI-generated · Verify before use
            </div>
          </div>
        </div>
      </div>

      {/* Guest conversion modal — shown when limit hit or user clicks Sign up */}
      {showConversionModal && auth?.isGuest && (
        <ConversionModal
          guestToken={auth.token}
          trigger={conversionTrigger}
          onConvert={(data) => {
            setShowConversionModal(false)
            if (onGuestConvert) onGuestConvert(data)
          }}
          onGoogleConvert={() => {
            setShowConversionModal(false)
            if (onGuestGoogleConvert) onGuestGoogleConvert()
          }}
          onClose={conversionTrigger === 'voluntary' ? () => setShowConversionModal(false) : undefined}
        />
      )}

    </div>
  )
}
