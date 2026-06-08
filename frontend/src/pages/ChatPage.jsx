import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Square, Sparkles, ChevronDown, Menu } from 'lucide-react'
import Sidebar from '../components/Sidebar'
import SettingsModal from '../components/SettingsModal'
import MessageBubble from '../components/MessageBubble'
import TypingIndicator from '../components/TypingIndicator'
import { streamQuery, queryMeta, getChatSession } from '../api/client'
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

const PLACEHOLDERS = [
  'Ask anything about your files…',
  'Summarise a document…',
  'Find key insights…',
  'Compare sections across files…',
]

function buildSuggestions(kbFiles) {
  if (!kbFiles.length) return []
  return kbFiles.slice(0, 3).map((f, i) => {
    const name = f.filename.replace(/\.[^.]+$/, '')
    if (i === 0) return `Summarise ${name}`
    if (i === 1) return `What does the ${name} show?`
    return `Key trends in ${name}?`
  })
}

export default function ChatPage({ auth, onLogout, dark, onToggleTheme, onStreamingChange }) {
  const [messages, setMessages]           = useState([])
  const [input, setInput]                 = useState('')
  const [kbFiles, setKbFiles]             = useState([])
  const [streaming, setStreaming]         = useState(false)
  const [streamingId, setStreamingId]     = useState(null)
  const [autoScroll, setAutoScroll]       = useState(true)
  const [showScrollBtn, setShowScrollBtn] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const isMobile = useIsMobile()
  const [sessionId, setSessionId]         = useState(() => crypto.randomUUID())
  const [loadingSession, setLoadingSession] = useState(false)
  const [inputFocused, setInputFocused]   = useState(false)
  const [placeholderIdx, setPlaceholderIdx] = useState(0)
  const [settingsOpen, setSettingsOpen]   = useState(false)
  const [showSources, setShowSources]     = useState(() => localStorage.getItem('magik_show_sources') !== 'false')

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

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      const mod = e.metaKey || e.ctrlKey
      if (mod && e.key === 'k') { e.preventDefault(); inputRef.current?.focus() }
      if (mod && e.shiftKey && e.key === 'N') { e.preventDefault(); handleNewChat() }
      if (e.key === 'Escape') setMobileSidebarOpen(false)
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

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

  // Strip LLM format artifacts from the streamed text so users never see
  // raw numeric citation indices, format fields, or summary epilogues.
  const cleanStreamText = (raw) => raw
    .replace(/\(\s*[\d.]+\s+is\s+(?:the\s+)?most\s+confident\s*\)/gi, '')  // "(1.0 is most confident)" confidence-field leak
    .replace(/\(\s*confidence\s*[:=]?\s*[\d.]+\s*\)/gi, '')                // "(confidence: 0.9)"
    .replace(/^\s*(?:Timeline|Overview|Summary|Context)\s*:\s*\n?/i, '')   // "Timeline:" header label
    .replace(/^\s*Source\s*:\s*<source[^>]*>\s*\n?/i, '')                  // "Source: <source number>"
    .replace(/\s*Source\s*:\s*$/im, '')                                     // Trailing bare "Source:" label
    .replace(/^\s*\[?\s*Answer\s*\]?\s*:\s*/i, '')                         // Leading "Answer:" or "[Answer:"
    .replace(/\n\s*\[?\s*Answer\s*\]?\s*:\s*/gi, '\n')                     // Mid-text "Answer:" on its own line
    .replace(/\s*\[\d+(?:,\s*\d+)*\]/g, '')                               // [1,3] [1,2,4]
    .replace(/\s+\(\d+\)(?=[\s.,;!?]|$)/g, '')                           // "(1)" footnote markers after words
    .replace(/\s*\[[^\]]{3,120}\.(txt|pdf|docx?|xlsx?|csv|mp3|mp4|wav|png|jpg|jpeg)\s*[^\]]{0,60}\]/gi, '')  // inline [filename.ext] cite tags
    .replace(/\s*\[[^\]\n]*<[A-Z_]{2,20}>[^\]\n]*\]/g, '')                // mangled cite tag, e.g. [aapl_def14a_<URL>cx] (scrubber ate the .docx)
    .replace(/\n?(Answer Tags|Confidence|Sources Used|Reasoning)\s*:\s*[^\n]*/gi, '')
    .replace(/\n?\s*Sources?\s*:\s*$/im, '')                              // trailing "Sources:" or "Source:" the LLM appended
    .replace(/\n?Therefore,?\s+the\s+answer\s+is\s*:?\s*<text>[\s\S]*?<\/text>/gi, '')
    .replace(/\n?Therefore,?\s+the\s+answer\s+is\s*:?[^\n]*/gi, '')
    .replace(/<(?:PERSON|LOCATION|ORG|NRP|GPE|DATE_TIME|AGE|ID|URL|IP_ADDRESS|US_SSN|CREDIT_CARD|PHONE_NUMBER|EMAIL_ADDRESS)>/gi, '')
    .replace(/\s*\[No source[^\]]{0,120}\]/gi, '')                         // "[No source for X]" cite-miss artifacts
    .replace(/\s*\(\s*Item\s+\d+[^()]{0,80}\)/g, '')                      // "(Item 1. Business. Overview)"
    .replace(/\s*\(\s*[A-Z][a-zA-Z ]{1,50}\)\s*(?=[.,;!?]|$)/g, '')      // "(Overview)" "(Human capital)" as sentence-end section refs
    .replace(/ +\.(?=[\s,;!?]|$)/g, '.')      // "legal risks ." → "legal risks."
    .replace(/([.!?])\s*[,;]\s*$/g, '$1')   // "Management. ," → "Management."
    .replace(/[,;]\s*$/, '')                  // bare trailing comma/semicolon
    .trim()

  const handleSend = useCallback(async (overrideText, { skipUserMessage = false } = {}) => {
    const text = (overrideText ?? input).trim()
    if (!text || streaming) return
    setInput('')
    setAutoScroll(true)

    const botId = Date.now()
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

    const controller = new AbortController()
    abortRef.current = controller
    let fullText = ''
    let streamedSources = null
    let refused = false

    // Fire queryMeta in parallel with streaming so sources + fallback answer
    // are ready the moment streaming finishes — no sequential wait.
    const metaPromise = queryMeta(auth.token, text, sessionId)

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
      const res = await streamQuery(auth.token, text, sessionId, controller.signal)
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
            // Sources event emitted by the stream after text is complete
            if (parsed && typeof parsed === 'object' && parsed.__type__ === 'sources') {
              streamedSources = parsed.data || []
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, sources: streamedSources } : m
              ))
              continue
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
        // Stream the accurate meta-path answer letter-by-letter. Sources are set
        // first so they render the moment `streaming` flips off in `finally`.
        const meta = await metaPromise
        const metaAnswer = cleanStreamText(meta?.answer || '')
        const finalText = metaAnswer ||
          'I could not find relevant information in your knowledge base to answer this question.'
        const metaSources = (meta?.sources?.length > 0) ? meta.sources : (streamedSources || [])
        setMessages(prev => prev.map(m =>
          m.id === botId ? { ...m, sources: metaSources } : m
        ))
        await streamTextIntoBubble(finalText)
      } else {
        // The streamed answer is the accurate one — finalize immediately so the
        // citation chips render the instant the answer completes, without
        // waiting on the parallel meta pipeline.
        setMessages(prev => prev.map(m =>
          m.id === botId ? {
            ...m,
            content: streamedAnswer,
            sources: streamedSources || [],
            pending: false, streaming: false,
          } : m
        ))

        // Background (non-blocking): backfill richer sources (e.g. web queries
        // whose stream carried none) without touching the answer text.
        metaPromise.then(meta => {
          const richerSources = (meta?.sources?.length > 0) ? meta.sources : null
          if (richerSources && (!streamedSources || streamedSources.length === 0)) {
            setMessages(prev => prev.map(m =>
              m.id === botId ? { ...m, sources: richerSources } : m
            ))
          }
        }).catch(() => {})
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
  }, [input, streaming, auth.token, sessionId])

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
    handleSend(lastUser.content, { skipUserMessage: true })
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
        return
      }
      const loaded = (session.messages || []).map((m, i) => ({
        role:    m.role,
        content: m.content,
        ts:      m.timestamp ? new Date(m.timestamp).getTime() : Date.now(),
        id:      `${targetId}-${i}`,
      }))
      setMessages(loaded)
      setSessionId(targetId)
      setInput('')
      setStreamingId(null)
      setAutoScroll(true)
      setShowScrollBtn(false)
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
        />
      </div>

      <SettingsModal
        isOpen={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        auth={auth}
        onLogout={onLogout}
        dark={dark}
        onToggleTheme={onToggleTheme}
        kbFiles={kbFiles}
        setKbFiles={setKbFiles}
        sessionId={sessionId}
        showSources={showSources}
        setShowSources={setShowSources}
      />

      {/* Main column */}
      <div className="flex-1 flex flex-col min-w-0 relative" style={{ background: 'var(--t-sur)' }}>

        {/* Mobile top bar — hamburger to open sidebar */}
        {isMobile && (
          <div className="flex items-center px-3 pt-3 pb-1 flex-shrink-0">
            <button
              onClick={() => setMobileSidebarOpen(true)}
              className="w-9 h-9 flex items-center justify-center rounded-xl"
              style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx4)' }}
              aria-label="Open sidebar"
            >
              <Menu size={18} />
            </button>
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
                : 'Upload files to get started.'}
            </h2>
            <p className="text-[15px] sm:text-[17px] mb-7 sm:mb-10 max-w-md leading-relaxed" style={{ color: 'var(--t-tx5)' }}>
              {kbFiles.length > 0
                ? 'Your knowledge base is ready. Try one of these to get started:'
                : 'Drop files into the sidebar to build your knowledge base, then ask questions.'}
            </p>
            {suggestions.length > 0 && (
              <div className="flex flex-wrap gap-3 justify-center max-w-xl">
                {suggestions.map((s, i) => (
                  <button
                    key={i}
                    onClick={() => handleSend(s)}
                    className="flex items-center gap-2 rounded-full px-5 py-2.5 text-[15px] transition-all"
                    style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx4)' }}
                    onMouseEnter={e => { e.currentTarget.style.background='var(--t-inp)'; e.currentTarget.style.borderColor='var(--t-bd4)'; e.currentTarget.style.color='var(--t-tx1)' }}
                    onMouseLeave={e => { e.currentTarget.style.background='var(--t-card)'; e.currentTarget.style.borderColor='var(--t-bd2)'; e.currentTarget.style.color='var(--t-tx4)' }}
                  >
                    {CHAT_ICON}{s}
                  </button>
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
                return (
                  <MessageBubble
                    key={msg.id || i}
                    message={msg}
                    isStreaming={msg.id === streamingId && streaming}
                    dark={dark}
                    onRegenerate={isLastAssistant ? handleRegenerate : null}
                    onEdit={msg.role === 'user' && !streaming ? (text) => handleEditMessage(i, text) : null}
                    showSources={showSources}
                  />
                )
              })}

              <div ref={messagesEndRef} />
            </div>
          </div>
        )}

        {/* Scroll to bottom */}
        {showScrollBtn && (
          <div className="absolute bottom-24 right-8 z-10">
            <button
              onClick={scrollToBottom}
              className="w-9 h-9 rounded-full flex items-center justify-center shadow-lg transition-all"
              style={{ background: 'var(--t-hov2)', border: '1px solid var(--t-bd4)', color: 'var(--t-tx4)' }}
            >
              <ChevronDown size={16} />
            </button>
          </div>
        )}

        {/* Input bar */}
        <div className="flex-shrink-0 px-6 pb-5 pt-2">
          <div className="max-w-3xl mx-auto">
            <div
              className={`flex items-center gap-3 rounded-2xl px-4 py-3 transition-all duration-200 ${inputFocused ? 'glow-input' : ''}`}
              style={{
                background: 'var(--t-card)',
                border: `1px solid ${streaming ? 'var(--t-bd3)' : 'var(--t-bd2)'}`,
              }}
            >
              <input
                ref={inputRef}
                type="text"
                placeholder={PLACEHOLDERS[placeholderIdx]}
                value={input}
                onChange={e => setInput(e.target.value)}
                onFocus={() => setInputFocused(true)}
                onBlur={() => setInputFocused(false)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend() }
                }}
                disabled={streaming}
                className="flex-1 bg-transparent text-[16px] outline-none min-w-0 disabled:opacity-60"
                style={{ color: 'var(--t-tx1)' }}
              />

              <button
                type="button"
                onClick={() => streaming ? handleStop() : handleSend()}
                disabled={!streaming && !input.trim()}
                title={streaming ? 'Stop generating' : 'Send'}
                className="w-8 h-8 flex items-center justify-center rounded-full flex-shrink-0 transition-all"
                style={
                  streaming
                    ? { background: 'var(--t-tx1)', color: 'var(--t-bg)' }
                    : input.trim()
                      ? { background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', color: '#fff' }
                      : { background: 'var(--t-hov3)', color: 'var(--t-tx5)' }
                }
              >
                {streaming ? <Square size={12} fill="currentColor" /> : <Send size={14} />}
              </button>
            </div>

            <div className="mt-3 rounded-t-2xl px-6 py-2.5 text-center text-[11.5px] leading-relaxed"
              style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)', borderBottom: 'none', color: 'var(--t-tx5)' }}>
              Responses are generated from your knowledge base and may not be fully accurate. Verify critical information before use.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
