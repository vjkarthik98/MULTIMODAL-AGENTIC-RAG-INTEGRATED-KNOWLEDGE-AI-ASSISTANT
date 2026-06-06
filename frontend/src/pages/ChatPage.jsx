import { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Square, Sparkles, ChevronDown } from 'lucide-react'
import Sidebar from '../components/Sidebar'
import SettingsModal from '../components/SettingsModal'
import MessageBubble from '../components/MessageBubble'
import TypingIndicator from '../components/TypingIndicator'
import { streamQuery, queryMeta, getChatSession } from '../api/client'
import { useToast } from '../context/ToastContext'

const CHAT_ICON = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>
)

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
            // Chunks arrive JSON-encoded so embedded newlines survive the
            // SSE "\n\n" framing; fall back to the raw string for safety.
            let token = raw
            try { token = JSON.parse(raw) } catch { /* not JSON — use raw */ }
            if (token) {
              fullText += token
              setMessages(prev => prev.map(m =>
                m.id === botId ? { ...m, content: fullText, pending: false } : m
              ))
            }
          }
          if (done) break
        }
      }

      const meta = await queryMeta(auth.token, text, sessionId)
      setMessages(prev => prev.map(m =>
        m.id === botId ? {
          ...m,
          content:  meta?.answer || fullText,
          sources:  meta?.sources || [],
          pending: false, streaming: false,
        } : m
      ))
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

      {/* Sidebar */}
      <div className={`flex-shrink-0 transition-all duration-300 ease-in-out overflow-hidden
        ${sidebarCollapsed ? 'w-[72px]' : 'w-80'}`}>
        <Sidebar
          auth={auth}
          kbFiles={kbFiles}
          setKbFiles={setKbFiles}
          onLogout={onLogout}
          onNewChat={handleNewChat}
          currentSessionId={sessionId}
          onSelectSession={handleLoadSession}
          streaming={streaming}
          collapsed={sidebarCollapsed}
          onToggleCollapse={() => setSidebarCollapsed(c => !c)}
          dark={dark}
          onToggleTheme={onToggleTheme}
          onOpenSettings={() => setSettingsOpen(true)}
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

        {/* Message area */}
        {messages.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center px-8 text-center select-none">
            <div className="mb-6" style={{ color: 'var(--t-accent)' }}>
              <Sparkles size={52} strokeWidth={1.3} />
            </div>
            <h2 className="text-4xl font-bold mb-3 leading-tight" style={{ color: 'var(--t-tx1)' }}>
              {kbFiles.length > 0
                ? `${kbFiles.length} file${kbFiles.length !== 1 ? 's' : ''} loaded. Ask anything.`
                : 'Upload files to get started.'}
            </h2>
            <p className="text-[17px] mb-10 max-w-md leading-relaxed" style={{ color: 'var(--t-tx5)' }}>
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
              MAGIK is AI and can make mistakes. Please double-check important information.
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
