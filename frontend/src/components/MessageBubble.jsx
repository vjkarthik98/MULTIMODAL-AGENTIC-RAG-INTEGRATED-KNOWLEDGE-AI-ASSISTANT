import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import python     from 'react-syntax-highlighter/dist/esm/languages/prism/python'
import javascript from 'react-syntax-highlighter/dist/esm/languages/prism/javascript'
import typescript from 'react-syntax-highlighter/dist/esm/languages/prism/typescript'
import bash       from 'react-syntax-highlighter/dist/esm/languages/prism/bash'
import json       from 'react-syntax-highlighter/dist/esm/languages/prism/json'
import sql        from 'react-syntax-highlighter/dist/esm/languages/prism/sql'
import markdown   from 'react-syntax-highlighter/dist/esm/languages/prism/markdown'
import yaml       from 'react-syntax-highlighter/dist/esm/languages/prism/yaml'

SyntaxHighlighter.registerLanguage('python',     python)
SyntaxHighlighter.registerLanguage('javascript', javascript)
SyntaxHighlighter.registerLanguage('js',         javascript)
SyntaxHighlighter.registerLanguage('typescript', typescript)
SyntaxHighlighter.registerLanguage('ts',         typescript)
SyntaxHighlighter.registerLanguage('bash',       bash)
SyntaxHighlighter.registerLanguage('sh',         bash)
SyntaxHighlighter.registerLanguage('json',       json)
SyntaxHighlighter.registerLanguage('sql',        sql)
SyntaxHighlighter.registerLanguage('markdown',   markdown)
SyntaxHighlighter.registerLanguage('yaml',       yaml)
SyntaxHighlighter.registerLanguage('yml',        yaml)
import { FileText, Copy, ThumbsUp, ThumbsDown, Check, RotateCcw, Pencil } from 'lucide-react'
import { useToast } from '../context/ToastContext'

/* ── Source chip ── */
function SourceChip({ source }) {
  const raw   = typeof source === 'string' ? source : source.filename || source.source || source.file || String(source)
  const page  = typeof source === 'object' && source.page ? ` · p.${source.page}` : ''
  const label = raw.includes('/') ? raw.split('/').pop() : raw
  return (
    <span className="inline-flex items-center gap-1.5 text-[11px] rounded-full px-3 py-1 mr-1.5 mb-1.5 transition-colors cursor-default select-none"
      style={{ background: 'var(--t-chp)', border: '1px solid var(--t-chpb)', color: 'var(--t-tx4)' }}>
      <FileText size={10} className="flex-shrink-0" style={{ color: 'var(--t-tx5)' }} />
      <span className="truncate max-w-[180px]">{label}{page}</span>
    </span>
  )
}

/* ── Timestamp ── */
function formatTime(ts) {
  if (!ts) return ''
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

/* ── Code renderer for ReactMarkdown ── */
function CodeBlock({ dark, inline, className, children }) {
  const match = /language-(\w+)/.exec(className || '')
  const lang  = match ? match[1] : 'text'
  const code  = String(children).replace(/\n$/, '')

  if (inline) {
    return (
      <code style={{
        background: 'var(--t-inp)',
        border: '1px solid var(--t-bd4)',
        borderRadius: 4,
        padding: '1px 5px',
        fontSize: '0.85em',
        fontFamily: 'ui-monospace, Consolas, monospace',
      }}>
        {children}
      </code>
    )
  }

  return (
    <div style={{ position: 'relative', margin: '0.6em 0' }}>
      <SyntaxHighlighter
        language={lang}
        style={dark ? oneDark : oneLight}
        customStyle={{
          margin: 0,
          borderRadius: 10,
          fontSize: '0.82em',
          border: `1px solid var(--t-bd3)`,
        }}
        showLineNumbers={code.split('\n').length > 4}
        wrapLongLines={false}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  )
}

/* ── Main component ── */
export default function MessageBubble({ message, isStreaming, dark, onRegenerate, onEdit, showSources = true }) {
  const isUser = message.role === 'user'
  const { addToast } = useToast()
  const [copied, setCopied]   = useState(false)
  const [vote, setVote]       = useState(null)   // 'up' | 'down' | null
  const [isEditing, setIsEditing] = useState(false)
  const [editText, setEditText]   = useState('')

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
            {message.content}
          </div>
          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity pr-0.5">
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
            <div className={`prose-chat ${isStreaming ? 'streaming-cursor' : ''}`}>
              <ReactMarkdown
                components={{
                  code: ({ inline, className, children, ...props }) => (
                    <CodeBlock dark={dark} inline={inline} className={className} {...props}>
                      {children}
                    </CodeBlock>
                  ),
                }}
              >
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Source chips */}
        {showSources && message.sources?.length > 0 && (
          <div className="mt-2 flex flex-wrap">
            {message.sources.map((src, i) => <SourceChip key={i} source={src} />)}
          </div>
        )}

        {/* Action row — hover reveal */}
        {!isEmpty && !isStreaming && (
          <div className="flex items-center gap-1 mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
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
              onClick={() => setVote(v => v === 'up' ? null : 'up')}
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
              onClick={() => setVote(v => v === 'down' ? null : 'down')}
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
