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
import { FileText, Copy, ThumbsUp, ThumbsDown, Check } from 'lucide-react'
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

/* ── Meta bar ── */
function MetaBar({ meta }) {
  if (!meta) return null
  const items = [
    meta.decision           && { label: 'Route',   value: meta.decision },
    meta.confidence != null && { label: 'Conf',    value: `${(meta.confidence * 100).toFixed(0)}%` },
    meta.latency            && { label: 'Latency', value: `${meta.latency}s` },
    meta.cache_hit          && { label: 'Cache',   value: 'hit' },
  ].filter(Boolean)
  if (!items.length) return null
  return (
    <div className="flex flex-wrap items-center gap-3 mt-2.5 px-1">
      {items.map(it => (
        <span key={it.label} className="text-[10px] font-mono" style={{ color: 'var(--t-met)' }}>
          {it.label}:<span className="ml-0.5" style={{ color: 'var(--t-metv)' }}>{it.value}</span>
        </span>
      ))}
    </div>
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
export default function MessageBubble({ message, isStreaming, dark }) {
  const isUser = message.role === 'user'
  const { addToast } = useToast()
  const [copied, setCopied]   = useState(false)
  const [vote, setVote]       = useState(null)   // 'up' | 'down' | null

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content || '')
    setCopied(true)
    addToast('Copied to clipboard', 'success')
    setTimeout(() => setCopied(false), 2000)
  }

  /* ── User bubble ── */
  if (isUser) {
    return (
      <div className="flex justify-end group">
        <div className="flex flex-col items-end gap-1 max-w-[72%]">
          <div className="rounded-2xl rounded-tr-sm px-5 py-3.5 text-[16px] leading-relaxed"
            style={{ background: 'var(--t-ubg)', border: '1px solid var(--t-ubd)', color: 'var(--t-tx1)' }}>
            {message.content}
          </div>
          {message.ts && (
            <span className="text-[10px] opacity-0 group-hover:opacity-100 transition-opacity pr-1"
              style={{ color: 'var(--t-tx5)' }}>{formatTime(message.ts)}</span>
          )}
        </div>
      </div>
    )
  }

  const isEmpty = !message.content && message.pending

  /* ── Bot bubble ── */
  return (
    <div className="flex justify-start gap-2.5 group">

      {/* Bot avatar */}
      <div className="w-7 h-7 rounded-full flex-shrink-0 mt-0.5 flex items-center justify-center shadow-sm"
        style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)' }}>
        <span className="text-white text-[10px] font-bold leading-none">✦</span>
      </div>

      <div className="flex-1 min-w-0 max-w-[84%]">
        {/* Bubble */}
        <div className="rounded-2xl rounded-tl-sm px-5 py-3.5 text-[16px] leading-relaxed"
          style={
            message.error
              ? { background: 'rgba(127,29,29,0.15)', border: '1px solid rgba(153,27,27,0.4)', color: '#fca5a5' }
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
        {message.sources?.length > 0 && (
          <div className="mt-2 flex flex-wrap">
            {message.sources.map((src, i) => <SourceChip key={i} source={src} />)}
          </div>
        )}

        <MetaBar meta={message.meta} />

        {/* Action row — hover reveal */}
        {!isEmpty && !isStreaming && (
          <div className="flex items-center gap-1 mt-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
            {/* Copy */}
            <button
              onClick={handleCopy}
              className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
              style={{ color: copied ? '#4ade80' : 'var(--t-tx5)' }}
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
              style={{ color: vote === 'down' ? '#f87171' : 'var(--t-tx5)' }}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              title="Bad response"
            >
              <ThumbsDown size={13} fill={vote === 'down' ? 'currentColor' : 'none'} />
            </button>

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
