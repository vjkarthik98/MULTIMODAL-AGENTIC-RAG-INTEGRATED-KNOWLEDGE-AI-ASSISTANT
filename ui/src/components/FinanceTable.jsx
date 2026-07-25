import { useState, useMemo } from 'react'
import { Copy, Check } from 'lucide-react'

/* ── Parse a GitHub-Flavored Markdown table into header + rows ── */
function parseMarkdownTable(md) {
  if (!md) return null
  const lines = md.trim().split('\n').filter(l => l.includes('|'))
  if (lines.length < 2) return null

  const parseCells = (line) =>
    line.split('|').map(c => c.trim()).filter((_, i, a) => i > 0 && i < a.length - 1)

  const headers = parseCells(lines[0])
  // Skip the separator row (--- row)
  const rows = lines.slice(2).map(parseCells)
  return { headers, rows }
}

/* ── Numeric cell detection: starts with optional $€£, then digits ── */
const _NUM_RE = /^[−\-$€£]?[\d,]+(\.\d+)?([KMBT%bps]*)$/i
const isNumeric = (cell) => _NUM_RE.test(cell.replace(/\s/g, ''))
const isNegative = (cell) => {
  const c = cell.trim()
  return c.startsWith('-') || c.startsWith('−') || c.startsWith('(') || parseFloat(c.replace(/[^0-9.\-]/g, '')) < 0
}

/* ── Convert table to CSV ── */
function toCSV({ headers, rows }) {
  const esc = (v) => `"${v.replace(/"/g, '""')}"`
  return [headers, ...rows].map(row => row.map(esc).join(',')).join('\n')
}

export default function FinanceTable({ markdown, title, rowRange }) {
  const [copied, setCopied] = useState(false)

  const parsed = useMemo(() => parseMarkdownTable(markdown), [markdown])

  if (!parsed) return null

  const { headers, rows } = parsed

  const handleCopy = async () => {
    await navigator.clipboard.writeText(toCSV(parsed))
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ border: '1px solid var(--t-bd2)', background: 'var(--t-card)' }}
    >
      {/* Header bar */}
      <div
        className="flex items-center justify-between px-3 py-2"
        style={{ borderBottom: '1px solid var(--t-bd1)', background: 'var(--t-inp)' }}
      >
        <span className="text-[12px] font-medium" style={{ color: 'var(--t-tx3)' }}>
          📊 {title || 'Financial Table'}{rowRange ? ` · rows ${rowRange}` : ''}
        </span>
        <button
          onClick={handleCopy}
          title="Copy as CSV"
          className="flex items-center gap-1 text-[11px] rounded-lg px-2 py-1 transition-colors"
          style={{
            color: copied ? 'var(--t-success)' : 'var(--t-tx4)',
            background: 'transparent',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
        >
          {copied ? <Check size={11} /> : <Copy size={11} />}
          {copied ? 'Copied' : 'CSV'}
        </button>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-[12px] border-collapse">
          <thead>
            <tr style={{ background: 'var(--t-inp)' }}>
              {headers.map((h, i) => (
                <th
                  key={i}
                  className="px-3 py-2 font-semibold"
                  style={{
                    textAlign: i === 0 ? 'left' : 'right',
                    color: 'var(--t-tx3)',
                    borderBottom: '1px solid var(--t-bd2)',
                    whiteSpace: 'nowrap',
                  }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, ri) => (
              <tr
                key={ri}
                style={{ borderBottom: '1px solid var(--t-bd1)' }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov1)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                {row.map((cell, ci) => {
                  const numeric = ci > 0 && isNumeric(cell)
                  const negative = numeric && isNegative(cell)
                  return (
                    <td
                      key={ci}
                      className="px-3 py-1.5 font-mono"
                      style={{
                        textAlign: ci === 0 ? 'left' : 'right',
                        color: numeric
                          ? negative ? '#f87171' : '#4ade80'
                          : 'var(--t-tx2)',
                        fontFamily: ci === 0 ? 'inherit' : 'ui-monospace, Consolas, monospace',
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {cell || '—'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
