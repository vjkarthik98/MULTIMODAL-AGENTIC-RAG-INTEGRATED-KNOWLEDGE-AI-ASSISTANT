import { useState, useEffect, useCallback } from 'react'
import { Trash2, RefreshCw, FileText, Image, Sheet, FileVideo, FileAudio, FileType, LetterText, File as FileIcon, Database } from 'lucide-react'
import { listKBFiles, deleteKBFileByHash } from '../api/client'
import { useToast } from '../context/ToastContext'

/* ── Modality icon ── */
function ModalityIcon({ modality, filename }) {
  const ext = (filename || '').split('.').pop().toUpperCase()
  const s = { size: 14, style: { flexShrink: 0 } }
  const mod = (modality || '').toLowerCase()
  if (mod === 'audio' || ['MP3','WAV','M4A','OGG','FLAC','OPUS','AIFF','WMA'].includes(ext))
    return <FileAudio  {...s} style={{ ...s.style, color: '#c084fc' }} />
  if (mod === 'video' || ['MP4','MOV','AVI','MKV','WEBM'].includes(ext))
    return <FileVideo  {...s} style={{ ...s.style, color: '#60a5fa' }} />
  if (mod === 'image' || ['PNG','JPG','JPEG','GIF','WEBP'].includes(ext))
    return <Image      {...s} style={{ ...s.style, color: '#2dd4bf' }} />
  if (['excel','xlsx'].includes(mod) || ['XLS','XLSX','CSV'].includes(ext))
    return <Sheet      {...s} style={{ ...s.style, color: '#4ade80' }} />
  if (mod === 'pdf' || ext === 'PDF')
    return <FileText   {...s} style={{ ...s.style, color: '#f87171' }} />
  if (['word','docx'].includes(mod) || ['DOC','DOCX'].includes(ext))
    return <LetterText {...s} style={{ ...s.style, color: '#60a5fa' }} />
  if (mod === 'txt' || ext === 'TXT')
    return <FileType   {...s} style={{ ...s.style, color: '#94a3b8' }} />
  return <FileIcon {...s} style={{ ...s.style, color: 'var(--t-tx5)' }} />
}

function fmtDate(iso) {
  if (!iso) return ''
  try {
    return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' })
  } catch { return iso }
}

/* ── Props:
     auth        – { token }
     onReIngest  – (fileHash: string, filename: string) => void, optional
     compact     – boolean (narrow sidebar mode)
*/
export default function KnowledgeBasePanel({ auth, onReIngest, compact = false }) {
  const [files, setFiles]       = useState([])
  const [loading, setLoading]   = useState(true)
  const [error, setError]       = useState(null)
  const [deleting, setDeleting] = useState(new Set())
  const { addToast }            = useToast()

  const refresh = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const data = await listKBFiles(auth.token)
      setFiles(data.files || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [auth.token])

  useEffect(() => { refresh() }, [refresh])

  const handleDelete = async (file) => {
    if (!window.confirm(`Delete "${file.filename}" from your knowledge base? This cannot be undone.`)) return
    setDeleting(prev => new Set([...prev, file.file_hash]))
    try {
      await deleteKBFileByHash(auth.token, file.file_hash)
      setFiles(prev => prev.filter(f => f.file_hash !== file.file_hash))
      addToast(`Deleted: ${file.filename}`, 'success')
    } catch (err) {
      addToast(`Delete failed: ${err.message}`, 'error')
    } finally {
      setDeleting(prev => { const s = new Set(prev); s.delete(file.file_hash); return s })
    }
  }

  const totalChunks = files.reduce((sum, f) => sum + (f.chunk_count || 0), 0)

  return (
    <div
      className="flex flex-col h-full"
      style={{ background: 'var(--t-card)' }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2.5 flex-shrink-0"
        style={{ borderBottom: '1px solid var(--t-bd2)', background: 'var(--t-inp)' }}
      >
        <div className="flex items-center gap-2">
          <Database size={13} style={{ color: '#8b5cf6' }} />
          <span className="text-[12px] font-semibold" style={{ color: 'var(--t-tx2)' }}>
            Knowledge Base
          </span>
        </div>
        <div className="flex items-center gap-2">
          {files.length > 0 && (
            <span className="text-[11px]" style={{ color: 'var(--t-tx5)' }}>
              {files.length} file{files.length !== 1 ? 's' : ''} · {totalChunks} chunks
            </span>
          )}
          <button
            onClick={refresh}
            disabled={loading}
            className="p-1 rounded-lg transition-colors"
            title="Refresh"
            style={{ color: 'var(--t-tx5)' }}
            onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto py-1">
        {loading && (
          <div className="text-center py-6 text-[12px]" style={{ color: 'var(--t-tx5)' }}>
            Loading…
          </div>
        )}
        {error && (
          <div className="text-center py-6 text-[12px]" style={{ color: 'var(--t-danger)' }}>
            {error}
          </div>
        )}
        {!loading && !error && files.length === 0 && (
          <div className="text-center py-6 text-[12px]" style={{ color: 'var(--t-tx5)' }}>
            No files ingested yet.
          </div>
        )}
        {!loading && files.map(file => (
          <div
            key={file.file_hash}
            className="flex items-start gap-2 px-3 py-2 group transition-colors"
            style={{ borderBottom: '1px solid var(--t-bd1)' }}
            onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov1)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <div className="flex-shrink-0 mt-0.5">
              <ModalityIcon modality={file.modality} filename={file.filename} />
            </div>

            <div className="flex-1 min-w-0">
              <div
                className="text-[12px] font-medium truncate"
                style={{ color: 'var(--t-tx2)' }}
                title={file.filename}
              >
                {file.filename}
              </div>
              <div className="text-[11px] flex items-center gap-2 mt-0.5" style={{ color: 'var(--t-tx5)' }}>
                {file.chunk_count != null && (
                  <span>{file.chunk_count} chunks</span>
                )}
                {file.modality && (
                  <span className="capitalize">{file.modality}</span>
                )}
                {file.ingested_at && (
                  <span>{fmtDate(file.ingested_at)}</span>
                )}
              </div>
            </div>

            {/* Action buttons */}
            <div className="flex items-center gap-1 flex-shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
              {onReIngest && (
                <button
                  onClick={() => onReIngest(file.file_hash, file.filename)}
                  title="Re-ingest"
                  className="p-1 rounded-md transition-colors"
                  style={{ color: 'var(--t-tx5)' }}
                  onMouseEnter={e => e.currentTarget.style.color = 'var(--t-accent)'}
                  onMouseLeave={e => e.currentTarget.style.color = 'var(--t-tx5)'}
                >
                  <RefreshCw size={11} />
                </button>
              )}
              <button
                onClick={() => handleDelete(file)}
                disabled={deleting.has(file.file_hash)}
                title="Delete from KB"
                className="p-1 rounded-md transition-colors"
                style={{ color: deleting.has(file.file_hash) ? 'var(--t-tx6)' : 'var(--t-tx5)' }}
                onMouseEnter={e => { if (!deleting.has(file.file_hash)) e.currentTarget.style.color = 'var(--t-danger)' }}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--t-tx5)' }
              >
                <Trash2 size={11} />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
