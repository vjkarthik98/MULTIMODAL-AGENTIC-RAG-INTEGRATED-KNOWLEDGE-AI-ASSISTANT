import { useEffect, useRef, useState } from 'react'
import {
  Upload, X, Loader2, LogOut, FileText, Image, Sheet,
  File, AlertCircle, PanelLeftClose, Sun, Moon,
  SquarePen, FolderOpen,
} from 'lucide-react'
import { listKB, deleteKBFile, ingestFile } from '../api/client'
import { useToast } from '../context/ToastContext'

const EXT_BADGE = {
  PDF:  { label: 'PDF', bg: 'bg-red-700' },
  PNG:  { label: 'IMG', bg: 'bg-teal-700' },
  JPG:  { label: 'IMG', bg: 'bg-teal-700' },
  JPEG: { label: 'IMG', bg: 'bg-teal-700' },
  GIF:  { label: 'IMG', bg: 'bg-teal-700' },
  WEBP: { label: 'IMG', bg: 'bg-teal-700' },
  XLS:  { label: 'XLS', bg: 'bg-green-700' },
  XLSX: { label: 'XLS', bg: 'bg-green-700' },
  CSV:  { label: 'CSV', bg: 'bg-green-800' },
  MP3:  { label: 'AUD', bg: 'bg-purple-700' },
  WAV:  { label: 'AUD', bg: 'bg-purple-700' },
  MP4:  { label: 'VID', bg: 'bg-blue-700' },
  MOV:  { label: 'VID', bg: 'bg-blue-700' },
  TXT:  { label: 'TXT', bg: 'bg-gray-600' },
  DOCX: { label: 'DOC', bg: 'bg-blue-700' },
  DOC:  { label: 'DOC', bg: 'bg-blue-700' },
}

function badge(filename) {
  const ext = (filename.split('.').pop() || '').toUpperCase()
  return EXT_BADGE[ext] || { label: ext.slice(0, 3) || '?', bg: 'bg-gray-700' }
}

function FileIcon({ filename }) {
  const ext = (filename.split('.').pop() || '').toUpperCase()
  if (['PNG','JPG','JPEG','GIF','WEBP'].includes(ext))
    return <Image size={15} className="text-teal-400 flex-shrink-0" />
  if (['XLS','XLSX','CSV'].includes(ext))
    return <Sheet size={15} className="text-green-400 flex-shrink-0" />
  if (ext === 'PDF')
    return <FileText size={15} className="text-red-400 flex-shrink-0" />
  return <File size={15} className="flex-shrink-0" style={{ color: 'var(--t-tx5)' }} />
}

/* ── Icon button helper ── */
function IconBtn({ onClick, title, children, accentHover = false }) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="w-10 h-10 rounded-xl flex items-center justify-center transition-all"
      style={{ color: 'var(--t-tx4)' }}
      onMouseEnter={e => {
        e.currentTarget.style.background = 'var(--t-hov2)'
        e.currentTarget.style.color = accentHover ? 'var(--t-accent)' : 'var(--t-tx1)'
      }}
      onMouseLeave={e => {
        e.currentTarget.style.background = 'transparent'
        e.currentTarget.style.color = 'var(--t-tx4)'
      }}
    >
      {children}
    </button>
  )
}

export default function Sidebar({
  auth, kbFiles, setKbFiles, onLogout, onNewChat,
  collapsed, onToggleCollapse,
  dark, onToggleTheme,
}) {
  const [dragOver, setDragOver]           = useState(false)
  const [uploadingFiles, setUploadingFiles] = useState(new Set())
  const [uploadError, setUploadError]     = useState('')
  const [loadingKB, setLoadingKB]         = useState(true)
  const [newFiles, setNewFiles]           = useState(new Set())
  const fileInputRef                      = useRef(null)
  const prevFilenamesRef                  = useRef(new Set())
  const avatarLetter                      = auth.email?.[0]?.toUpperCase() || 'U'
  const { addToast }                      = useToast()

  const refreshKB = async () => {
    setLoadingKB(true)
    try {
      const files = await listKB(auth.token)
      const prevNames = prevFilenamesRef.current
      const added = new Set(files.map(f => f.filename).filter(n => !prevNames.has(n)))
      if (added.size) {
        setNewFiles(added)
        setTimeout(() => setNewFiles(new Set()), 600)
      }
      prevFilenamesRef.current = new Set(files.map(f => f.filename))
      setKbFiles(files)
    } catch {}
    setLoadingKB(false)
  }

  useEffect(() => { refreshKB() }, [])

  const handleUpload = async (files) => {
    setUploadError('')
    const errors = []
    for (const file of files) {
      setUploadingFiles(prev => new Set([...prev, file.name]))
      try {
        await ingestFile(auth.token, file)
        addToast(`Uploaded: ${file.name}`, 'success')
      } catch (err) {
        errors.push(`${file.name}: ${err.message}`)
        addToast(`Failed: ${file.name}`, 'error')
      }
      setUploadingFiles(prev => { const s = new Set(prev); s.delete(file.name); return s })
    }
    if (errors.length) setUploadError(errors.join('; '))
    await refreshKB()
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleDelete = async (filename) => {
    try {
      await deleteKBFile(auth.token, filename)
      setKbFiles(prev => prev.filter(f => f.filename !== filename))
      prevFilenamesRef.current.delete(filename)
    } catch {}
  }

  const handleDrop = (e) => {
    e.preventDefault(); setDragOver(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length) handleUpload(files)
  }

  const isUploading = uploadingFiles.size > 0

  const sidebarStyle = {
    background: 'var(--t-sb)',
    borderRight: '1px solid var(--t-bd1)',
  }

  /* ── COLLAPSED — icon rail ── */
  if (collapsed) {
    return (
      <div className="w-[72px] h-full flex flex-col items-center py-4 gap-1 overflow-hidden" style={sidebarStyle}>

        {/* Logo → expands sidebar */}
        <button
          onClick={onToggleCollapse}
          title="Expand sidebar"
          className="w-10 h-10 rounded-xl flex items-center justify-center mb-2 transition-all"
          onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
        >
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center flex-shrink-0">
            <span className="text-white text-xs font-bold leading-none">✦</span>
          </div>
        </button>

        {/* Divider */}
        <div className="w-8 h-px mb-1" style={{ background: 'var(--t-bd2)' }} />

        {/* New chat */}
        <IconBtn onClick={onNewChat} title="New chat">
          <SquarePen size={18} />
        </IconBtn>

        {/* Knowledge base */}
        <IconBtn title="Knowledge Base">
          <FolderOpen size={18} />
        </IconBtn>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Theme toggle */}
        <IconBtn onClick={onToggleTheme} title={dark ? 'Light mode' : 'Dark mode'} accentHover>
          {dark ? <Sun size={16} /> : <Moon size={16} />}
        </IconBtn>

        {/* Avatar / sign out */}
        <button
          onClick={onLogout}
          title={`${auth.email} · Sign out`}
          className="mt-1 w-8 h-8 rounded-full flex items-center justify-center transition-opacity hover:opacity-80"
          style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)' }}
        >
          <span className="text-white text-sm font-bold">{avatarLetter}</span>
        </button>
      </div>
    )
  }

  /* ── EXPANDED — full sidebar ── */
  return (
    <div className="w-80 flex-shrink-0 flex flex-col h-full select-none" style={sidebarStyle}>

      {/* Logo row */}
      <div className="flex items-center gap-3 px-5 pt-5 pb-4">
        <div className="w-9 h-9 bg-gradient-to-br from-violet-500 to-blue-500 rounded-lg flex items-center justify-center flex-shrink-0 shadow">
          <span className="text-white text-[15px] font-bold leading-none">✦</span>
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-lg font-bold leading-tight" style={{
            background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
          }}>MAGIK</div>
        </div>
        <button
          onClick={onToggleCollapse}
          className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-md transition-colors"
          style={{ color: 'var(--t-tx5)' }}
          onMouseEnter={e => { e.currentTarget.style.color = 'var(--t-tx1)'; e.currentTarget.style.background = 'var(--t-hov2)' }}
          onMouseLeave={e => { e.currentTarget.style.color = 'var(--t-tx5)'; e.currentTarget.style.background = 'transparent' }}
          title="Collapse sidebar"
        >
          <PanelLeftClose size={17} />
        </button>
      </div>

      {/* New Chat */}
      <div className="px-4 mb-5">
        <button
          onClick={onNewChat}
          className="w-full flex items-center gap-2.5 text-white rounded-xl py-3 px-4 text-base font-medium transition-all"
          style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)' }}
          onMouseEnter={e => e.currentTarget.style.opacity = '0.85'}
          onMouseLeave={e => e.currentTarget.style.opacity = '1'}
        >
          <SquarePen size={16} />
          New chat
        </button>
      </div>

      {/* KB header */}
      <div className="px-5 mb-2">
        <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--t-tx5)' }}>
          Knowledge Base{kbFiles.length > 0 ? ` (${kbFiles.length})` : ''}
        </span>
      </div>

      {/* File list + drop zone */}
      <div className="flex-1 overflow-y-auto px-4 pb-2 space-y-0.5">

        {/* Skeleton while loading */}
        {loadingKB && kbFiles.length === 0 && (
          [0,1,2].map(i => (
            <div key={i} className="skeleton h-9 mx-1 mb-1" style={{ opacity: 1 - i * 0.2 }} />
          ))
        )}

        {!loadingKB && kbFiles.length === 0 && uploadingFiles.size === 0 && (
          <p className="text-sm text-center py-4 leading-relaxed px-2" style={{ color: 'var(--t-tx6)' }}>
            No files yet.<br/>Drop files below to get started.
          </p>
        )}

        {kbFiles.map((f) => {
          const b           = badge(f.filename)
          const isNew       = newFiles.has(f.filename)
          const fileUploading = uploadingFiles.has(f.filename)
          return (
            <div
              key={f.filename}
              className={`flex items-center gap-2.5 group rounded-lg px-2.5 py-2.5 transition-colors cursor-default ${isNew ? 'file-row-enter' : ''}`}
              onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov)'}
              onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
            >
              {fileUploading
                ? <Loader2 size={15} className="animate-spin flex-shrink-0" style={{ color: 'var(--t-accent)' }} />
                : <FileIcon filename={f.filename} />
              }
              <span className="flex-1 text-sm truncate min-w-0" style={{ color: 'var(--t-tx3)' }}>{f.filename}</span>
              <span className={`${b.bg} text-white text-[10px] font-bold px-1.5 py-0.5 rounded flex-shrink-0`}>{b.label}</span>
              <button
                onClick={() => handleDelete(f.filename)}
                className="hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all flex-shrink-0"
                style={{ color: 'var(--t-tx6)' }}
                title={`Remove ${f.filename}`}
              >
                <X size={13} />
              </button>
            </div>
          )
        })}

        {/* Drop zone */}
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !isUploading && fileInputRef.current?.click()}
          className={`mt-3 border border-dashed rounded-xl py-6 flex flex-col items-center gap-2 transition-colors ${isUploading ? 'cursor-default' : 'cursor-pointer'}`}
          style={{
            borderColor: dragOver ? 'var(--t-accent)' : 'var(--t-bd2)',
            background: dragOver ? 'var(--t-accent-soft)' : 'transparent',
          }}
          onMouseEnter={e => { if (!isUploading && !dragOver) e.currentTarget.style.borderColor = 'var(--t-bd4)' }}
          onMouseLeave={e => { if (!dragOver) e.currentTarget.style.borderColor = 'var(--t-bd2)' }}
        >
          {isUploading
            ? <Loader2 size={18} className="animate-spin" style={{ color: 'var(--t-accent)' }} />
            : <Upload size={18} style={{ color: 'var(--t-tx5)' }} />
          }
          <span className="text-sm text-center leading-snug px-3" style={{ color: 'var(--t-ph)' }}>
            {isUploading ? 'Uploading…' : 'Drop files or click to add'}
          </span>
          <input ref={fileInputRef} type="file" multiple className="hidden" onChange={e => handleUpload(Array.from(e.target.files))} />
        </div>

        {uploadError && (
          <div className="mt-2 flex items-start gap-1.5 text-red-400 text-sm px-1">
            <AlertCircle size={13} className="mt-0.5 flex-shrink-0" />
            <span className="leading-tight">{uploadError}</span>
          </div>
        )}
      </div>

      {/* User row */}
      <div className="px-4 py-3.5 flex items-center gap-3" style={{ borderTop: '1px solid var(--t-bd1)' }}>
        <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)' }}>
          <span className="text-white text-sm font-bold">{avatarLetter}</span>
        </div>
        <span className="flex-1 text-sm truncate min-w-0" style={{ color: 'var(--t-tx4)' }}>{auth.email}</span>
        <button
          onClick={onToggleTheme}
          className="flex-shrink-0 transition-colors"
          style={{ color: 'var(--t-tx5)' }}
          onMouseEnter={e => e.currentTarget.style.color = 'var(--t-accent)'}
          onMouseLeave={e => e.currentTarget.style.color = 'var(--t-tx5)'}
          title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {dark ? <Sun size={15} /> : <Moon size={15} />}
        </button>
        <button
          onClick={onLogout}
          className="flex-shrink-0 transition-colors"
          style={{ color: 'var(--t-tx5)' }}
          onMouseEnter={e => e.currentTarget.style.color = 'var(--t-tx4)'}
          onMouseLeave={e => e.currentTarget.style.color = 'var(--t-tx5)'}
          title="Sign out"
        >
          <LogOut size={15} />
        </button>
      </div>
    </div>
  )
}
