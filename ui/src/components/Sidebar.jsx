import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  Upload, X, Loader2, LogOut, User, FileText, Image, Sheet,
  File, AlertCircle, PanelLeftClose, Sun, Moon,
  SquarePen, FolderOpen, FileVideo, FileAudio, FileType, LetterText, Settings,
  MessageSquare, MoreHorizontal, Pin, PinOff, Pencil,
  Archive, ArchiveRestore, Trash2,
} from 'lucide-react'
import { listKB, deleteKBFile, ingestFile, listChatSessions, deleteChatSession, updateChatSession, getIngestionStatus } from '../api/client'
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
  DOCX: { label: 'DOC', bg: 'bg-indigo-700' },
  DOC:  { label: 'DOC', bg: 'bg-indigo-700' },
}

function getInitials(email) {
  const local = (email || '').split('@')[0].trim()
  if (!local) return 'U'
  const parts = local.split(/[._\-+\s]+/).filter(Boolean)
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase()
  return local.slice(0, 2).toUpperCase() || 'U'
}

function badge(filename) {
  const ext = (filename.split('.').pop() || '').toUpperCase()
  return EXT_BADGE[ext] || { label: ext.slice(0, 3) || '?', bg: 'bg-gray-700' }
}

function formatSize(bytes) {
  if (!bytes) return ''
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

const EXT_CARD_COLOR = {
  PDF:  '#dc2626',
  PNG:  '#0f766e', JPG: '#0f766e', JPEG: '#0f766e', GIF: '#0f766e', WEBP: '#0f766e',
  XLS:  '#15803d', XLSX: '#15803d', CSV: '#166534',
  MP3:  '#7e22ce', WAV: '#7e22ce', M4A: '#7e22ce', OGG: '#7e22ce', FLAC: '#7e22ce',
  MP4:  '#1d4ed8', MOV: '#1d4ed8', AVI: '#1d4ed8', MKV: '#1d4ed8', WEBM: '#1d4ed8',
  TXT:  '#475569',
  DOCX: '#4338ca', DOC: '#4338ca',
}
function uploadCardColor(filename) {
  const ext = (filename.split('.').pop() || '').toUpperCase()
  return EXT_CARD_COLOR[ext] || '#475569'
}

function CircularProgress({ pct }) {
  const size  = 16
  const sw    = 2.2
  const r     = (size - sw) / 2
  const circ  = 2 * Math.PI * r
  const offset = circ * (1 - Math.min(pct, 100) / 100)
  const cx = size / 2
  const cy = size / 2

  if (pct >= 100) {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="flex-shrink-0">
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="#22c55e" strokeWidth={sw} />
      </svg>
    )
  }

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="flex-shrink-0">
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--t-bd4)" strokeWidth={sw} />
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#8b5cf6" strokeWidth={sw}
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        transform={`rotate(-90 ${cx} ${cy})`}
        style={{ transition: 'stroke-dashoffset 0.4s ease' }} />
    </svg>
  )
}

function UploadCardIcon({ filename }) {
  const ext = (filename.split('.').pop() || '').toUpperCase()
  if (['PNG','JPG','JPEG','GIF','WEBP'].includes(ext)) return <Image size={20} className="text-white" />
  if (['XLS','XLSX','CSV'].includes(ext))              return <Sheet size={20} className="text-white" />
  if (ext === 'PDF')                                   return <FileText size={20} className="text-white" />
  if (['MP4','MOV','AVI','MKV','WEBM'].includes(ext)) return <FileVideo size={20} className="text-white" />
  if (['MP3','WAV','M4A','OGG','FLAC'].includes(ext)) return <FileAudio size={20} className="text-white" />
  if (ext === 'TXT')                                   return <FileType size={20} className="text-white" />
  if (['DOC','DOCX'].includes(ext))                    return <LetterText size={20} className="text-white" />
  return <File size={20} className="text-white" />
}

function FileIcon({ filename }) {
  const ext = (filename.split('.').pop() || '').toUpperCase()
  if (['PNG','JPG','JPEG','GIF','WEBP'].includes(ext))
    return <Image size={15} className="text-teal-400 flex-shrink-0" />
  if (['XLS','XLSX','CSV'].includes(ext))
    return <Sheet size={15} className="text-green-400 flex-shrink-0" />
  if (ext === 'PDF')
    return <FileText size={15} className="text-red-400 flex-shrink-0" />
  if (['MP4','MOV','AVI','MKV','WEBM'].includes(ext))
    return <FileVideo size={15} className="text-blue-400 flex-shrink-0" />
  if (['MP3','WAV','M4A','OGG','FLAC'].includes(ext))
    return <FileAudio size={15} className="text-purple-400 flex-shrink-0" />
  if (ext === 'TXT')
    return <FileType size={15} className="text-gray-400 flex-shrink-0" />
  if (['DOC','DOCX'].includes(ext))
    return <LetterText size={15} className="text-blue-400 flex-shrink-0" />
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
  currentSessionId, onSelectSession, streaming,
  collapsed, onToggleCollapse,
  dark, onToggleTheme,
  onOpenSettings,
  onSetUploadHandler,
  historyClearedAt,
  staleSessionId,
  onGuestUploadLimit,
  onShowLogin,
}) {
  const [dragOver, setDragOver]           = useState(false)
  const [sessions, setSessions]           = useState([])
  const [loadingSessions, setLoadingSessions] = useState(true)
  const [sessionMenuId, setSessionMenuId]  = useState(null)
  const [sessionMenuPos, setSessionMenuPos] = useState(null)
  const [renamingId, setRenamingId]        = useState(null)
  const [renameValue, setRenameValue]      = useState('')
  const [showArchived, setShowArchived]    = useState(false)
  const sessionMenuTriggerRef              = useRef(null)
  const sessionMenuPanelRef                = useRef(null)
  const renameInputRef                     = useRef(null)
  const [uploadingFiles, setUploadingFiles] = useState(new Set())
  const [uploadProgress, setUploadProgress] = useState({}) // filename → 0-100 (smoothed, displayed)
  const uploadTargets = useRef({})       // filename → real stage target the bar eases toward
  const uploadControllers = useRef({}) // filename → AbortController
  const [uploadError, setUploadError]     = useState('')
  const [dupPrompt, setDupPrompt]         = useState(null) // { fresh: File[], dups: File[] }
  const [loadingKB, setLoadingKB]         = useState(true)
  const [newFiles, setNewFiles]           = useState(new Set())
  const [kbTooltip, setKbTooltip]         = useState({ text: '', x: 0, y: 0 })
  const [menuOpen, setMenuOpen]           = useState(false)
  const [menuPos, setMenuPos]             = useState(null)
  const fileInputRef                      = useRef(null)
  const prevFilenamesRef                  = useRef(new Set())
  const menuRef                           = useRef(null)
  const menuPanelRef                      = useRef(null)
  const avatarBtnRef                      = useRef(null)
  const avatarInitials                    = getInitials(auth.email)
  const { addToast }                      = useToast()

  // Collapsed rail: the menu is portalled to <body>, so position it off the trigger's rect
  useLayoutEffect(() => {
    if (menuOpen && collapsed && avatarBtnRef.current) {
      const r = avatarBtnRef.current.getBoundingClientRect()
      setMenuPos({ left: r.right + 10, bottom: window.innerHeight - r.bottom })
    }
  }, [menuOpen, collapsed])

  useEffect(() => {
    if (!menuOpen) return
    const onClickOutside = (e) => {
      const inTrigger = menuRef.current?.contains(e.target)
      const inPanel   = menuPanelRef.current?.contains(e.target)
      if (!inTrigger && !inPanel) setMenuOpen(false)
    }
    const onEscape = (e) => { if (e.key === 'Escape') setMenuOpen(false) }
    document.addEventListener('mousedown', onClickOutside)
    document.addEventListener('keydown', onEscape)
    return () => {
      document.removeEventListener('mousedown', onClickOutside)
      document.removeEventListener('keydown', onEscape)
    }
  }, [menuOpen])

  // Per-row session menu (Pin / Rename / Archive / Delete) — portalled to <body>
  useEffect(() => {
    if (!sessionMenuId) return
    const onClickOutside = (e) => {
      const inTrigger = sessionMenuTriggerRef.current?.contains(e.target)
      const inPanel   = sessionMenuPanelRef.current?.contains(e.target)
      if (!inTrigger && !inPanel) setSessionMenuId(null)
    }
    const onEscape = (e) => { if (e.key === 'Escape') setSessionMenuId(null) }
    document.addEventListener('mousedown', onClickOutside)
    document.addEventListener('keydown', onEscape)
    return () => {
      document.removeEventListener('mousedown', onClickOutside)
      document.removeEventListener('keydown', onEscape)
    }
  }, [sessionMenuId])

  // Focus + select the inline rename field when it appears
  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus()
      renameInputRef.current.select()
    }
  }, [renamingId])

  /* ── Account menu (Settings / Log in or Log out) — shared by both layouts ── */
  const AccountMenu = ({ panelRef, className, style }) => (
    <div ref={panelRef} className={className}
      style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)', ...style }}>
      {!auth?.isGuest && (
        <>
          <button
            onClick={() => { setMenuOpen(false); onOpenSettings?.() }}
            className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-sm text-left transition-colors"
            style={{ color: 'var(--t-tx3)' }}
            onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov)'}
            onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
          >
            <Settings size={15} />
            Settings
          </button>
          <div className="h-px mx-2 my-1" style={{ background: 'var(--t-bd1)' }} />
        </>
      )}
      <button
        onClick={() => { setMenuOpen(false); auth?.isGuest ? onShowLogin?.() : onLogout() }}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-sm text-left transition-colors"
        style={{ color: 'var(--t-tx3)' }}
        onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov)'}
        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
      >
        {auth?.isGuest ? <User size={15} /> : <LogOut size={15} />}
        {auth?.isGuest ? 'Log in' : 'Log out'}
      </button>
    </div>
  )

  /* ── Per-row session menu: Pin / Rename / Archive / Delete ── */
  const sessionMenuItemStyle = {
    color: 'var(--t-tx3)',
  }
  const sessionMenuItemHover = (e) => { e.currentTarget.style.background = 'var(--t-hov)' }
  const sessionMenuItemLeave = (e) => { e.currentTarget.style.background = 'transparent' }

  const SessionMenu = ({ session }) => (
    <div
      ref={sessionMenuPanelRef}
      className="fixed z-50 w-48 rounded-xl shadow-2xl py-1.5 overflow-hidden"
      style={{ top: sessionMenuPos?.top, left: sessionMenuPos?.left, background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}
    >
      <button
        onClick={(e) => handleTogglePin(e, session)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-sm text-left transition-colors"
        style={sessionMenuItemStyle}
        onMouseEnter={sessionMenuItemHover}
        onMouseLeave={sessionMenuItemLeave}
      >
        {session.pinned ? <PinOff size={15} /> : <Pin size={15} />}
        {session.pinned ? 'Unpin' : 'Pin'}
      </button>
      <button
        onClick={(e) => startRenameSession(e, session)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-sm text-left transition-colors"
        style={sessionMenuItemStyle}
        onMouseEnter={sessionMenuItemHover}
        onMouseLeave={sessionMenuItemLeave}
      >
        <Pencil size={15} />
        Rename
      </button>
      <button
        onClick={(e) => handleToggleArchive(e, session)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-sm text-left transition-colors"
        style={sessionMenuItemStyle}
        onMouseEnter={sessionMenuItemHover}
        onMouseLeave={sessionMenuItemLeave}
      >
        {session.archived ? <ArchiveRestore size={15} /> : <Archive size={15} />}
        {session.archived ? 'Unarchive' : 'Archive'}
      </button>
      <div className="h-px mx-2 my-1" style={{ background: 'var(--t-bd1)' }} />
      <button
        onClick={(e) => handleDeleteSession(e, session.session_id)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-sm text-left transition-colors"
        style={{ color: 'var(--t-danger)' }}
        onMouseEnter={sessionMenuItemHover}
        onMouseLeave={sessionMenuItemLeave}
      >
        <Trash2 size={15} />
        Delete
      </button>
    </div>
  )

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

  // Re-fetch whenever the session identity changes (login/logout, guest<->real
  // user, or account switch) — not just on mount. Without auth?.token in the
  // deps, a stale closure keeps whatever the FIRST-mounted auth fetched even
  // after `auth` later flips (e.g. the initial auth-check race in App.jsx),
  // leaking one session's KB/Recents into another's UI. Clear immediately so
  // the previous session's data never lingers on screen while refetching.
  useEffect(() => { prevFilenamesRef.current = new Set(); setKbFiles([]); refreshKB() }, [auth?.token])

  /* ── Smooth upload progress animator ───────────────────────────────────────
   * The server reports progress in coarse stages (queued 2% → extracting 20% →
   * chunking 50% → embedding 80% → done 100%), polled once per second, which
   * makes the bar jump (2→20→80). This ticks the *displayed* percentage up by a
   * small amount every ~70ms toward the current stage target, so the user sees a
   * continuous 1%, 2%, 3%… count. While a stage is still processing (display has
   * reached its target but it isn't done yet) the bar trickles forward slowly,
   * capped below the next milestone, so it never looks frozen or overshoots. */
  useEffect(() => {
    if (uploadingFiles.size === 0) return
    let tick = 0
    const id = setInterval(() => {
      tick++
      setUploadProgress(prev => {
        let changed = false
        const next = { ...prev }
        for (const name of Object.keys(next)) {
          const target = uploadTargets.current[name] ?? next[name]
          const cur = next[name]
          if (cur < target) {
            // Ease toward the real milestone — faster when far, ≥1 so it always moves.
            next[name] = Math.min(target, cur + Math.max(1, Math.ceil((target - cur) / 10)))
            changed = true
          } else if (target < 100 && cur < 95 && tick % 8 === 0) {
            // Stage still processing: gentle trickle so the bar keeps moving,
            // but never reaches the next milestone (cap a bit below 100).
            next[name] = Math.min(95, cur + 1)
            changed = true
          }
        }
        return changed ? next : prev
      })
    }, 70)
    return () => clearInterval(id)
  }, [uploadingFiles])

  /* ── Recents (saved chat sessions) ── */
  const refreshSessions = async () => {
    try {
      setSessions(await listChatSessions(auth.token))
    } catch {}
    setLoadingSessions(false)
  }

  // Refetch on mount, and again whenever a generation finishes — that's the
  // only moment a session's title/recency in Mongo can actually have changed.
  useEffect(() => { if (!streaming) refreshSessions() }, [streaming])

  // Refetch (and clear immediately, before the round-trip) whenever the
  // session identity changes — see the matching KB-refresh effect above for
  // why auth?.token must be a dependency: without it, a stale closure keeps
  // showing whichever session's Recents were fetched first, leaking one
  // account's chat history into another session's UI (e.g. the auth-check
  // race in App.jsx resolving to a guest after the real user's data already
  // rendered).
  useEffect(() => { setSessions([]); refreshSessions() }, [auth?.token])

  // When all history is cleared, empty the list instantly without a round-trip.
  useEffect(() => { if (historyClearedAt) setSessions([]) }, [historyClearedAt])

  // When a session is no longer in MongoDB, remove it from Recents instantly.
  useEffect(() => {
    if (staleSessionId) setSessions(prev => prev.filter(s => s.session_id !== staleSessionId))
  }, [staleSessionId])

  const handleDeleteSession = async (e, id) => {
    e.stopPropagation()
    setSessionMenuId(null)
    // Optimistic: remove from sidebar and switch away immediately
    setSessions(prev => prev.filter(s => s.session_id !== id))
    if (id === currentSessionId) onNewChat?.()
    try {
      await deleteChatSession(auth.token, id)
      addToast('Chat deleted', 'success')
    } catch (err) {
      // 404 = already gone on server — still a success from the user's perspective
      const alreadyGone = err.message?.toLowerCase().includes('not found') ||
                          err.message?.includes('404')
      if (alreadyGone) {
        addToast('Chat deleted', 'success')
      } else {
        addToast(err.message || 'Failed to delete chat', 'error')
      }
    }
  }

  const openSessionMenu = (e, session) => {
    e.stopPropagation()
    if (sessionMenuId === session.session_id) { setSessionMenuId(null); return }
    const r = e.currentTarget.getBoundingClientRect()
    sessionMenuTriggerRef.current = e.currentTarget
    setSessionMenuPos({ top: r.bottom + 6, left: Math.min(r.left, window.innerWidth - 196) })
    setSessionMenuId(session.session_id)
  }

  const handleTogglePin = async (e, session) => {
    e.stopPropagation()
    setSessionMenuId(null)
    try {
      await updateChatSession(auth.token, session.session_id, { pinned: !session.pinned })
      addToast(session.pinned ? 'Chat unpinned' : 'Chat pinned to top', 'success')
      await refreshSessions()
    } catch (err) {
      addToast(err.message || 'Failed to update chat', 'error')
    }
  }

  const handleToggleArchive = async (e, session) => {
    e.stopPropagation()
    setSessionMenuId(null)
    try {
      await updateChatSession(auth.token, session.session_id, { archived: !session.archived })
      addToast(session.archived ? 'Chat moved back to Recents' : 'Chat archived', 'success')
      if (!session.archived && session.session_id === currentSessionId) onNewChat?.()
      await refreshSessions()
    } catch (err) {
      addToast(err.message || 'Failed to update chat', 'error')
    }
  }

  const startRenameSession = (e, session) => {
    e.stopPropagation()
    setSessionMenuId(null)
    setRenamingId(session.session_id)
    setRenameValue(session.title)
  }

  const commitRenameSession = async (id) => {
    const title = renameValue.trim()
    const session = sessions.find(s => s.session_id === id)
    setRenamingId(null)
    if (!session || !title || title === session.title) return
    setSessions(prev => prev.map(s => s.session_id === id ? { ...s, title } : s))
    try {
      await updateChatSession(auth.token, id, { title })
    } catch (err) {
      addToast(err.message || 'Failed to rename chat', 'error')
      await refreshSessions()
    }
  }

  const handleUpload = (filesInput) => {
    const files = Array.isArray(filesInput) ? filesInput : Array.from(filesInput)
    if (!files.length) return
    const existingNames = new Set(kbFiles.map(f => f.filename))
    const dups  = files.filter(f => existingNames.has(f.name))
    const fresh = files.filter(f => !existingNames.has(f.name))
    if (dups.length > 0) { setDupPrompt({ fresh, dups }); return }
    _doUpload(files)
  }

  const handleDupSkip = () => {
    const fresh = dupPrompt?.fresh ?? []
    setDupPrompt(null)
    if (fresh.length) _doUpload(fresh)
  }

  const handleDupReplace = () => {
    const { fresh, dups } = dupPrompt
    setDupPrompt(null)
    _doUpload([...fresh, ...dups])
  }

  // Stage progress map — all modalities now return a job_id immediately (server-side background jobs).
  // queued(2%) → extracting(20%) → chunking(50%) → embedding(80%) → done(100%)
  const _STAGE_PROGRESS = { queued: 2, extracting: 20, chunking: 50, embedding: 80, done: 100, error: 0 }

  const _pollJobStatus = async (token, jobId, filename) => {
    const INTERVAL = 1000, MAX_POLLS = 1800  // poll every 1s; up to 30 min (covers large video files)
    for (let i = 0; i < MAX_POLLS; i++) {
      await new Promise(r => setTimeout(r, INTERVAL))
      try {
        const job = await getIngestionStatus(token, jobId)
        // Prefer the server's real fractional progress when it's ahead of the
        // coarse stage floor, so the bar reflects actual chunk progress.
        const stagePct = _STAGE_PROGRESS[job.status] ?? 0
        const realPct = Math.round((job.progress || 0) * 100)
        const pct = Math.max(stagePct, realPct)
        // Set the TARGET; the animator eases the displayed value toward it.
        uploadTargets.current[filename] = Math.max(uploadTargets.current[filename] || 0, pct)
        if (job.status === 'done') return { ok: true, chunks: job.chunks_done }
        if (job.status === 'error') return { ok: false, error: job.error || 'Processing failed' }
      } catch (_) { /* network blip — keep polling */ }
    }
    return { ok: false, error: 'Timed out waiting for processing' }
  }

  const _doUpload = async (files) => {
    setUploadError('')
    const errors = []
    for (const file of files) {
      const ctrl = new AbortController()
      uploadControllers.current[file.name] = ctrl

      setUploadingFiles(prev => new Set([...prev, file.name]))
      uploadTargets.current[file.name] = 0
      setUploadProgress(prev => ({ ...prev, [file.name]: 0 }))   // animate up from 0

      // Real network-transfer progress (actual bytes sent), mapped to the
      // first 15% of the bar — the remaining 15-100% is real backend
      // pipeline progress from _pollJobStatus below. Previously this whole
      // phase sat frozen at a hardcoded "2%" for as long as the real upload
      // took (tens of seconds for a large audio/video file), so the animator's
      // time-based trickle was the only thing moving the bar — it looked like
      // progress but wasn't tied to anything real. Math.max guards against a
      // late/out-of-order progress event regressing the bar.
      const onUploadProgress = (fraction) => {
        const pct = Math.round(fraction * 15)
        uploadTargets.current[file.name] = Math.max(uploadTargets.current[file.name] || 0, pct)
      }

      try {
        // Server returns job_id immediately (<1s) for ALL modalities — file is already
        // visible in sidebar. Poll real pipeline stages until done.
        const result = await ingestFile(auth.token, file, 'default', ctrl, onUploadProgress)

        if (result?.job_id) {
          const { ok, error: pollErr, chunks } = await _pollJobStatus(auth.token, result.job_id, file.name)
          if (ok) {
            uploadTargets.current[file.name] = 100        // let the animator finish to 100
            await new Promise(r => setTimeout(r, 550))    // ...and let that climb render
            addToast(`Uploaded: ${file.name}${chunks ? ` (${chunks} chunks)` : ''}`, 'success')
          } else {
            errors.push(`${file.name}: ${pollErr}`)
            addToast(`Failed: ${file.name}`, 'error')
          }
        } else {
          // Fallback: older server response without job_id (duplicate or instant result)
          uploadTargets.current[file.name] = 100
          await new Promise(r => setTimeout(r, 550))
          addToast(`Uploaded: ${file.name}`, 'success')
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          addToast(`Cancelled: ${file.name}`, 'info')
          deleteKBFile(auth.token, file.name).catch(() => {})
        } else if (err.message?.toLowerCase().includes('guest upload limit')) {
          if (onGuestUploadLimit) onGuestUploadLimit()
          else addToast('Sign up free to upload more files', 'info')
        } else {
          errors.push(`${file.name}: ${err.message}`)
          addToast(`Failed: ${file.name}`, 'error')
        }
      } finally {
        delete uploadControllers.current[file.name]
      }
      setUploadingFiles(prev => { const s = new Set(prev); s.delete(file.name); return s })
      setUploadProgress(prev => { const p = { ...prev }; delete p[file.name]; return p })
      delete uploadTargets.current[file.name]
    }
    if (errors.length) setUploadError(errors.join('; '))
    await refreshKB()
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleCancelUpload = (filename) => {
    uploadControllers.current[filename]?.abort()
  }

  // Wire to parent so drag-drop onto the chat area triggers KB uploads
  if (onSetUploadHandler) onSetUploadHandler.current = handleUpload

  const handleDelete = async (filename) => {
    try {
      await deleteKBFile(auth.token, filename)
      setKbFiles(prev => prev.filter(f => f.filename !== filename))
      prevFilenamesRef.current.delete(filename)
      addToast(`Deleted: ${filename}`, 'success')
    } catch (err) {
      addToast(err.message || `Failed to delete: ${filename}`, 'error')
    }
  }

  const handleDrop = (e) => {
    e.preventDefault(); setDragOver(false)
    const files = Array.from(e.dataTransfer.files)
    if (files.length) handleUpload(files)
  }

  const isUploading = uploadingFiles.size > 0

  const archivedCount   = sessions.filter(s => s.archived).length
  const visibleSessions = sessions.filter(s => showArchived ? s.archived : !s.archived)
  const menuSession     = sessionMenuId ? sessions.find(s => s.session_id === sessionMenuId) : null

  const sidebarStyle = {
    background: 'var(--t-sb)',
    borderRight: '1px solid var(--t-bd1)',
  }

  /* ── COLLAPSED — icon rail ── */
  if (collapsed) {
    return (
      <>
      <div className="w-[72px] h-full flex flex-col items-center py-4 gap-1 overflow-hidden" style={sidebarStyle}>

        {/* Logo → expands sidebar */}
        <button
          onClick={onToggleCollapse}
          title="Expand sidebar"
          className="w-10 h-10 rounded-xl flex items-center justify-center mb-2 transition-all"
          onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov2)'}
          onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
        >
          <img src="/logo.png" alt="MAGIK" className="w-7 h-7 rounded-lg object-cover flex-shrink-0" />
        </button>

        {/* Divider */}
        <div className="w-8 h-px mb-1" style={{ background: 'var(--t-bd2)' }} />

        {/* New chat */}
        <IconBtn onClick={onNewChat} title="New chat">
          <SquarePen size={18} />
        </IconBtn>

        {/* Knowledge base — expands the sidebar so the KB list becomes visible */}
        <IconBtn onClick={onToggleCollapse} title="Knowledge Base">
          <FolderOpen size={18} />
        </IconBtn>

        {/* Spacer */}
        <div className="flex-1" />

        {/* Theme toggle */}
        <IconBtn onClick={onToggleTheme} title={dark ? 'Light mode' : 'Dark mode'} accentHover>
          {dark ? <Sun size={16} /> : <Moon size={16} />}
        </IconBtn>

        {/* Avatar → account menu (or Log in button for guests) */}
        {auth?.isGuest ? (
          <button
            onClick={onShowLogin}
            title="Log in"
            className="mt-1 w-8 h-8 rounded-full flex items-center justify-center transition-colors"
            style={{ background: 'var(--t-hov2)' }}
          >
            <User size={16} style={{ color: 'var(--t-tx3)' }} />
          </button>
        ) : (
          <>
            <button
              ref={avatarBtnRef}
              onClick={() => setMenuOpen(o => !o)}
              title={auth.email}
              className="mt-1 w-8 h-8 rounded-full flex items-center justify-center transition-transform"
              style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', transform: menuOpen ? 'scale(1.06)' : 'scale(1)' }}
            >
              <span className="text-white text-[11px] font-bold tracking-tight">{avatarInitials}</span>
            </button>
          </>
        )}
      </div>

      {!auth?.isGuest && menuOpen && menuPos && createPortal(
        <AccountMenu
          panelRef={menuPanelRef}
          className="fixed w-44 rounded-xl py-1.5 shadow-xl overflow-hidden z-50"
          style={{ left: menuPos.left, bottom: menuPos.bottom }}
        />,
        document.body
      )}
      </>
    )
  }

  /* ── EXPANDED — full sidebar ── */
  return (
    <>
    <div className="w-80 flex-shrink-0 flex flex-col h-full select-none" style={sidebarStyle}>

      {/* Logo row */}
      <div className="flex items-center gap-3 px-5 pt-5 pb-4">
        <img src="/logo.png" alt="MAGIK" className="w-9 h-9 rounded-lg object-cover flex-shrink-0 shadow" />
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

      {/* Recents */}
      <div className="px-4 mb-1">
          <div className="px-1 mb-2 flex items-center justify-between gap-2">
            <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--t-tx5)' }}>
              {showArchived ? 'Archived' : 'Recents'}
            </span>
            {showArchived ? (
              <button
                onClick={() => setShowArchived(false)}
                className="text-[11px] font-medium transition-colors flex-shrink-0"
                style={{ color: 'var(--t-accent)' }}
              >
                ← Back to Recents
              </button>
            ) : archivedCount > 0 && (
              <button
                onClick={() => setShowArchived(true)}
                className="flex items-center gap-1 text-[11px] font-medium transition-colors flex-shrink-0"
                style={{ color: 'var(--t-tx5)' }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--t-tx3)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--t-tx5)'}
                title="View archived chats"
              >
                <Archive size={11} />
                {archivedCount} archived
              </button>
            )}
          </div>
          <div className="max-h-[38vh] overflow-y-auto space-y-0.5 pr-0.5">
            {loadingSessions && sessions.length === 0 && (
              [0, 1].map(i => (
                <div key={i} className="skeleton h-9 mx-1 mb-1" style={{ opacity: 1 - i * 0.25 }} />
              ))
            )}
            {!loadingSessions && !showArchived && sessions.length === 0 && (
              <div className="px-2.5 py-3 text-sm text-center leading-relaxed" style={{ color: 'var(--t-tx6)' }}>
                No chats yet.<br />
                <button
                  onClick={onNewChat}
                  className="text-xs mt-1.5"
                  style={{ color: 'var(--t-accent)' }}
                >
                  Start your first chat →
                </button>
              </div>
            )}
            {!loadingSessions && showArchived && visibleSessions.length === 0 && (
              <div className="px-2.5 py-4 text-sm text-center" style={{ color: 'var(--t-tx5)' }}>
                No archived chats
              </div>
            )}
            {visibleSessions.map((s) => {
              const active   = s.session_id === currentSessionId
              const renaming = renamingId === s.session_id
              const menuOpenForRow = sessionMenuId === s.session_id
              return (
                <div
                  key={s.session_id}
                  onClick={() => !renaming && onSelectSession?.(s.session_id)}
                  className="flex items-center gap-2.5 group rounded-lg px-2.5 py-2.5 transition-colors cursor-pointer"
                  style={{ background: active || menuOpenForRow ? 'var(--t-hov2)' : 'transparent' }}
                  onMouseEnter={e => { if (!active && !menuOpenForRow) e.currentTarget.style.background = 'var(--t-hov)' }}
                  onMouseLeave={e => { if (!active && !menuOpenForRow) e.currentTarget.style.background = 'transparent' }}
                >
                  {s.pinned ? (
                    <Pin size={14} className="flex-shrink-0" style={{ color: 'var(--t-accent)' }} />
                  ) : (
                    <MessageSquare size={14} className="flex-shrink-0" style={{ color: active ? 'var(--t-accent)' : 'var(--t-tx5)' }} />
                  )}
                  {renaming ? (
                    <input
                      ref={renameInputRef}
                      value={renameValue}
                      onChange={e => setRenameValue(e.target.value)}
                      onClick={e => e.stopPropagation()}
                      onKeyDown={e => {
                        if (e.key === 'Enter') commitRenameSession(s.session_id)
                        if (e.key === 'Escape') setRenamingId(null)
                      }}
                      onBlur={() => commitRenameSession(s.session_id)}
                      maxLength={200}
                      className="flex-1 text-sm min-w-0 bg-transparent outline-none border-b"
                      style={{ color: 'var(--t-tx1)', borderColor: 'var(--t-accent)' }}
                    />
                  ) : (
                    <span className="flex-1 text-sm truncate min-w-0" style={{ color: active ? 'var(--t-tx1)' : 'var(--t-tx3)' }}>
                      {s.title}
                    </span>
                  )}
                  {!renaming && (
                    <button
                      onClick={(e) => openSessionMenu(e, s)}
                      className={`transition-all flex-shrink-0 ${menuOpenForRow ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'}`}
                      style={{ color: menuOpenForRow ? 'var(--t-tx2)' : 'var(--t-tx6)' }}
                      onMouseEnter={e => e.currentTarget.style.color = 'var(--t-tx2)'}
                      onMouseLeave={e => { if (!menuOpenForRow) e.currentTarget.style.color = 'var(--t-tx6)' }}
                      title="Chat options"
                    >
                      <MoreHorizontal size={15} />
                    </button>
                  )}
                </div>
              )
            })}
          </div>
          <div className="h-px mt-3 mx-1" style={{ background: 'var(--t-bd1)' }} />
        </div>

      {/* Per-row session menu — portalled so it escapes the scroll container */}
      {menuSession && sessionMenuPos && createPortal(
        <SessionMenu session={menuSession} />,
        document.body
      )}

      {/* KB header */}
      <div className="px-5 mb-2">
        <span className="text-xs font-semibold tracking-widest uppercase" style={{ color: 'var(--t-tx5)' }}>
          Knowledge Base{kbFiles.length > 0 ? ` (${kbFiles.length})` : ''}
        </span>
      </div>

      {/* KB drop zone — compact strip pinned below heading */}
      <div className="px-4 mb-2">
        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => !isUploading && fileInputRef.current?.click()}
          className={`border border-dashed rounded-lg py-2.5 flex items-center justify-center gap-2 transition-colors ${isUploading ? 'cursor-default' : 'cursor-pointer'}`}
          style={{
            borderColor: dragOver ? 'var(--t-accent)' : 'var(--t-bd2)',
            background: dragOver ? 'var(--t-accent-soft)' : 'transparent',
          }}
          onMouseEnter={e => { if (!isUploading && !dragOver) e.currentTarget.style.borderColor = 'var(--t-bd4)' }}
          onMouseLeave={e => { if (!dragOver) e.currentTarget.style.borderColor = 'var(--t-bd2)' }}
          aria-label="Upload files to knowledge base"
          role="button"
        >
          <Upload size={13} style={{ color: isUploading ? 'var(--t-accent)' : 'var(--t-tx5)', flexShrink: 0 }} />
          <span className="text-xs" style={{ color: isUploading ? 'var(--t-accent)' : 'var(--t-ph)' }}>
            {isUploading ? 'Drop more files' : 'Drop files or click to add'}
          </span>
          <input ref={fileInputRef} type="file" multiple className="hidden" aria-label="File input" onChange={e => handleUpload(Array.from(e.target.files))} />
        </div>
      </div>

      {/* File list */}
      <div className="flex-1 overflow-y-auto px-4 pb-2 space-y-0.5">

        {/* Skeleton while loading */}
        {loadingKB && kbFiles.length === 0 && (
          [0,1,2].map(i => (
            <div key={i} className="skeleton h-9 mx-1 mb-1" style={{ opacity: 1 - i * 0.2 }} />
          ))
        )}

        {!loadingKB && kbFiles.length === 0 && uploadingFiles.size === 0 && (
          <p className="text-xs text-center py-3" style={{ color: 'var(--t-tx6)' }}>
            No files yet
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
              <span
                className="flex-1 text-sm truncate min-w-0"
                style={{ color: 'var(--t-tx3)' }}
                onMouseEnter={e => {
                  const r = e.currentTarget.getBoundingClientRect()
                  setKbTooltip({ text: f.filename, x: r.left, y: r.bottom + 6 })
                }}
                onMouseLeave={() => setKbTooltip({ text: '', x: 0, y: 0 })}
              >
                {f.filename}
              </span>
              {f.size_bytes > 0 && (
                <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--t-tx6)' }}>
                  {formatSize(f.size_bytes)}
                </span>
              )}
              <span className={`${b.bg} text-white text-[10px] font-bold px-1.5 py-0.5 rounded flex-shrink-0`}>{b.label}</span>
              <button
                onClick={() => handleDelete(f.filename)}
                className="opacity-0 group-hover:opacity-100 transition-all flex-shrink-0"
                style={{ color: 'var(--t-tx6)' }}
                onMouseEnter={e => e.currentTarget.style.color = 'var(--t-danger)'}
                onMouseLeave={e => e.currentTarget.style.color = 'var(--t-tx6)'}
                title={`Remove ${f.filename}`}
              >
                <X size={13} />
              </button>
            </div>
          )
        })}

        {/* Per-file upload cards */}
        {Object.entries(uploadProgress).map(([name, pct]) => {
          const b = badge(name)
          return (
            <div key={name} className="mt-2 rounded-xl"
              style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}>
              <div className="flex items-center gap-3 px-3 py-2.5">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0"
                  style={{ background: uploadCardColor(name) }}>
                  <UploadCardIcon filename={name} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate" style={{ color: 'var(--t-tx2)' }}>{name}</p>
                  <p className="text-[11px] mt-0.5 flex items-center gap-1.5" style={{ color: 'var(--t-tx5)' }}>
                    <CircularProgress pct={pct} />
                    {pct < 100 ? `Uploading ${pct}%` : 'Processing…'}
                  </p>
                </div>
                <span className={`${b.bg} text-white text-[10px] font-bold px-1.5 py-0.5 rounded flex-shrink-0`}>
                  {b.label}
                </span>
                <button
                  onClick={() => handleCancelUpload(name)}
                  title="Cancel upload"
                  aria-label="Cancel upload"
                  className="flex-shrink-0 rounded p-1 transition-colors"
                  style={{ color: 'var(--t-tx5)' }}
                  onMouseEnter={e => { e.currentTarget.style.color = 'var(--t-danger)'; e.currentTarget.style.background = 'var(--t-hov)' }}
                  onMouseLeave={e => { e.currentTarget.style.color = 'var(--t-tx5)'; e.currentTarget.style.background = 'transparent' }}
                >
                  <X size={13} />
                </button>
              </div>
            </div>
          )
        })}

        {/* Duplicate-file warning */}
        {dupPrompt && (
          <div className="mt-2 rounded-xl p-3"
            style={{ background: 'rgba(234,179,8,0.07)', border: '1px solid rgba(234,179,8,0.28)' }}>
            <div className="flex items-start gap-2 mb-2.5">
              <AlertCircle size={14} className="flex-shrink-0 mt-0.5" style={{ color: '#ca8a04' }} />
              <div className="min-w-0">
                <p className="text-xs font-semibold mb-1" style={{ color: '#ca8a04' }}>
                  {dupPrompt.dups.length === 1 ? 'File already in KB' : `${dupPrompt.dups.length} files already in KB`}
                </p>
                {dupPrompt.dups.map(f => (
                  <p key={f.name} className="text-[11px] truncate" style={{ color: 'var(--t-tx4)' }}>{f.name}</p>
                ))}
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={handleDupSkip}
                className="flex-1 text-[11px] py-1.5 rounded-lg font-medium transition-colors"
                style={{ background: 'var(--t-hov2)', color: 'var(--t-tx3)' }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov3)'}
                onMouseLeave={e => e.currentTarget.style.background = 'var(--t-hov2)'}>
                Skip
              </button>
              <button onClick={handleDupReplace}
                className="flex-1 text-[11px] py-1.5 rounded-lg font-medium transition-colors"
                style={{ background: 'rgba(234,179,8,0.15)', color: '#ca8a04' }}
                onMouseEnter={e => e.currentTarget.style.background = 'rgba(234,179,8,0.25)'}
                onMouseLeave={e => e.currentTarget.style.background = 'rgba(234,179,8,0.15)'}>
                Replace
              </button>
            </div>
          </div>
        )}

        {uploadError && (
          <div className="mt-2 flex items-start gap-1.5 text-sm px-1" style={{ color: 'var(--t-danger)' }}>
            <AlertCircle size={13} className="mt-0.5 flex-shrink-0" />
            <span className="leading-tight">{uploadError}</span>
          </div>
        )}
      </div>

      {/* Bottom section — guest CTA or real user row */}
      {auth?.isGuest ? (
        <div className="px-4 pt-4 pb-3" style={{ borderTop: '1px solid var(--t-bd1)' }}>
          <p className="text-sm font-bold mb-1" style={{ color: 'var(--t-tx1)' }}>
            Get responses tailored to you
          </p>
          <p className="text-xs mb-3 leading-relaxed" style={{ color: 'var(--t-tx5)' }}>
            Log in to save your documents, chat history, and unlock unlimited queries.
          </p>
          <button
            onClick={onShowLogin}
            className="w-full flex items-center justify-center rounded-full py-3 text-sm font-bold transition-opacity mb-2"
            style={{ background: 'var(--t-tx1)', color: 'var(--t-bg)' }}
            onMouseEnter={e => e.currentTarget.style.opacity = '0.88'}
            onMouseLeave={e => e.currentTarget.style.opacity = '1'}
          >
            Log in
          </button>
          <button
            onClick={onToggleTheme}
            className="w-full flex items-center justify-center gap-1.5 rounded-lg py-1.5 text-xs transition-colors"
            style={{ color: 'var(--t-tx6)' }}
            onMouseEnter={e => e.currentTarget.style.color = 'var(--t-tx4)'}
            onMouseLeave={e => e.currentTarget.style.color = 'var(--t-tx6)'}
            title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {dark ? <Sun size={13} /> : <Moon size={13} />}
            {dark ? 'Light mode' : 'Dark mode'}
          </button>
        </div>
      ) : (
        <div ref={menuRef} className="relative px-4 py-3.5" style={{ borderTop: '1px solid var(--t-bd1)' }}>
          {/* Account menu */}
          {menuOpen && (
            <AccountMenu
              panelRef={menuPanelRef}
              className="absolute left-4 right-4 rounded-xl py-1.5 shadow-xl overflow-hidden z-20"
              style={{ bottom: 'calc(100% + 6px)' }}
            />
          )}

          {/* Trigger row */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => setMenuOpen(o => !o)}
              title={auth.email}
              className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 transition-transform"
              style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', transform: menuOpen ? 'scale(1.06)' : 'scale(1)' }}
            >
              <span className="text-white text-[11px] font-bold tracking-tight">{avatarInitials}</span>
            </button>
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
          </div>
        </div>
      )}
    </div>

    {/* KB filename tooltip — portalled to escape overflow container */}
    {kbTooltip.text && createPortal(
      <div
        className="fixed z-[300] text-xs px-2.5 py-1.5 rounded-lg pointer-events-none break-all max-w-[280px]"
        style={{
          top:        kbTooltip.y,
          left:       kbTooltip.x,
          background: 'var(--t-card)',
          border:     '1px solid var(--t-bd3)',
          color:      'var(--t-tx1)',
          boxShadow:  '0 4px 16px rgba(0,0,0,0.35)',
        }}
      >
        {kbTooltip.text}
      </div>,
      document.body
    )}
    </>
  )
}
