import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import {
  Upload, X, Loader2, LogOut, FileText, Image, Sheet,
  File, AlertCircle, PanelLeftClose, Sun, Moon,
  SquarePen, FolderOpen, FileVideo, FileAudio, FileType, LetterText, Settings,
  BrainCircuit, MessageSquare, MoreHorizontal, Pin, PinOff, Pencil,
  Archive, ArchiveRestore, Trash2,
} from 'lucide-react'
import { listKB, deleteKBFile, ingestFile, listChatSessions, deleteChatSession, updateChatSession } from '../api/client'
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
  const [uploadError, setUploadError]     = useState('')
  const [loadingKB, setLoadingKB]         = useState(true)
  const [newFiles, setNewFiles]           = useState(new Set())
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

  /* ── Account menu (Settings / Log out) — shared by both layouts ── */
  const AccountMenu = ({ panelRef, className, style }) => (
    <div ref={panelRef} className={className}
      style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)', ...style }}>
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
      <button
        onClick={() => { setMenuOpen(false); onLogout() }}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-sm text-left transition-colors"
        style={{ color: 'var(--t-tx3)' }}
        onMouseEnter={e => e.currentTarget.style.background = 'var(--t-hov)'}
        onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
      >
        <LogOut size={15} />
        Log out
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

  useEffect(() => { refreshKB() }, [])

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
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center flex-shrink-0">
            <BrainCircuit size={16} strokeWidth={2} className="text-white" />
          </div>
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

        {/* Avatar → account menu */}
        <button
          ref={avatarBtnRef}
          onClick={() => setMenuOpen(o => !o)}
          title={auth.email}
          className="mt-1 w-8 h-8 rounded-full flex items-center justify-center transition-transform"
          style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)', transform: menuOpen ? 'scale(1.06)' : 'scale(1)' }}
        >
          <span className="text-white text-[11px] font-bold tracking-tight">{avatarInitials}</span>
        </button>
      </div>

      {menuOpen && menuPos && createPortal(
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
    <div className="w-80 flex-shrink-0 flex flex-col h-full select-none" style={sidebarStyle}>

      {/* Logo row */}
      <div className="flex items-center gap-3 px-5 pt-5 pb-4">
        <div className="w-9 h-9 bg-gradient-to-br from-violet-500 to-blue-500 rounded-lg flex items-center justify-center flex-shrink-0 shadow">
          <BrainCircuit size={19} strokeWidth={2} className="text-white" />
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

      {/* Recents */}
      {(loadingSessions || sessions.length > 0) && (
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
      )}

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
          <div className="mt-2 flex items-start gap-1.5 text-sm px-1" style={{ color: 'var(--t-danger)' }}>
            <AlertCircle size={13} className="mt-0.5 flex-shrink-0" />
            <span className="leading-tight">{uploadError}</span>
          </div>
        )}
      </div>

      {/* User row */}
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
    </div>
  )
}
