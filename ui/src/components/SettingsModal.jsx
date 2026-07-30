import { useState, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import {
  X, User, Database, ShieldCheck, ShieldAlert, Sun, Moon,
  Loader2, Trash2, FileText, LogOut, RotateCcw, AlertTriangle, ChevronRight, ChevronLeft,
  Keyboard,
} from 'lucide-react'
import { listKB, deleteKBFile, getMe, changePassword, logoutAll, deleteAccount, clearMemory, deleteAllChatSessions } from '../api/client'
import { useToast } from '../context/ToastContext'
import useIsMobile from '../hooks/useIsMobile'

const NAV = [
  { id: 'account',   label: 'Account',         Icon: User },
  { id: 'kb',        label: 'Knowledge base',  Icon: Database },
  { id: 'security',  label: 'Security',        Icon: ShieldCheck },
  { id: 'privacy',   label: 'Privacy & data',  Icon: ShieldAlert },
  { id: 'shortcuts', label: 'Shortcuts',       Icon: Keyboard },
]

function initialsOf(email) {
  const local = (email || '').split('@')[0].trim()
  if (!local) return 'U'
  const parts = local.split(/[._\-+\s]+/).filter(Boolean)
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase()
  return local.slice(0, 2).toUpperCase() || 'U'
}

/* ── Shared bits ───────────────────────────────────── */
function Toggle({ checked, onChange }) {
  return (
    <button
      type="button"
      role="switch" aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="relative w-9 h-5 rounded-full flex-shrink-0 transition-colors duration-200"
      style={{ background: checked ? 'var(--t-accent)' : 'var(--t-bd4)' }}
    >
      <span
        className="absolute top-0.5 w-4 h-4 rounded-full bg-white transition-all duration-200"
        style={{ left: checked ? 18 : 2 }}
      />
    </button>
  )
}

function Row({ title, desc, children }) {
  return (
    <div className="flex items-center justify-between gap-4 py-3.5" style={{ borderTop: '1px solid var(--t-bd1)' }}>
      <div className="min-w-0">
        <p className="text-sm font-medium" style={{ color: 'var(--t-tx2)' }}>{title}</p>
        {desc && <p className="text-[12.5px] mt-0.5 leading-snug" style={{ color: 'var(--t-tx5)' }}>{desc}</p>}
      </div>
      {children}
    </div>
  )
}

function Card({ title, children }) {
  return (
    <div className="rounded-2xl px-5" style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}>
      {title && <h3 className="text-[12px] font-semibold uppercase tracking-wide pt-4 pb-1" style={{ color: 'var(--t-tx5)' }}>{title}</h3>}
      {children}
    </div>
  )
}

function GhostButton({ onClick, disabled, icon: Icon, spinning, children, style, ...rest }) {
  return (
    <button
      type="button" onClick={onClick} disabled={disabled}
      className="flex items-center gap-2 rounded-lg px-3 py-1.5 text-[13px] transition-colors disabled:opacity-50 flex-shrink-0"
      style={{ background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx3)', ...style }}
      onMouseEnter={e => { if (!disabled) e.currentTarget.style.background = 'var(--t-hov2)' }}
      onMouseLeave={e => { e.currentTarget.style.background = 'var(--t-inp)' }}
      {...rest}
    >
      {spinning ? <Loader2 size={13} className="animate-spin" /> : Icon && <Icon size={13} />}
      {children}
    </button>
  )
}

const inputStyle = { background: 'var(--t-inp)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx1)' }
const inputCls   = 'w-full rounded-xl px-4 py-2.5 text-sm outline-none transition-colors t-focus'

/* ── Account ───────────────────────────────────────── */
function AccountSection({ auth, dark, onToggleTheme, showSources, setShowSources, sessionId, addToast, onClearConversation, onClearAllHistory }) {
  const [profile, setProfile] = useState(null)
  const [clearing, setClearing] = useState(false)
  const [clearingHistory, setClearingHistory] = useState(false)
  const [confirmClearHistory, setConfirmClearHistory] = useState(false)

  useEffect(() => { getMe(auth.token).then(setProfile).catch(() => {}) }, [auth.token])

  const memberSince = profile?.created_at
    ? new Date(profile.created_at).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })
    : null

  const handleClearMemory = async () => {
    setClearing(true)
    try {
      await clearMemory(auth.token, sessionId)
      addToast('Conversation cleared', 'success')
      onClearConversation?.()
    } catch (err) {
      addToast(err.message, 'error')
    } finally {
      setClearing(false)
    }
  }

  const handleClearAllHistory = async () => {
    if (!confirmClearHistory) { setConfirmClearHistory(true); return }
    setClearingHistory(true)
    setConfirmClearHistory(false)
    try {
      await deleteAllChatSessions(auth.token)
      addToast('All chat history deleted', 'success')
      onClearAllHistory?.()
    } catch (err) {
      addToast(err.message, 'error')
    } finally {
      setClearingHistory(false)
    }
  }

  return (
    <div className="space-y-5">
      <div className="rounded-2xl p-5 flex items-center gap-4" style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}>
        <div className="w-12 h-12 rounded-full flex items-center justify-center flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)' }}>
          <span className="text-white text-[15px] font-bold tracking-tight">{initialsOf(auth.email)}</span>
        </div>
        <div className="min-w-0">
          <p className="text-[15px] font-semibold truncate" style={{ color: 'var(--t-tx1)' }}>{auth.email}</p>
          <p className="text-[12.5px] mt-0.5" style={{ color: 'var(--t-tx5)' }}>
            {profile ? `${profile.role.charAt(0).toUpperCase()}${profile.role.slice(1)}` : 'MAGIK account'}
            {memberSince ? ` · Member since ${memberSince}` : ''}
          </p>
        </div>
      </div>

      <Card title="Appearance">
        <Row title="Theme" desc={dark ? 'Dark mode is currently on' : 'Light mode is currently on'}>
          <GhostButton onClick={onToggleTheme} icon={dark ? Moon : Sun}>{dark ? 'Dark' : 'Light'}</GhostButton>
        </Row>
      </Card>

      <Card title="Chat display">
        <Row title="Source citations" desc="Show the document chips MAGIK cites underneath each reply">
          <Toggle checked={showSources} onChange={setShowSources} />
        </Row>
      </Card>

      <Card title="Session">
        <Row title="Conversation memory" desc="Reset what MAGIK remembers from this conversation — your knowledge base stays intact">
          <GhostButton onClick={handleClearMemory} disabled={clearing} spinning={clearing} icon={RotateCcw}>
            Clear memory
          </GhostButton>
        </Row>
        <Row title="Chat history" desc="Permanently delete all conversations from Recents — cannot be undone">
          <GhostButton
            onClick={handleClearAllHistory}
            disabled={clearingHistory}
            spinning={clearingHistory}
            icon={Trash2}
            style={{ color: confirmClearHistory ? 'var(--t-danger)' : undefined, borderColor: confirmClearHistory ? 'var(--t-danger)' : undefined }}
          >
            {confirmClearHistory ? 'Confirm delete' : 'Clear all history'}
          </GhostButton>
        </Row>
      </Card>
    </div>
  )
}

/* ── Knowledge base ────────────────────────────────── */
function KBSection({ auth, kbFiles, setKbFiles, addToast }) {
  const [deleting, setDeleting] = useState(null)
  const [confirmDeleteAll, setConfirmDeleteAll] = useState(false)
  const [deletingAll, setDeletingAll] = useState(false)

  const handleDelete = async (filename) => {
    setDeleting(filename)
    try {
      await deleteKBFile(auth.token, filename)
      setKbFiles(prev => prev.filter(f => f.filename !== filename))
      addToast(`Removed ${filename}`, 'success')
    } catch (err) {
      addToast(err.message, 'error')
    } finally {
      setDeleting(null)
    }
  }

  const handleDeleteAll = async () => {
    if (!confirmDeleteAll) { setConfirmDeleteAll(true); return }
    setConfirmDeleteAll(false)
    setDeletingAll(true)
    const targets = kbFiles.map(f => f.filename)
    let failed = 0
    for (const filename of targets) {
      try {
        await deleteKBFile(auth.token, filename)
        setKbFiles(prev => prev.filter(f => f.filename !== filename))
      } catch {
        failed++
      }
    }
    setDeletingAll(false)
    if (failed === 0) {
      addToast(`Deleted all ${targets.length} file${targets.length !== 1 ? 's' : ''}`, 'success')
    } else {
      addToast(`Deleted ${targets.length - failed} of ${targets.length} files — ${failed} failed`, 'error')
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-[13px]" style={{ color: 'var(--t-tx5)' }}>
          {kbFiles.length} file{kbFiles.length !== 1 ? 's' : ''} indexed for retrieval. Add more from the sidebar's upload zone.
        </p>
        {kbFiles.length > 0 && (
          <GhostButton
            onClick={handleDeleteAll}
            disabled={deletingAll}
            spinning={deletingAll}
            icon={Trash2}
            style={{ color: confirmDeleteAll ? 'var(--t-danger)' : undefined, borderColor: confirmDeleteAll ? 'var(--t-danger)' : undefined }}
          >
            {confirmDeleteAll ? 'Confirm delete all' : 'Delete all'}
          </GhostButton>
        )}
      </div>

      {kbFiles.length === 0 ? (
        <div className="rounded-2xl py-10 text-center" style={{ background: 'var(--t-card)', border: '1px dashed var(--t-bd3)' }}>
          <FileText size={22} className="mx-auto mb-2" style={{ color: 'var(--t-tx6)' }} />
          <p className="text-sm" style={{ color: 'var(--t-tx5)' }}>Your knowledge base is empty.</p>
        </div>
      ) : (
        <div className="rounded-2xl overflow-hidden" style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}>
          {kbFiles.map((f, i) => {
            const ext = (f.filename.split('.').pop() || '').toUpperCase()
            return (
              <div key={f.filename} className="flex items-center gap-3 px-4 py-3"
                style={{ borderTop: i > 0 ? '1px solid var(--t-bd1)' : 'none' }}>
                <FileText size={16} className="flex-shrink-0" style={{ color: 'var(--t-tx5)' }} />
                <span className="flex-1 min-w-0 truncate text-sm" style={{ color: 'var(--t-tx2)' }}>{f.filename}</span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded flex-shrink-0"
                  style={{ background: 'var(--t-chp)', border: '1px solid var(--t-chpb)', color: 'var(--t-tx5)' }}>
                  {ext || '?'}
                </span>
                <button
                  onClick={() => handleDelete(f.filename)}
                  disabled={deleting === f.filename}
                  className="w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0 transition-colors disabled:opacity-50"
                  style={{ color: 'var(--t-tx5)' }}
                  onMouseEnter={e => { e.currentTarget.style.color = 'var(--t-danger)'; e.currentTarget.style.background = 'var(--t-hov2)' }}
                  onMouseLeave={e => { e.currentTarget.style.color = 'var(--t-tx5)'; e.currentTarget.style.background = 'transparent' }}
                  title="Remove from knowledge base"
                >
                  {deleting === f.filename ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />}
                </button>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

/* ── Security ──────────────────────────────────────── */
function SecuritySection({ auth, onLogout, addToast }) {
  const [current, setCurrent]       = useState('')
  const [next, setNext]             = useState('')
  const [confirm, setConfirm]       = useState('')
  const [reveal, setReveal]         = useState(false)
  const [saving, setSaving]         = useState(false)
  const [signingOut, setSigningOut] = useState(false)

  const submitPassword = async (e) => {
    e.preventDefault()
    if (next !== confirm) { addToast('New passwords do not match', 'error'); return }
    if (next.length < 8)  { addToast('New password must be at least 8 characters', 'error'); return }
    setSaving(true)
    try {
      await changePassword(auth.token, current, next)
      addToast('Password changed — signing you out everywhere…', 'success')
      setTimeout(onLogout, 1200)
    } catch (err) {
      addToast(err.message, 'error')
      setSaving(false)
    }
  }

  const handleSignOutAll = async () => {
    setSigningOut(true)
    try {
      await logoutAll(auth.token)
      addToast('All sessions revoked — signing you out…', 'success')
      setTimeout(onLogout, 1000)
    } catch (err) {
      addToast(err.message, 'error')
      setSigningOut(false)
    }
  }

  return (
    <div className="space-y-5">
      <Card title="Change password">
        <form onSubmit={submitPassword} className="py-4 space-y-3">
          <input type={reveal ? 'text' : 'password'} placeholder="Current password" value={current}
            onChange={e => setCurrent(e.target.value)} required autoComplete="current-password"
            className={inputCls} style={inputStyle} />
          <input type={reveal ? 'text' : 'password'} placeholder="New password" value={next}
            onChange={e => setNext(e.target.value)} required autoComplete="new-password"
            className={inputCls} style={inputStyle} />
          <input type={reveal ? 'text' : 'password'} placeholder="Confirm new password" value={confirm}
            onChange={e => setConfirm(e.target.value)} required autoComplete="new-password"
            className={inputCls} style={inputStyle} />
          <label className="flex items-center gap-2 text-[12.5px] cursor-pointer select-none" style={{ color: 'var(--t-tx5)' }}>
            <input type="checkbox" checked={reveal} onChange={e => setReveal(e.target.checked)} className="accent-violet-500" />
            Show passwords
          </label>
          <div className="flex items-center gap-3 pt-1">
            <button type="submit" disabled={saving}
              className="flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium text-white transition-opacity disabled:opacity-60"
              style={{ background: 'linear-gradient(135deg, #8b5cf6, #3b82f6)' }}>
              {saving && <Loader2 size={14} className="animate-spin" />}
              Update password
            </button>
            <p className="text-[12px] flex-1" style={{ color: 'var(--t-tx5)' }}>
              You'll be signed out everywhere and need to log back in.
            </p>
          </div>
        </form>
      </Card>

      <Card title="Sessions">
        <Row title="Sign out of all devices" desc="Immediately revoke every active session for this account, including this one">
          <GhostButton onClick={handleSignOutAll} disabled={signingOut} spinning={signingOut} icon={LogOut}>
            Sign out everywhere
          </GhostButton>
        </Row>
      </Card>
    </div>
  )
}

/* ── Privacy & data ────────────────────────────────── */
function PrivacySection({ auth, onLogout, addToast }) {
  const [confirmText, setConfirmText] = useState('')
  const [deleting, setDeleting]       = useState(false)
  const ready = confirmText.trim().toLowerCase() === 'delete my account'

  const handleDelete = async () => {
    if (!ready || deleting) return
    setDeleting(true)
    try {
      await deleteAccount(auth.token)
      localStorage.removeItem('magik_device_token')
      addToast('Your account and all data have been permanently deleted', 'success')
      setTimeout(onLogout, 1000)
    } catch (err) {
      addToast(err.message, 'error')
      setDeleting(false)
    }
  }

  return (
    <div className="space-y-5">
      <div className="rounded-2xl p-5" style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)' }}>
        <h3 className="text-[14px] font-semibold mb-1.5" style={{ color: 'var(--t-tx1)' }}>Your data</h3>
        <p className="text-[13px] leading-relaxed" style={{ color: 'var(--t-tx4)' }}>
          MAGIK stores your uploaded files, conversation memory and account profile in
          Qdrant, Redis and MongoDB — scoped to your account and isolated from other tenants.
          Nothing is shared with third parties.
        </p>
      </div>

      <div className="rounded-2xl p-5" style={{ background: 'var(--t-err-bg)', border: '1px solid var(--t-err-bd)' }}>
        <div className="flex items-start gap-3 mb-4">
          <AlertTriangle size={18} className="flex-shrink-0 mt-0.5" style={{ color: 'var(--t-err-tx)' }} />
          <div>
            <h3 className="text-[14px] font-semibold" style={{ color: 'var(--t-err-tx)' }}>Delete account</h3>
            <p className="text-[12.5px] mt-1 leading-relaxed" style={{ color: 'var(--t-err-tx)', opacity: 0.85 }}>
              Permanently purges your knowledge base, conversation history and profile from every
              store, then deactivates your account. This cannot be undone.
            </p>
          </div>
        </div>
        <p className="text-[12.5px] mb-2" style={{ color: 'var(--t-err-tx)', opacity: 0.85 }}>
          Type <span className="font-mono font-semibold">delete my account</span> to confirm
        </p>
        <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5">
          <input
            value={confirmText}
            onChange={e => setConfirmText(e.target.value)}
            placeholder="delete my account"
            className="flex-1 rounded-xl px-4 py-2.5 text-sm outline-none t-focus"
            style={{ background: 'var(--t-inp)', border: '1px solid var(--t-err-bd)', color: 'var(--t-tx1)' }}
          />
          <button
            onClick={handleDelete}
            disabled={!ready || deleting}
            className="flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-medium text-white transition-opacity disabled:opacity-50 sm:flex-shrink-0"
            style={{ background: 'var(--t-danger)' }}
          >
            {deleting && <Loader2 size={14} className="animate-spin" />}
            Delete permanently
          </button>
        </div>
      </div>
    </div>
  )
}

/* ── Shortcuts ─────────────────────────────────────── */
function ShortcutsSection() {
  const kbdStyle = {
    background: 'var(--t-hov2)',
    color: 'var(--t-tx3)',
    border: '1px solid var(--t-bd3)',
  }
  const Keys = ({ keys }) => (
    <div className="flex items-center gap-1 flex-shrink-0">
      {keys.map(k => (
        <kbd key={k} className="text-[11px] px-1.5 py-0.5 rounded font-mono min-w-[22px] text-center" style={kbdStyle}>{k}</kbd>
      ))}
    </div>
  )

  return (
    <div className="space-y-5">
      <Card title="Chat">
        <Row title="Focus input" desc="Jump cursor to the chat box"><Keys keys={['⌘', 'K']} /></Row>
        <Row title="Send message" desc="Submit your typed message"><Keys keys={['Enter']} /></Row>
        <Row title="New line" desc="Insert a line break without sending"><Keys keys={['⇧', 'Enter']} /></Row>
        <Row title="New chat" desc="Start a fresh conversation"><Keys keys={['⌘', '⇧', 'N']} /></Row>
        <Row title="Scope to file" desc="Type @ to pick a specific KB file for your query"><Keys keys={['@']} /></Row>
      </Card>
      <Card title="Interface">
        <Row title="Collapse sidebar" desc="Close or collapse the left sidebar"><Keys keys={['Esc']} /></Row>
      </Card>
      <p className="text-[12px] px-1" style={{ color: 'var(--t-tx6)' }}>
        ⌘ = Cmd on Mac · Ctrl on Windows/Linux
      </p>
    </div>
  )
}

/* ── Modal shell ───────────────────────────────────── */
export default function SettingsModal({
  isOpen, onClose, auth, onLogout,
  dark, onToggleTheme,
  kbFiles, setKbFiles,
  sessionId,
  showSources, setShowSources,
  initialSection = 'account',
  onClearConversation,
  onClearAllHistory,
}) {
  const [section, setSection]       = useState('account')
  const [mobileView, setMobileView] = useState('nav')  // 'nav' | 'content'
  const isMobile  = useIsMobile()
  const { addToast } = useToast()
  const onCloseRef = useRef(onClose)
  useEffect(() => { onCloseRef.current = onClose })

  useEffect(() => {
    if (!isOpen) return
    setSection(initialSection)
    setMobileView(initialSection !== 'account' ? 'content' : 'nav')
    const onEsc = e => { if (e.key === 'Escape') onCloseRef.current() }
    document.addEventListener('keydown', onEsc)
    return () => document.removeEventListener('keydown', onEsc)
  }, [isOpen])

  if (!isOpen) return null

  const selectSection = (id) => {
    setSection(id)
    if (isMobile) setMobileView('content')
  }

  const sectionContent = (
    <>
      {section === 'account' && (
        <AccountSection
          auth={auth} dark={dark} onToggleTheme={onToggleTheme}
          showSources={showSources} setShowSources={setShowSources}
          sessionId={sessionId} addToast={addToast}
          onClearConversation={onClearConversation}
          onClearAllHistory={onClearAllHistory}
        />
      )}
      {section === 'kb'        && <KBSection auth={auth} kbFiles={kbFiles} setKbFiles={setKbFiles} addToast={addToast} />}
      {section === 'security'  && <SecuritySection auth={auth} onLogout={onLogout} addToast={addToast} />}
      {section === 'privacy'   && <PrivacySection  auth={auth} onLogout={onLogout} addToast={addToast} />}
      {section === 'shortcuts' && <ShortcutsSection />}
    </>
  )

  return createPortal(
    <div
      onClick={onClose}
      className="fixed inset-0 z-[100] flex items-center justify-center p-3 sm:p-4"
      style={{ background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(2px)' }}
    >
      <div
        onClick={e => e.stopPropagation()}
        className="w-full max-w-3xl rounded-2xl overflow-hidden shadow-2xl"
        style={{
          height: 'min(640px, 92vh)',
          background: 'var(--t-sur)',
          border: '1px solid var(--t-bd2)',
          display: 'flex',
          flexDirection: isMobile ? 'column' : 'row',
        }}
      >
        {/* ── MOBILE: nav list (full-width, shown when mobileView==='nav') ── */}
        {isMobile && mobileView === 'nav' && (
          <div className="flex flex-col h-full" style={{ background: 'var(--t-sb)' }}>
            <div className="flex items-center justify-between px-5 pt-5 pb-3 flex-shrink-0">
              <h2 className="text-[17px] font-semibold" style={{ color: 'var(--t-tx1)' }}>Settings</h2>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ color: 'var(--t-tx5)' }}
              >
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-3 pb-4">
              {NAV.map(({ id, label, Icon }) => (
                <button
                  key={id}
                  onClick={() => selectSection(id)}
                  className="w-full flex items-center justify-between px-4 py-3.5 rounded-xl text-[15px] text-left transition-colors mb-1"
                  style={{ background: 'var(--t-card)', border: '1px solid var(--t-bd2)', color: 'var(--t-tx2)' }}
                >
                  <span className="flex items-center gap-3">
                    <Icon size={17} style={{ color: 'var(--t-tx5)' }} />
                    {label}
                  </span>
                  <ChevronRight size={15} style={{ color: 'var(--t-tx5)' }} />
                </button>
              ))}
            </div>
          </div>
        )}

        {/* ── MOBILE: section content (full-width, shown when mobileView==='content') ── */}
        {isMobile && mobileView === 'content' && (
          <div className="flex flex-col h-full min-w-0">
            <div className="flex items-center justify-between px-4 pt-4 pb-2 flex-shrink-0">
              <button
                onClick={() => setMobileView('nav')}
                className="flex items-center gap-1 text-[13px] py-1 pr-2 rounded-lg"
                style={{ color: 'var(--t-accent)' }}
              >
                <ChevronLeft size={16} />
                Settings
              </button>
              <h3 className="text-[15px] font-semibold" style={{ color: 'var(--t-tx1)' }}>
                {NAV.find(n => n.id === section)?.label}
              </h3>
              <button
                onClick={onClose}
                className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ color: 'var(--t-tx5)' }}
              >
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto px-4 pb-6 pt-1">
              {sectionContent}
            </div>
          </div>
        )}

        {/* ── DESKTOP: side-by-side (unchanged) ── */}
        {!isMobile && (
          <>
            {/* Left nav */}
            <div className="w-56 flex-shrink-0 flex flex-col py-5 px-3 overflow-y-auto"
              style={{ background: 'var(--t-sb)', borderRight: '1px solid var(--t-bd1)' }}>
              <h2 className="text-[15px] font-semibold px-3 mb-4" style={{ color: 'var(--t-tx1)' }}>Settings</h2>
              {NAV.map(({ id, label, Icon }) => (
                <button
                  key={id}
                  onClick={() => selectSection(id)}
                  className="flex items-center gap-2.5 px-3 py-2.5 rounded-xl text-[13.5px] text-left transition-colors mb-0.5"
                  style={{
                    background: section === id ? 'var(--t-hov)' : 'transparent',
                    color:      section === id ? 'var(--t-tx1)' : 'var(--t-tx4)',
                  }}
                  onMouseEnter={e => { if (section !== id) e.currentTarget.style.background = 'var(--t-hov)' }}
                  onMouseLeave={e => { if (section !== id) e.currentTarget.style.background = 'transparent' }}
                >
                  <Icon size={15} />
                  {label}
                </button>
              ))}
            </div>

            {/* Content */}
            <div className="flex-1 min-w-0 flex flex-col">
              <div className="flex items-center justify-between px-7 pt-5 pb-2 flex-shrink-0">
                <h3 className="text-[17px] font-semibold" style={{ color: 'var(--t-tx1)' }}>
                  {NAV.find(n => n.id === section)?.label}
                </h3>
                <button
                  onClick={onClose}
                  className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors"
                  style={{ color: 'var(--t-tx5)' }}
                  onMouseEnter={e => { e.currentTarget.style.background = 'var(--t-hov2)'; e.currentTarget.style.color = 'var(--t-tx2)' }}
                  onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--t-tx5)' }}
                  title="Close"
                >
                  <X size={16} />
                </button>
              </div>
              <div className="flex-1 overflow-y-auto px-7 pb-6 pt-2">
                {sectionContent}
              </div>
            </div>
          </>
        )}
      </div>
    </div>,
    document.body
  )
}
