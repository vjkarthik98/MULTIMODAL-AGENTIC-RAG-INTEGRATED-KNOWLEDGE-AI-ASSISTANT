import { useToast } from '../context/ToastContext'
import { CheckCircle, AlertCircle, Info, X } from 'lucide-react'

const ICONS = {
  success: <CheckCircle size={15} className="flex-shrink-0" style={{ color: 'var(--t-success)' }} />,
  error:   <AlertCircle  size={15} className="flex-shrink-0" style={{ color: 'var(--t-danger)' }} />,
  info:    <Info         size={15} className="flex-shrink-0" style={{ color: 'var(--t-accent)' }} />,
}

export default function Toast() {
  const { toasts } = useToast()

  if (!toasts.length) return null

  return (
    <div className="fixed top-6 right-6 z-[9999] flex flex-col gap-2 pointer-events-none">
      {toasts.map(t => (
        <div
          key={t.id}
          className={`pointer-events-auto flex items-center gap-2.5 px-4 py-3 rounded-xl shadow-xl text-sm max-w-[320px]
            ${t.exiting ? 'toast-exit' : 'toast-enter'}`}
          style={{
            background: 'var(--t-card)',
            border: '1px solid var(--t-bd3)',
            color: 'var(--t-tx2)',
            backdropFilter: 'blur(8px)',
          }}
        >
          {ICONS[t.type] || ICONS.info}
          <span className="flex-1 leading-snug">{t.message}</span>
        </div>
      ))}
    </div>
  )
}
