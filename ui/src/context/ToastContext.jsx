import { createContext, useContext, useState, useCallback } from 'react'

const ToastContext = createContext({ addToast: () => {} })

export function ToastProvider({ children }) {
  const [toasts, setToasts] = useState([])

  const addToast = useCallback((message, type = 'info') => {
    const id = Date.now() + Math.random()
    setToasts(prev => [...prev, { id, message, type, exiting: false }])
    setTimeout(() => {
      setToasts(prev => prev.map(t => t.id === id ? { ...t, exiting: true } : t))
      setTimeout(() => setToasts(prev => prev.filter(t => t.id !== id)), 220)
    }, 2800)
  }, [])

  return (
    <ToastContext.Provider value={{ addToast, toasts }}>
      {children}
    </ToastContext.Provider>
  )
}

export const useToast = () => useContext(ToastContext)
