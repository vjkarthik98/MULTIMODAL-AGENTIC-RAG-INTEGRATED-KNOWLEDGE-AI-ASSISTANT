export default function MagikIcon({ size = 24, strokeWidth = 2, className = '' }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24"
      fill="none" stroke="currentColor"
      strokeWidth={strokeWidth} strokeLinecap="round" strokeLinejoin="round"
      className={className}
    >
      {/* Left brain lobe */}
      <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.98-3 2.5 2.5 0 0 1-1.32-4.24 3 3 0 0 1 .34-5.58 2.5 2.5 0 0 1 2.98-3.19A2.5 2.5 0 0 1 9.5 2Z" />
      {/* Right brain lobe */}
      <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 1.98-3 2.5 2.5 0 0 0 1.32-4.24 3 3 0 0 0-.34-5.58 2.5 2.5 0 0 0-2.98-3.19A2.5 2.5 0 0 0 14.5 2Z" />
      {/* Neural nodes */}
      <circle cx="9" cy="8.5" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="15" cy="8.5" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="12" cy="13.5" r="1.2" fill="currentColor" stroke="none" />
      {/* Axonal connections */}
      <line x1="9" y1="8.5" x2="15" y2="8.5" />
      <line x1="9" y1="8.5" x2="12" y2="13.5" />
      <line x1="15" y1="8.5" x2="12" y2="13.5" />
    </svg>
  )
}
