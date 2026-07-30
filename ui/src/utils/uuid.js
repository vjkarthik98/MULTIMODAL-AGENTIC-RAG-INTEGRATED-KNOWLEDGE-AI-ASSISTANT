/**
 * UUID v4 that works outside a secure context.
 *
 * `crypto.randomUUID()` is only exposed on HTTPS origins and localhost. The
 * deployed demo is served over plain http:// on an Elastic IP, where it is
 * simply undefined — calling it threw
 *
 *     TypeError: crypto.randomUUID is not a function
 *
 * during ChatPage's very first useState initialiser, which took down the whole
 * app behind the error boundary before a single request was made. Development
 * never caught it because `npm run dev` serves on localhost, which browsers
 * treat as a secure context.
 *
 * Note `crypto.getRandomValues()` has no such restriction, so the primary
 * fallback is still cryptographically sound; the Math.random() branch exists
 * only for environments with no Web Crypto at all.
 */
export function uuid() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const b = new Uint8Array(16)
    crypto.getRandomValues(b)
    b[6] = (b[6] & 0x0f) | 0x40 // version 4
    b[8] = (b[8] & 0x3f) | 0x80 // variant 10xx
    const h = Array.from(b, (x) => x.toString(16).padStart(2, '0'))
    return (
      h.slice(0, 4).join('') +
      '-' + h.slice(4, 6).join('') +
      '-' + h.slice(6, 8).join('') +
      '-' + h.slice(8, 10).join('') +
      '-' + h.slice(10, 16).join('')
    )
  }

  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}
