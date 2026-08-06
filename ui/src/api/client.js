const API = ''  // empty = same origin via Vite proxy

// Auth no longer travels through JS at all: the access/refresh/device tokens
// are httpOnly cookies set by the server (app/auth/cookies.py) and are never
// readable by this code, localStorage, or a URL. `credentials: 'include'`
// makes the browser attach them automatically on every request.
//
// State-changing requests still carry a `csrf` value (the double-submit CSRF
// token — see app/api/middleware.py::CSRFMiddleware) as the `X-CSRF-Token`
// header. Unlike the old bearer token, this value is NOT a secret: on its own
// it grants no access, so it being visible in devtools/React state is fine —
// that's the point of the double-submit pattern.
function csrfHeaders(csrf, extra = {}) {
  const h = { ...extra }
  if (csrf) h['X-CSRF-Token'] = csrf
  return h
}

// Reads the JS-readable CSRF cookie the server sets alongside the httpOnly
// session cookies. Called after login/refresh/OAuth to populate auth state.
export function readCsrfCookie() {
  const match = document.cookie.match(/(?:^|; )magik_csrf=([^;]*)/)
  return match ? decodeURIComponent(match[1]) : ''
}

export async function login(email, password) {
  // No device_token in the body — the trusted-device cookie (if any) rides
  // along automatically via credentials:'include'.
  const res = await fetch(`${API}/auth/login`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Login failed (${res.status})`)
  }
  return res.json()  // { otp_required, otp_token } or { status: 'ok' } (cookies set)
}

export async function verifyOtp(otpToken, code) {
  const res = await fetch(`${API}/auth/verify-otp`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ otp_token: otpToken, code }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Verification failed (${res.status})`)
  }
  return res.json()
}

export async function resendOtp(otpToken) {
  const res = await fetch(`${API}/auth/resend-otp`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ otp_token: otpToken }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    const error = new Error(err.detail || `Resend failed (${res.status})`)
    error.status = res.status
    throw error
  }
  return res.json()  // { sent: true, cooldown_seconds }
}

export async function forgotPassword(email) {
  const res = await fetch(`${API}/auth/forgot-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Request failed (${res.status})`)
  }
  return res.json()
}

export async function resetPassword(token, newPassword) {
  const res = await fetch(`${API}/auth/reset-password`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token, new_password: newPassword }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Reset failed (${res.status})`)
  }
  return res.json()
}

export async function register(email, password) {
  const res = await fetch(`${API}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Registration failed (${res.status})`)
  }
  return res.json()  // { otp_required: true, otp_token: "..." }
}

// GET — cookie-authenticated, no CSRF needed (safe method).
export async function getMe() {
  const res = await fetch(`${API}/auth/me`, { credentials: 'include' })
  if (!res.ok) throw new Error('Session expired')
  return res.json()  // { user_id, email, ... }
}

// Exchanges the httpOnly refresh cookie for a fresh access + refresh pair —
// keeps the user signed in past the short access-token lifetime without
// re-entering credentials. No body: the server reads magik_refresh itself.
export async function refreshAccessToken() {
  const res = await fetch(`${API}/auth/refresh`, {
    method: 'POST',
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  })
  if (!res.ok) {
    // Tag with the HTTP status so callers can tell "server rejected this token"
    // (401 — genuinely expired/revoked/already-rotated) apart from a transient
    // 5xx/network hiccup, where the refresh token itself is still good and a
    // retry should NOT log the user out.
    const err = new Error('Refresh token expired')
    err.status = res.status
    throw err
  }
  return res.json()  // { access_token, refresh_token, token_type, expires_in } — cookies also set
}

export async function logout(csrf) {
  try {
    await fetch(`${API}/auth/logout`, {
      method: 'POST',
      credentials: 'include',
      headers: csrfHeaders(csrf, { 'Content-Type': 'application/json' }),
      body: '{}',
    })
  } catch {
    // best-effort server-side revocation — local sign-out proceeds regardless
  }
}

export async function listKB() {
  const res = await fetch(`${API}/rag/knowledge-base`, { credentials: 'include' })
  if (!res.ok) throw new Error('Failed to load knowledge base')
  const data = await res.json()  // { user_id, file_count, files: [...] }
  return Array.isArray(data) ? data : (data.files || [])
}

export async function deleteKBFile(csrf, filename) {
  const res = await fetch(`${API}/rag/knowledge-base/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    credentials: 'include',
    headers: csrfHeaders(csrf),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Delete failed')
  }
  return res.json()
}

export function ingestFile(csrf, file, sessionId = 'default', abortController = null, onProgress = null) {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)
    form.append('session_id', sessionId)
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API}/rag/ingest`)
    xhr.withCredentials = true
    if (csrf) xhr.setRequestHeader('X-CSRF-Token', csrf)
    if (abortController) {
      abortController.signal.addEventListener('abort', () => xhr.abort(), { once: true })
    }
    if (onProgress) {
      // Real network-transfer progress (bytes actually sent), not a timer —
      // lengthComputable is false only for chunked/unknown-size bodies, which
      // FormData file uploads never are.
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress(e.loaded / e.total)
      }
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText))
      } else {
        const err = (() => { try { return JSON.parse(xhr.responseText) } catch { return {} } })()
        reject(new Error(err.detail || `Upload failed (${xhr.status})`))
      }
    }
    xhr.onerror = () => reject(new Error('Network error — upload failed'))
    xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'))
    xhr.ontimeout = () => reject(new Error('Upload timed out'))
    xhr.timeout = 300000 // 5 min for large files
    xhr.send(form)
  })
}

// Returns the raw fetch Response so caller can stream the body
export function streamQuery(csrf, query, sessionId, signal, noCache = false, sources = null, forceWeb = false, regenerate = false) {
  const body = { query, session_id: sessionId, no_cache: noCache }
  // Distinct from no_cache: no_cache only skips the stored answer, which on a
  // deterministic pipeline still recomputes the identical text. regenerate is
  // what makes the retry actually different (app/llm/regeneration.py).
  if (regenerate) body.regenerate = true
  if (sources && sources.length) body.sources = sources
  if (forceWeb) body.force_web = true
  return fetch(`${API}/rag/query/stream`, {
    method: 'POST',
    credentials: 'include',
    headers: csrfHeaders(csrf, { 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
    signal,
  })
}

export async function changePassword(csrf, currentPassword, newPassword) {
  const res = await fetch(`${API}/auth/password`, {
    method: 'POST',
    credentials: 'include',
    headers: csrfHeaders(csrf, { 'Content-Type': 'application/json' }),
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Password change failed (${res.status})`)
  }
  return res.json()
}

export async function logoutAll(csrf) {
  const res = await fetch(`${API}/auth/logout-all`, {
    method: 'POST',
    credentials: 'include',
    headers: csrfHeaders(csrf),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to sign out other sessions (${res.status})`)
  }
  return res.json()
}

export async function deleteAccount(csrf) {
  const res = await fetch(`${API}/auth/me`, {
    method: 'DELETE',
    credentials: 'include',
    headers: csrfHeaders(csrf),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Account deletion failed (${res.status})`)
  }
  return res.json()
}

export async function clearMemory(csrf, sessionId) {
  const res = await fetch(`${API}/rag/memory/clear`, {
    method: 'POST',
    credentials: 'include',
    headers: csrfHeaders(csrf, { 'Content-Type': 'application/json' }),
    body: JSON.stringify({ session_id: sessionId }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to clear memory (${res.status})`)
  }
  return res.json()
}

export async function listChatSessions() {
  const res = await fetch(`${API}/rag/sessions`, { credentials: 'include' })
  if (!res.ok) throw new Error('Failed to load chat history')
  const data = await res.json()
  return data.sessions || []
}

export async function getChatSession(sessionId) {
  const res = await fetch(`${API}/rag/sessions/${encodeURIComponent(sessionId)}`, { credentials: 'include' })
  if (!res.ok) {
    if (res.status === 404) return null
    throw new Error('Failed to load chat')
  }
  return res.json()  // { session_id, title, messages: [{ role, content, timestamp }] }
}

export async function updateChatSession(csrf, sessionId, fields) {
  const res = await fetch(`${API}/rag/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    credentials: 'include',
    headers: csrfHeaders(csrf, { 'Content-Type': 'application/json' }),
    body: JSON.stringify(fields),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to update chat')
  }
  return res.json()
}

export async function deleteChatSession(csrf, sessionId) {
  const res = await fetch(`${API}/rag/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    credentials: 'include',
    headers: csrfHeaders(csrf),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to delete chat')
  }
  return res.json()
}

// Called after streaming to get sources + metadata (hits Redis cache)
export async function queryMeta(csrf, query, sessionId, noCache = false, sources = null, regenerate = false, forceWeb = false) {
  const body = { query, session_id: sessionId, no_cache: noCache }
  if (regenerate) body.regenerate = true
  // Must travel with the request: this call is the fallback for a failed
  // stream, and without it a web-mode question gets answered from the KB.
  if (forceWeb) body.force_web = true
  if (sources && sources.length) body.sources = sources
  const res = await fetch(`${API}/rag/query`, {
    method: 'POST',
    credentials: 'include',
    headers: csrfHeaders(csrf, { 'Content-Type': 'application/json' }),
    body: JSON.stringify(body),
  })
  if (!res.ok) return null
  return res.json()
  // { answer, sources, confidence, decision, latency, ... }
}

export async function deleteAllChatSessions(csrf) {
  const res = await fetch(`${API}/rag/sessions`, {
    method: 'DELETE',
    credentials: 'include',
    headers: csrfHeaders(csrf),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to clear history')
  }
  return res.json()
}

export async function submitFeedback(csrf, sessionId, vote, messageId, query, responseSnippet) {
  try {
    await fetch(`${API}/rag/feedback`, {
      method: 'POST',
      credentials: 'include',
      headers: csrfHeaders(csrf, { 'Content-Type': 'application/json' }),
      body: JSON.stringify({
        session_id: sessionId,
        vote,
        message_id: messageId || null,
        query: query || null,
        response_snippet: responseSnippet ? responseSnippet.slice(0, 500) : null,
      }),
    })
  } catch (_) {}  // fire-and-forget — feedback failure must never interrupt the user
}

// Overwrite the last assistant message in the session so reload = what user saw.
export async function patchLastMessage(csrf, sessionId, content, sources, msgId = null) {
  try {
    await fetch(`${API}/rag/sessions/${encodeURIComponent(sessionId)}/last-message`, {
      method: 'PATCH',
      credentials: 'include',
      headers: csrfHeaders(csrf, { 'Content-Type': 'application/json' }),
      body: JSON.stringify({ content, sources: sources || [], msg_id: msgId }),
    })
  } catch (_) {}  // fire-and-forget — failure is non-critical
}

// ── Phase 8/9 — New endpoints ──────────────────────────────────────────────

export async function getIngestionStatus(jobId) {
  const res = await fetch(`${API}/rag/ingestion/status/${encodeURIComponent(jobId)}`, {
    credentials: 'include',
  })
  if (!res.ok) {
    const err = new Error(`Status check failed (${res.status})`)
    err.status = res.status
    throw err
  }
  return res.json()  // IngestJob: { job_id, filename, modality, status, progress, chunks_done, chunks_total }
}

export async function listKBFiles() {
  const res = await fetch(`${API}/api/kb/files`, { credentials: 'include' })
  if (!res.ok) throw new Error(`KB list failed (${res.status})`)
  return res.json()  // { files: [{ file_hash, filename, modality, chunk_count, ingested_at }] }
}

export async function deleteKBFileByHash(csrf, fileHash) {
  const res = await fetch(`${API}/api/kb/files/${encodeURIComponent(fileHash)}`, {
    method: 'DELETE',
    credentials: 'include',
    headers: csrfHeaders(csrf),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Delete failed (${res.status})`)
  }
  return res.json()
}

export async function getTranscript(fileHash) {
  const res = await fetch(`${API}/api/transcript/${encodeURIComponent(fileHash)}`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error(`Transcript fetch failed (${res.status})`)
  return res.json()  // { chunks: [{ start_timestamp, end_timestamp, speaker_name, speaker_role, call_section, transcript }] }
}

export async function getChunkMeta(chunkId) {
  const res = await fetch(`${API}/api/sources/${encodeURIComponent(chunkId)}`, {
    credentials: 'include',
  })
  if (!res.ok) throw new Error(`Chunk metadata fetch failed (${res.status})`)
  return res.json()
}

