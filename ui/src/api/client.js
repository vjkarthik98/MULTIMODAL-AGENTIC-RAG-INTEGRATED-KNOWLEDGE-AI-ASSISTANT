const API = ''  // empty = same origin via Vite proxy

function bearer(token) {
  return { Authorization: `Bearer ${token}` }
}

export async function login(email, password) {
  const device_token = localStorage.getItem('magik_device_token') || undefined
  const res = await fetch(`${API}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password, device_token }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Login failed (${res.status})`)
  }
  return res.json()  // { access_token, refresh_token, device_token? } or { otp_required, otp_token }
}

export async function verifyOtp(otpToken, code) {
  const res = await fetch(`${API}/auth/verify-otp`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ otp_token: otpToken, code }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Verification failed (${res.status})`)
  }
  return res.json()  // { access_token, refresh_token, ... }
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

export async function getMe(token) {
  const res = await fetch(`${API}/auth/me`, { headers: bearer(token) })
  if (!res.ok) throw new Error('Session expired')
  return res.json()  // { user_id, email, ... }
}

// Exchanges a refresh token for a fresh access + refresh token pair — keeps the
// user signed in past the short access-token lifetime without re-entering credentials.
export async function refreshAccessToken(refreshToken) {
  const res = await fetch(`${API}/auth/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refreshToken }),
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
  return res.json()  // { access_token, refresh_token, token_type, expires_in }
}

export async function logout(token, refreshToken) {
  try {
    await fetch(`${API}/auth/logout`, {
      method: 'POST',
      headers: { ...bearer(token), 'Content-Type': 'application/json' },
      body: JSON.stringify({ refresh_token: refreshToken || null }),
    })
  } catch {
    // best-effort server-side revocation — local sign-out proceeds regardless
  }
}

export async function listKB(token) {
  const res = await fetch(`${API}/rag/knowledge-base`, { headers: bearer(token) })
  if (!res.ok) throw new Error('Failed to load knowledge base')
  const data = await res.json()  // { user_id, file_count, files: [...] }
  return Array.isArray(data) ? data : (data.files || [])
}

export async function deleteKBFile(token, filename) {
  const res = await fetch(`${API}/rag/knowledge-base/${encodeURIComponent(filename)}`, {
    method: 'DELETE',
    headers: bearer(token),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Delete failed')
  }
  return res.json()
}

export function ingestFile(token, file, sessionId = 'default', abortController = null, onProgress = null) {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    form.append('file', file)
    form.append('session_id', sessionId)
    const xhr = new XMLHttpRequest()
    xhr.open('POST', `${API}/rag/ingest`)
    xhr.setRequestHeader('Authorization', `Bearer ${token}`)
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
export function streamQuery(token, query, sessionId, signal, noCache = false, sources = null, forceWeb = false) {
  const body = { query, session_id: sessionId, no_cache: noCache }
  if (sources && sources.length) body.sources = sources
  if (forceWeb) body.force_web = true
  return fetch(`${API}/rag/query/stream`, {
    method: 'POST',
    headers: { ...bearer(token), 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}

export async function changePassword(token, currentPassword, newPassword) {
  const res = await fetch(`${API}/auth/password`, {
    method: 'POST',
    headers: { ...bearer(token), 'Content-Type': 'application/json' },
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Password change failed (${res.status})`)
  }
  return res.json()
}

export async function logoutAll(token) {
  const res = await fetch(`${API}/auth/logout-all`, {
    method: 'POST',
    headers: bearer(token),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to sign out other sessions (${res.status})`)
  }
  return res.json()
}

export async function deleteAccount(token) {
  const res = await fetch(`${API}/auth/me`, {
    method: 'DELETE',
    headers: bearer(token),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Account deletion failed (${res.status})`)
  }
  return res.json()
}

export async function clearMemory(token, sessionId) {
  const res = await fetch(`${API}/rag/memory/clear`, {
    method: 'POST',
    headers: { ...bearer(token), 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Failed to clear memory (${res.status})`)
  }
  return res.json()
}

export async function listChatSessions(token) {
  const res = await fetch(`${API}/rag/sessions`, { headers: bearer(token) })
  if (!res.ok) throw new Error('Failed to load chat history')
  const data = await res.json()
  return data.sessions || []
}

export async function getChatSession(token, sessionId) {
  const res = await fetch(`${API}/rag/sessions/${encodeURIComponent(sessionId)}`, { headers: bearer(token) })
  if (!res.ok) {
    if (res.status === 404) return null
    throw new Error('Failed to load chat')
  }
  return res.json()  // { session_id, title, messages: [{ role, content, timestamp }] }
}

export async function updateChatSession(token, sessionId, fields) {
  const res = await fetch(`${API}/rag/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    headers: { ...bearer(token), 'Content-Type': 'application/json' },
    body: JSON.stringify(fields),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to update chat')
  }
  return res.json()
}

export async function deleteChatSession(token, sessionId) {
  const res = await fetch(`${API}/rag/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'DELETE',
    headers: bearer(token),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to delete chat')
  }
  return res.json()
}

// Called after streaming to get sources + metadata (hits Redis cache)
export async function queryMeta(token, query, sessionId, noCache = false, sources = null) {
  const body = { query, session_id: sessionId, no_cache: noCache }
  if (sources && sources.length) body.sources = sources
  const res = await fetch(`${API}/rag/query`, {
    method: 'POST',
    headers: { ...bearer(token), 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) return null
  return res.json()
  // { answer, sources, confidence, decision, latency, ... }
}

export async function deleteAllChatSessions(token) {
  const res = await fetch(`${API}/rag/sessions`, {
    method: 'DELETE',
    headers: bearer(token),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || 'Failed to clear history')
  }
  return res.json()
}

export async function submitFeedback(token, sessionId, vote, messageId, query, responseSnippet) {
  try {
    await fetch(`${API}/rag/feedback`, {
      method: 'POST',
      headers: { ...bearer(token), 'Content-Type': 'application/json' },
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
export async function patchLastMessage(token, sessionId, content, sources, msgId = null) {
  try {
    await fetch(`${API}/rag/sessions/${encodeURIComponent(sessionId)}/last-message`, {
      method: 'PATCH',
      headers: { ...bearer(token), 'Content-Type': 'application/json' },
      body: JSON.stringify({ content, sources: sources || [], msg_id: msgId }),
    })
  } catch (_) {}  // fire-and-forget — failure is non-critical
}

// ── Phase 8/9 — New endpoints ──────────────────────────────────────────────

export async function getIngestionStatus(token, jobId) {
  const res = await fetch(`${API}/rag/ingestion/status/${encodeURIComponent(jobId)}`, {
    headers: bearer(token),
  })
  if (!res.ok) {
    const err = new Error(`Status check failed (${res.status})`)
    err.status = res.status
    throw err
  }
  return res.json()  // IngestJob: { job_id, filename, modality, status, progress, chunks_done, chunks_total }
}

export async function listKBFiles(token) {
  const res = await fetch(`${API}/api/kb/files`, { headers: bearer(token) })
  if (!res.ok) throw new Error(`KB list failed (${res.status})`)
  return res.json()  // { files: [{ file_hash, filename, modality, chunk_count, ingested_at }] }
}

export async function deleteKBFileByHash(token, fileHash) {
  const res = await fetch(`${API}/api/kb/files/${encodeURIComponent(fileHash)}`, {
    method: 'DELETE',
    headers: bearer(token),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Delete failed (${res.status})`)
  }
  return res.json()
}

export async function getTranscript(token, fileHash) {
  const res = await fetch(`${API}/api/transcript/${encodeURIComponent(fileHash)}`, {
    headers: bearer(token),
  })
  if (!res.ok) throw new Error(`Transcript fetch failed (${res.status})`)
  return res.json()  // { chunks: [{ start_timestamp, end_timestamp, speaker_name, speaker_role, call_section, transcript }] }
}

export async function getChunkMeta(token, chunkId) {
  const res = await fetch(`${API}/api/sources/${encodeURIComponent(chunkId)}`, {
    headers: bearer(token),
  })
  if (!res.ok) throw new Error(`Chunk metadata fetch failed (${res.status})`)
  return res.json()
}

// ── Guest mode ─────────────────────────────────────────────────────────────

export async function createGuestSession() {
  const res = await fetch(`${API}/auth/guest`, { method: 'POST' })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Guest session failed (${res.status})`)
  }
  return res.json()
  // { access_token, token_type, expires_in, guest_user_id, queries_left, uploads_left }
}

export async function getGuestLimits(token) {
  try {
    const res = await fetch(`${API}/auth/guest/limits`, { headers: bearer(token) })
    if (!res.ok) return null
    return res.json()  // { queries_left, uploads_left, queries_used, uploads_used }
  } catch {
    return null
  }
}

export async function convertGuestToUser(guestToken, payload) {
  // payload: { email, password } for email/password path
  const res = await fetch(`${API}/auth/guest/convert`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ guest_token: guestToken, ...payload }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    const status = res.status
    const error = new Error(err.detail || `Conversion failed (${status})`)
    error.status = status
    throw error
  }
  return res.json()
  // { access_token, refresh_token, token_type, expires_in, email, migration_stats }
}

export async function migrateGuestData(realToken, guestToken) {
  // Called after Google OAuth conversion — real JWT in Authorization header, guest token in body
  const res = await fetch(`${API}/auth/guest/migrate`, {
    method: 'POST',
    headers: { ...bearer(realToken), 'Content-Type': 'application/json' },
    body: JSON.stringify({ guest_token: guestToken }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Migration failed (${res.status})`)
  }
  return res.json()  // { migrated: true, migration_stats }
}
