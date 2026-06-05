const API = ''  // empty = same origin via Vite proxy

function bearer(token) {
  return { Authorization: `Bearer ${token}` }
}

export async function login(email, password) {
  const res = await fetch(`${API}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Login failed (${res.status})`)
  }
  return res.json()  // { access_token, token_type, refresh_token? }
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
  return res.json()
}

export async function getMe(token) {
  const res = await fetch(`${API}/auth/me`, { headers: bearer(token) })
  if (!res.ok) throw new Error('Session expired')
  return res.json()  // { user_id, email, ... }
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

export async function ingestFile(token, file, sessionId = 'default') {
  const form = new FormData()
  form.append('file', file)
  form.append('session_id', sessionId)
  const res = await fetch(`${API}/rag/ingest`, {
    method: 'POST',
    headers: bearer(token),
    body: form,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `Upload failed (${res.status})`)
  }
  return res.json()
}

// Returns the raw fetch Response so caller can stream the body
export function streamQuery(token, query, sessionId) {
  return fetch(`${API}/rag/query/stream`, {
    method: 'POST',
    headers: { ...bearer(token), 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId }),
  })
}

// Called after streaming to get sources + metadata (hits Redis cache)
export async function queryMeta(token, query, sessionId) {
  const res = await fetch(`${API}/rag/query`, {
    method: 'POST',
    headers: { ...bearer(token), 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId }),
  })
  if (!res.ok) return null
  return res.json()
  // { answer, sources, confidence, decision, latency, ... }
}
