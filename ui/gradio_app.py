"""
ui/gradio_app.py — Phase 28 Gradio UI
Multimodal AGENTIC RAG Knowledge AI Assistant (MAGIK-AI)

Layout (Claude-inspired dark UI):
  ┌─────────────────┬──────────────────────────────────────────┐
  │  Sidebar        │  Main Chat Area                           │
  │  • Brand logo   │  Welcome: "Good afternoon, <name>"        │
  │  • New Chat     │  Chatbot (messages type)                  │
  │  • KB file list │  Transparency panel (route/conf/sources)  │
  │  • Upload files │  Feedback row (👍 👎)                      │
  │  • Delete file  │  Message input + Send button              │
  │  • User / Logout│                                           │
  └─────────────────┴──────────────────────────────────────────┘

Login flow:
  1. "Continue with Google" — browser navigates to /auth/google, OAuth completes,
     backend redirects to Gradio with ?magik_token=...&magik_email=...
     demo.load() picks these up and auto-logs the user in.
  2. "Continue with email" — reveals password field → Sign In
  3. "Create an account" — switches to register form
  4. "Back to sign in" — returns to entry mode

Hard rule: every API call routes through ui/client.py → FastAPI.
Never import app.* directly.
"""
from __future__ import annotations

import os
import urllib.parse
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from ui import client
from ui.feedback import save_feedback
from ui.theme import CSS, SHOW_PASSWORD_JS

BACKEND_URL: str = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ── Constants ─────────────────────────────────────────────────────────────────

MODALITY_ICONS: Dict[str, str] = {
    "pdf":     "📄",
    "text":    "📝",
    "docx":    "📝",
    "xlsx":    "📊",
    "image":   "🖼️",
    "audio":   "🎵",
    "video":   "🎬",
    "unknown": "📎",
}

ALLOWED_EXTENSIONS = [
    ".pdf", ".txt", ".md", ".docx", ".xlsx", ".xls",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac",
    ".mp4", ".avi", ".mov", ".mkv", ".webm",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _greeting() -> str:
    h = datetime.now().hour
    if h < 12:
        return "Good morning"
    elif h < 17:
        return "Good afternoon"
    return "Good evening"


def _email_initial(email: str) -> str:
    return email[0].upper() if email else "?"


def _email_name(email: str) -> str:
    return email.split("@")[0] if "@" in email else email


def _welcome_html(email: str) -> str:
    name = _email_name(email).replace(".", " ").title()
    return (
        f'<div class="welcome-header">'
        f'<span class="welcome-star">✦</span>'
        f'<h2>{_greeting()}, <strong>{name}</strong></h2>'
        f'</div>'
    )


def _user_chip_html(email: str) -> str:
    initial = _email_initial(email)
    return (
        f'<div class="user-chip">'
        f'<div class="user-avatar">{initial}</div>'
        f'<span class="user-email" title="{email}">{email}</span>'
        f'</div>'
    )


def _kb_html(files: List[Dict]) -> str:
    if not files:
        return '<p class="kb-empty">No files uploaded yet</p>'
    rows = []
    for f in files:
        icon = MODALITY_ICONS.get(f.get("modality", "unknown"), "📎")
        name = f.get("filename", "")
        size = f.get("size_mb", 0.0)
        rows.append(
            f'<div class="kb-file-row">'
            f'<span class="kb-icon">{icon}</span>'
            f'<span class="kb-name" title="{name}">{name}</span>'
            f'<span class="kb-size">{size:.2f}MB</span>'
            f'</div>'
        )
    return '<div class="kb-file-list">' + "\n".join(rows) + "</div>"


def _kb_choices(files: List[Dict]) -> List[str]:
    return [f["filename"] for f in files if "filename" in f]


def _format_transparency(result: Dict) -> str:
    decision    = (result.get("decision") or "rag").upper()
    confidence  = result.get("confidence", 0.0)
    latency     = result.get("latency", 0.0)
    sources     = result.get("sources") or []
    hw          = result.get("hallucination_warning", False)
    cache_hit   = result.get("cache_hit", False)

    badges = [
        f"**Route:** `{decision}`",
        f"**Confidence:** `{confidence:.0%}`",
        f"**Latency:** `{latency:.1f}s`",
    ]
    if cache_hit:
        badges.append("⚡ **Cache Hit**")
    if hw:
        badges.append("⚠️ **Low Confidence**")

    header = "  ·  ".join(badges)

    src_lines: List[str] = []
    for i, s in enumerate(sources[:5], 1):
        if isinstance(s, dict):
            name    = s.get("source") or s.get("filename") or "Unknown"
            snippet = (s.get("content") or s.get("text") or "")[:140]
            src_lines.append(f"**{i}.** `{name}` — {snippet}{'…' if len(snippet) >= 140 else ''}")
        elif isinstance(s, str):
            src_lines.append(f"**{i}.** {s[:140]}")

    if src_lines:
        return header + "\n\n**Sources:**\n" + "\n".join(src_lines)
    return header


# ── Login Flow Helpers ────────────────────────────────────────────────────────

_GOOGLE_SVG = (
    '<svg width="18" height="18" viewBox="0 0 18 18" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M17.64 9.205c0-.639-.057-1.252-.164-1.841H9v3.481h4.844a4.14 4.14 0 01-1.796'
    ' 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>'
    '<path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86'
    '-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 009 18z" fill="#34A853"/>'
    '<path d="M3.964 10.71A5.41 5.41 0 013.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996'
    ' 8.996 0 000 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>'
    '<path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9'
    ' 0A8.997 8.997 0 00.957 4.958L3.964 6.29C4.672 4.163 6.656 3.58 9 3.58z" fill="#EA4335"/>'
    '</svg>'
)


def _login_brand_html() -> str:
    return (
        '<div class="login-brand-section">'
        '<span class="login-brand-icon">✦</span>'
        '<h1 class="login-heading">Think smart,<br>build knowledge</h1>'
        '<p class="login-subtitle">Multimodal Agentic Knowledge AI Assistant</p>'
        '</div>'
    )


def _google_btn_html() -> str:
    return (
        f'<a href="{BACKEND_URL}/auth/google" class="google-btn-link">'
        f'{_GOOGLE_SVG}'
        f'<span>Continue with Google</span>'
        f'</a>'
        f'<div class="or-divider"><span>OR</span></div>'
    )


# ── Event Handlers ────────────────────────────────────────────────────────────

def _no_auth_return(error_msg: str) -> Tuple:
    """Helper — return a failed login state (keeps login view visible)."""
    return (
        None,
        gr.update(visible=True),
        gr.update(visible=False),
        f'<p class="login-error">{error_msg}</p>',
        "", "",
        '<p class="kb-empty">No files uploaded yet</p>',
        gr.update(choices=[], value=None),
    )


def _handle_login(email: str, password: str) -> Tuple:
    """
    Step 1 of login — just authenticate and switch views.
    KB load happens in a separate .then() so it never blocks the view switch.
    """
    try:
        email    = (email or "").strip()
        password = (password or "").strip()

        if not email or not password:
            return _no_auth_return("Email and password are required.")

        result = client.login(email, password)

        if "access_token" not in result:
            detail = result.get("detail", result.get("msg", "Invalid email or password"))
            return _no_auth_return(detail)

        token      = result["access_token"]
        session_id = str(uuid.uuid4())
        auth       = {"token": token, "email": email, "session_id": session_id}

        return (
            auth,
            gr.update(visible=False),          # hide login
            gr.update(visible=True),            # show main
            "",                                 # clear login error
            _welcome_html(email),
            _user_chip_html(email),
            '<p class="kb-empty">Loading knowledge base…</p>',
            gr.update(choices=[], value=None),  # refreshed by .then()
        )
    except Exception as exc:
        return _no_auth_return(f"Error: {str(exc)[:160]}")


def _refresh_kb(auth_state: Optional[Dict]) -> Tuple:
    """Step 2 of login — load KB file list after views have switched."""
    if not auth_state or not auth_state.get("token"):
        return '<p class="kb-empty">Not logged in</p>', gr.update(choices=[], value=None)
    kb_html, kb_choices = _load_kb(auth_state["token"])
    return kb_html, gr.update(choices=kb_choices, value=None)


def _handle_register(email: str, password: str) -> Tuple:
    """Register a new account. On success, switch to sign-in mode."""
    email    = (email or "").strip()
    password = (password or "").strip()

    if not email or not password:
        return (
            '<p class="login-error">Email and password are required.</p>',
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )
    if len(password) < 8:
        return (
            '<p class="login-error">Password must be at least 8 characters.</p>',
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )

    try:
        result = client.register(email, password)
    except Exception as e:
        return (
            f'<p class="login-error">Connection error: {str(e)[:120]}</p>',
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        )

    if "user_id" in result or "email" in result:
        # Switch to sign-in mode so user can immediately log in
        return (
            '<p class="login-success">✓ Account created — sign in below.</p>',
            gr.update(interactive=False),       # lock email
            gr.update(visible=False),           # hide continue btn
            gr.update(visible=True, value=""),  # show password (cleared)
            gr.update(visible=True),            # show signin btn
            gr.update(visible=False),           # hide reg btn
            gr.update(visible=False),           # hide back btn
        )

    detail = result.get("detail", "Registration failed")
    return (
        f'<p class="login-error">{detail}</p>',
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
    )


def _continue_with_email(email: str) -> Tuple:
    """Validate email then reveal password field and Sign In button."""
    email = (email or "").strip()
    if not email or "@" not in email:
        return (
            '<p class="login-error">Please enter a valid email address.</p>',
            gr.update(),            # login_email
            gr.update(visible=True),   # continue_email_btn
            gr.update(visible=False),  # login_password
            gr.update(visible=False),  # signin_btn
        )
    return (
        "",                              # clear error
        gr.update(interactive=False),    # lock email
        gr.update(visible=False),        # hide continue btn
        gr.update(visible=True),         # show password
        gr.update(visible=True),         # show signin btn
    )


def _switch_to_register(email: str) -> Tuple:
    """Switch login form to Create Account mode."""
    return (
        "",                                                         # clear error
        gr.update(interactive=True, value=(email or "").strip()),   # unlock email
        gr.update(visible=False),                                   # hide continue btn
        gr.update(visible=True, value="", placeholder="Password (min 8 characters)"),  # password
        gr.update(visible=False),                                   # hide signin btn
        gr.update(visible=True),                                    # show reg btn
        gr.update(visible=False),                                   # hide create-account btn
        gr.update(visible=True),                                    # show back btn
    )


def _back_to_entry() -> Tuple:
    """Return from register/signin mode back to the initial email-entry state."""
    return (
        "",                                           # clear error
        gr.update(interactive=True, value=""),        # reset email
        gr.update(visible=True),                      # show continue btn
        gr.update(visible=False, value=""),           # hide + clear password
        gr.update(visible=False),                     # hide signin btn
        gr.update(visible=False),                     # hide reg btn
        gr.update(visible=True),                      # show create-account btn
        gr.update(visible=False),                     # hide back btn
    )


def _on_load(request: gr.Request) -> Tuple:
    """
    Fires on every page load.  If Google OAuth redirected here with
    ?magik_token=...&magik_email=..., auto-log the user in.
    """
    params = dict(request.query_params) if request else {}

    oauth_error = urllib.parse.unquote(params.get("oauth_error", "")).strip()
    token       = params.get("magik_token", "").strip()
    email_raw   = params.get("magik_email", "").strip()

    if oauth_error:
        return (
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            f'<p class="login-error">Google login failed: {oauth_error[:120]}</p>',
            "", "",
            '<p class="kb-empty">No files uploaded yet</p>',
            gr.update(choices=[], value=None),
        )

    if token and email_raw:
        email      = urllib.parse.unquote(email_raw)
        session_id = str(uuid.uuid4())
        auth       = {"token": token, "email": email, "session_id": session_id}
        return (
            auth,
            gr.update(visible=False),
            gr.update(visible=True),
            "",
            _welcome_html(email),
            _user_chip_html(email),
            '<p class="kb-empty">Loading knowledge base…</p>',
            gr.update(choices=[], value=None),
        )

    # Normal page load — show login, clear everything
    return (
        None,
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
    )


def _handle_logout(auth_state: Optional[Dict]) -> Tuple:
    """Revoke token, return to login view, and reset the login form to entry state."""
    if auth_state and auth_state.get("token"):
        try:
            client.logout_user(auth_state["token"])
        except Exception:
            pass

    return (
        None,                                       # clear auth state
        [],                                         # clear chat history
        {},                                         # clear transparency state
        gr.update(visible=True),                    # show login view
        gr.update(visible=False),                   # hide main view
        "",                                         # clear login error
        '<p class="kb-empty">No files uploaded yet</p>',
        [],                                         # clear delete dropdown
        # ── Reset login form to entry state ──────────────────────────────
        gr.update(interactive=True, value=""),      # reset email field
        gr.update(visible=True),                    # show continue-email btn
        gr.update(visible=False, value=""),         # hide + clear password
        gr.update(visible=False),                   # hide signin btn
        gr.update(visible=False),                   # hide reg btn
        gr.update(visible=True),                    # show create-account btn
        gr.update(visible=False),                   # hide back btn
    )


def _new_chat(auth_state: Optional[Dict]) -> Tuple:
    """Start a fresh conversation (new session_id, empty history, clear memory)."""
    if not auth_state:
        return auth_state, [], {}

    new_session = str(uuid.uuid4())
    new_auth    = {**auth_state, "session_id": new_session}

    try:
        client.clear_memory(new_session, auth_state["token"])
    except Exception:
        pass

    return new_auth, [], {}


def _handle_chat(
    message: str,
    history: List[Dict],
    auth_state: Optional[Dict],
    transparency_state: Dict,
):
    """
    Generator — yields (history, transparency_md, msg_input_clear, trans_state).
    Step 1: immediately adds user msg + "Thinking…" placeholder.
    Step 2: calls /query, replaces placeholder with real answer + metadata.
    """
    message = (message or "").strip()
    if not message:
        yield history, gr.update(), "", transparency_state
        return

    token      = (auth_state or {}).get("token")
    session_id = (auth_state or {}).get("session_id", "default")

    if not token:
        error_history = history + [
            {"role": "user",      "content": message},
            {"role": "assistant", "content": "⚠️ Please login to use the assistant."},
        ]
        yield error_history, gr.update(), "", transparency_state
        return

    # ── Step 1: show user msg + thinking indicator ────────────────────────
    base_history = history + [{"role": "user", "content": message}]
    thinking_history = base_history + [{"role": "assistant", "content": "⏳ Thinking…"}]
    yield thinking_history, gr.update(), "", transparency_state

    # ── Step 2: call /query (non-streaming, returns full metadata) ────────
    try:
        result = client.query(message, session_id, token)
        answer = result.get("answer") or "No response returned."

        trans_md  = _format_transparency(result)
        new_trans = {
            "query":      message,
            "answer":     answer,
            "decision":   result.get("decision", "rag"),
            "sources":    result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "session_id": session_id,
        }

        # ── Step 3: stream the answer word-by-word ────────────────────────
        words     = answer.split(" ")
        displayed = ""
        for i, word in enumerate(words):
            displayed += ("" if i == 0 else " ") + word
            stream_history = base_history + [{"role": "assistant", "content": displayed + " ▌"}]
            yield stream_history, gr.update(), "", transparency_state

        # Final yield — remove cursor, show transparency
        final_history = base_history + [{"role": "assistant", "content": answer}]
        yield final_history, trans_md, "", new_trans

    except Exception as exc:
        error_history = base_history + [
            {"role": "assistant", "content": f"❌ Error: {str(exc)[:300]}"}
        ]
        yield error_history, gr.update(), "", transparency_state


def _load_kb(token: str) -> Tuple[str, List[str]]:
    """Fetch KB file list; return (html_str, [filename, ...])."""
    try:
        data  = client.list_kb(token)
        files = data.get("files", [])
        return _kb_html(files), _kb_choices(files)
    except Exception:
        return '<p class="kb-empty">Could not load knowledge base</p>', []


def _handle_upload(
    files,                          # Gradio UploadButton returns list of temp paths
    auth_state: Optional[Dict],
) -> Tuple[str, str, List[str]]:
    """Upload every selected file to /ingest; return (status_html, kb_html, choices)."""
    if not auth_state or not auth_state.get("token"):
        return (
            '<span class="upload-status" style="color:#e05252">Not logged in</span>',
            '<p class="kb-empty">No files uploaded yet</p>',
            [],
        )

    token      = auth_state["token"]
    session_id = auth_state.get("session_id", "default")

    if not files:
        return (
            '<span class="upload-status" style="color:#888">No files selected</span>',
            *_load_kb(token)[0:2],
        )

    # Gradio passes a list of temp file paths (strings or NamedString objects)
    paths = [f if isinstance(f, str) else f.name for f in (files if isinstance(files, list) else [files])]

    results, ok, fail = [], 0, 0
    for path in paths:
        try:
            r = client.ingest(path, session_id, token)
            status = r.get("status", "")
            if status in ("success", "partial_failure"):
                fname = r.get("filename", path.split("/")[-1])
                chunks = r.get("chunks", 0)
                results.append(f"✓ {fname} ({chunks} chunks)")
                ok += 1
            elif status == "duplicate":
                fname = r.get("filename", path.split("/")[-1])
                results.append(f"≡ {fname} (duplicate)")
                ok += 1
            else:
                ec = r.get("error_code", r.get("detail", "failed"))
                results.append(f"✗ {path.split('/')[-1]}: {ec}")
                fail += 1
        except Exception as e:
            results.append(f"✗ {path.split('/')[-1]}: {str(e)[:80]}")
            fail += 1

    color  = "#4caf50" if fail == 0 else "#e05252"
    summary = f"{ok} uploaded" + (f", {fail} failed" if fail else "")
    detail_lines = "<br>".join(results[:8])
    status_html  = (
        f'<span class="upload-status" style="color:{color}">{summary}</span>'
        f'<div style="font-size:11px;color:#666;margin-top:2px">{detail_lines}</div>'
    )

    kb_html, kb_choices = _load_kb(token)
    return status_html, kb_html, gr.update(choices=kb_choices, value=None)


def _handle_delete(
    filename: Optional[str],
    auth_state: Optional[Dict],
) -> Tuple[str, str, List[str]]:
    """Delete one file from the KB; refresh list."""
    if not auth_state or not auth_state.get("token"):
        return (
            '<span class="delete-status" style="color:#e05252">Not logged in</span>',
            '<p class="kb-empty">No files uploaded yet</p>',
            [],
        )
    if not filename:
        return (
            '<span class="delete-status" style="color:#888">Select a file first</span>',
            *_load_kb(auth_state["token"])[0:2],
        )

    token = auth_state["token"]
    try:
        client.delete_kb_file(filename, token)
        status_html = f'<span class="delete-status" style="color:#4caf50">✓ Deleted "{filename}"</span>'
    except Exception as e:
        status_html = f'<span class="delete-status" style="color:#e05252">✗ {str(e)[:120]}</span>'

    kb_html, kb_choices = _load_kb(token)
    return status_html, kb_html, gr.update(choices=kb_choices, value=None)


def _handle_feedback(
    rating: str,
    transparency_state: Dict,
    auth_state: Optional[Dict],
) -> str:
    """Save thumbs feedback to gold JSONL; return status string."""
    if not transparency_state or not transparency_state.get("query"):
        return '<span class="feedback-status">No response to rate yet</span>'

    try:
        save_feedback(
            query      = transparency_state.get("query", ""),
            answer     = transparency_state.get("answer", ""),
            rating     = rating,
            route      = transparency_state.get("decision", "rag"),
            sources    = transparency_state.get("sources", []),
            session_id = transparency_state.get("session_id", ""),
            confidence = transparency_state.get("confidence", 0.0),
        )
        icon = "👍" if rating == "positive" else "👎"
        return f'<span class="feedback-status">{icon} Feedback saved</span>'
    except Exception as e:
        return f'<span class="feedback-status" style="color:#e05252">Save failed: {str(e)[:60]}</span>'


# ── Build the App ─────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="MAGIK-AI — Multimodal Agentic Knowledge AI",
        analytics_enabled=False,
    ) as demo:

        # ── Global State ──────────────────────────────────────────────────
        auth_state          = gr.State(None)   # {token, email, session_id}
        transparency_state  = gr.State({})     # last query result metadata

        # ══════════════════════════════════════════════════════════════════
        # LOGIN VIEW  —  Claude-inspired design
        # ══════════════════════════════════════════════════════════════════
        with gr.Column(visible=True, elem_id="login-view") as login_view:
            with gr.Column(elem_id="login-card"):

                # ── Brand header ──────────────────────────────────────────
                gr.HTML(_login_brand_html())

                # ── Form card ─────────────────────────────────────────────
                with gr.Column(elem_id="login-form-container"):

                    # Google OAuth button
                    gr.HTML(_google_btn_html())

                    # Email input (always visible)
                    login_email = gr.Textbox(
                        placeholder="Enter your email",
                        show_label=False,
                        elem_id="login-email",
                    )

                    # "Continue with email" — reveals password step
                    continue_email_btn = gr.Button(
                        "Continue with email",
                        elem_id="continue-email-btn",
                        variant="primary",
                    )

                    # Password (hidden until "Continue with email" is clicked)
                    login_password = gr.Textbox(
                        type="password",
                        placeholder="Password",
                        show_label=False,
                        elem_id="login-password",
                        visible=False,
                    )

                    # Sign In button (hidden initially)
                    signin_btn = gr.Button(
                        "Sign In",
                        elem_id="signin-btn",
                        variant="primary",
                        visible=False,
                    )

                    # Register button (hidden initially, shown in register mode)
                    reg_btn = gr.Button(
                        "Create Account",
                        elem_id="reg-btn",
                        variant="primary",
                        visible=False,
                    )

                    # Error / success feedback
                    login_error = gr.HTML("")

                    # Footer nav — toggle between sign-in and create-account
                    with gr.Row(elem_id="login-footer"):
                        create_account_btn = gr.Button(
                            "Create an account",
                            elem_id="create-account-btn",
                            size="sm",
                        )
                        back_signin_btn = gr.Button(
                            "← Back to sign in",
                            elem_id="back-signin-btn",
                            size="sm",
                            visible=False,
                        )

        # ══════════════════════════════════════════════════════════════════
        # MAIN VIEW
        # ══════════════════════════════════════════════════════════════════
        with gr.Column(visible=False, elem_id="main-view") as main_view:
            with gr.Row(elem_id="app-row", equal_height=True):

                # ── Sidebar ───────────────────────────────────────────────
                with gr.Column(scale=1, min_width=260, elem_id="sidebar"):

                    gr.HTML(
                        '<div class="sidebar-brand">'
                        '<span class="star">✦</span>'
                        '<div>'
                        '<div class="brand-name">MAGIK-AI</div>'
                        '<div class="brand-sub">Multimodal · Agentic · Knowledge</div>'
                        '</div>'
                        '</div>'
                    )

                    new_chat_btn = gr.Button("＋  New Chat", elem_id="new-chat-btn")

                    gr.HTML('<div class="section-header">Knowledge Base</div>')

                    kb_list_html = gr.HTML(
                        '<p class="kb-empty">No files uploaded yet</p>',
                        elem_id="kb-list",
                    )

                    upload_files = gr.File(
                        file_count="multiple",
                        label="Drop files here to upload",
                        file_types=ALLOWED_EXTENSIONS,
                        elem_id="upload-zone",
                        height=80,
                    )
                    upload_btn    = gr.Button("Upload to Knowledge Base", elem_id="upload-btn", size="sm")
                    upload_status = gr.HTML("", elem_id="upload-status")

                    gr.HTML('<div class="section-header" style="margin-top:8px">Delete File</div>')
                    delete_select = gr.Dropdown(
                        choices=[],
                        label="File to delete",
                        interactive=True,
                        elem_id="delete-select",
                    )
                    delete_btn    = gr.Button("Delete from Knowledge Base", elem_id="delete-btn", size="sm", variant="stop")
                    delete_status = gr.HTML("", elem_id="delete-status")

                    # Spacer pushes user chip to bottom
                    gr.HTML('<div style="flex:1"></div>')

                    with gr.Column(elem_id="sidebar-footer"):
                        user_chip_html = gr.HTML("", elem_id="user-chip")
                        logout_btn     = gr.Button("Log out", elem_id="logout-btn", size="sm")

                # ── Main Chat Area ────────────────────────────────────────
                with gr.Column(scale=4, elem_id="main-area"):

                    welcome_html = gr.HTML("", elem_id="welcome-html")

                    chatbot = gr.Chatbot(
                        elem_id="chatbot-window",
                        show_label=False,
                        height=480,
                        placeholder=(
                            "Upload documents using the sidebar, then ask anything about them.\n\n"
                            "Supports PDF · Word · Excel · Image · Audio · Video · Text"
                        ),
                        render_markdown=True,
                    )

                    # Transparency / Response Detail accordion
                    with gr.Accordion(
                        "Response Details",
                        open=False,
                        elem_id="transparency-panel",
                    ) as trans_accordion:
                        transparency_md = gr.Markdown(
                            "",
                            elem_id="transparency-content",
                        )

                    # Feedback row
                    with gr.Row(elem_id="feedback-row"):
                        thumbs_up_btn   = gr.Button("👍", elem_id="thumbs-up-btn",   size="sm")
                        thumbs_down_btn = gr.Button("👎", elem_id="thumbs-down-btn", size="sm")
                        feedback_status = gr.HTML("", elem_id="feedback-status")

                    # Message input row
                    with gr.Row(elem_id="input-row"):
                        msg_box = gr.Textbox(
                            placeholder="How can I help you today?",
                            show_label=False,
                            lines=1,
                            max_lines=6,
                            elem_id="msg-input",
                            scale=10,
                            submit_btn=False,
                        )
                        send_btn = gr.Button(
                            "↑",
                            variant="primary",
                            elem_id="send-btn",
                            scale=0,
                            min_width=44,
                        )

        # ══════════════════════════════════════════════════════════════════
        # WIRING — Login
        # ══════════════════════════════════════════════════════════════════

        _login_outputs = [
            auth_state,
            login_view,
            main_view,
            login_error,
            welcome_html,
            user_chip_html,
            kb_list_html,
            delete_select,
        ]

        # ── OAuth auto-login on page load ─────────────────────────────────
        demo.load(
            fn=_on_load,
            inputs=None,
            outputs=_login_outputs,
        ).then(
            fn=_refresh_kb,
            inputs=[auth_state],
            outputs=[kb_list_html, delete_select],
        )

        # ── "Continue with email" — reveal password step ──────────────────
        _continue_outputs = [login_error, login_email, continue_email_btn, login_password, signin_btn]
        continue_email_btn.click(
            fn=_continue_with_email,
            inputs=[login_email],
            outputs=_continue_outputs,
        )
        login_email.submit(
            fn=_continue_with_email,
            inputs=[login_email],
            outputs=_continue_outputs,
        )

        # ── Sign In ───────────────────────────────────────────────────────
        signin_btn.click(
            fn=_handle_login,
            inputs=[login_email, login_password],
            outputs=_login_outputs,
        ).then(
            fn=_refresh_kb,
            inputs=[auth_state],
            outputs=[kb_list_html, delete_select],
        )
        login_password.submit(
            fn=_handle_login,
            inputs=[login_email, login_password],
            outputs=_login_outputs,
        ).then(
            fn=_refresh_kb,
            inputs=[auth_state],
            outputs=[kb_list_html, delete_select],
        )

        # ── Switch to Create Account mode ─────────────────────────────────
        _mode_outputs = [
            login_error, login_email, continue_email_btn,
            login_password, signin_btn, reg_btn,
            create_account_btn, back_signin_btn,
        ]
        create_account_btn.click(
            fn=_switch_to_register,
            inputs=[login_email],
            outputs=_mode_outputs,
        )

        # ── Back to Sign In ────────────────────────────────────────────────
        back_signin_btn.click(
            fn=_back_to_entry,
            inputs=None,
            outputs=_mode_outputs,
        )

        # ── Register / Create Account ──────────────────────────────────────
        _register_outputs = [
            login_error, login_email, continue_email_btn,
            login_password, signin_btn, reg_btn, back_signin_btn,
        ]
        reg_btn.click(
            fn=_handle_register,
            inputs=[login_email, login_password],
            outputs=_register_outputs,
        )

        # ── Logout ────────────────────────────────────────────────────────
        _logout_outputs = [
            auth_state,
            chatbot,
            transparency_state,
            login_view,
            main_view,
            login_error,
            kb_list_html,
            delete_select,
            # Reset login form to entry state
            login_email,
            continue_email_btn,
            login_password,
            signin_btn,
            reg_btn,
            create_account_btn,
            back_signin_btn,
        ]
        logout_btn.click(
            fn=_handle_logout,
            inputs=[auth_state],
            outputs=_logout_outputs,
        )

        # ── New Chat ──────────────────────────────────────────────────────
        new_chat_btn.click(
            fn=_new_chat,
            inputs=[auth_state],
            outputs=[auth_state, chatbot, transparency_state],
        ).then(
            fn=lambda: ("", ""),
            outputs=[transparency_md, feedback_status],
        )

        # ══════════════════════════════════════════════════════════════════
        # WIRING — Chat
        # ══════════════════════════════════════════════════════════════════

        _chat_outputs = [chatbot, transparency_md, msg_box, transparency_state]

        send_btn.click(
            fn=_handle_chat,
            inputs=[msg_box, chatbot, auth_state, transparency_state],
            outputs=_chat_outputs,
        )
        msg_box.submit(
            fn=_handle_chat,
            inputs=[msg_box, chatbot, auth_state, transparency_state],
            outputs=_chat_outputs,
        )

        # ── Transparency accordion auto-open when new content arrives ─────
        transparency_md.change(
            fn=lambda md: gr.update(open=bool(md and md.strip())),
            inputs=[transparency_md],
            outputs=[trans_accordion],
        )

        # ══════════════════════════════════════════════════════════════════
        # WIRING — Knowledge Base
        # ══════════════════════════════════════════════════════════════════

        _upload_outputs = [upload_status, kb_list_html, delete_select]

        upload_btn.click(
            fn=_handle_upload,
            inputs=[upload_files, auth_state],
            outputs=_upload_outputs,
        )

        _delete_outputs = [delete_status, kb_list_html, delete_select]

        delete_btn.click(
            fn=_handle_delete,
            inputs=[delete_select, auth_state],
            outputs=_delete_outputs,
        )

        # ══════════════════════════════════════════════════════════════════
        # WIRING — Feedback
        # ══════════════════════════════════════════════════════════════════

        thumbs_up_btn.click(
            fn=lambda ts, auth: _handle_feedback("positive", ts, auth),
            inputs=[transparency_state, auth_state],
            outputs=[feedback_status],
        )
        thumbs_down_btn.click(
            fn=lambda ts, auth: _handle_feedback("negative", ts, auth),
            inputs=[transparency_state, auth_state],
            outputs=[feedback_status],
        )

    return demo


# ── Entry Point ───────────────────────────────────────────────────────────────

app = build_app()

if __name__ == "__main__":
    port       = int(os.getenv("GRADIO_PORT", "7860"))
    share      = os.getenv("GRADIO_SHARE", "false").lower() == "true"
    server_name = os.getenv("GRADIO_HOST", "0.0.0.0")

    app.launch(
        server_name=server_name,
        server_port=port,
        share=share,
        show_error=True,
        css=CSS,
        js=SHOW_PASSWORD_JS,
    )
