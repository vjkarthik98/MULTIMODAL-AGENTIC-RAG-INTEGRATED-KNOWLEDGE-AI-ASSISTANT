"""
ui/theme.py — Dark CSS theme inspired by Claude's desktop UI.

Colour palette
  bg-deep    #1a1a1a   overall page background
  bg-surface #212121   main chat area
  bg-card    #2a2a2a   message cards, input box
  bg-sidebar #1e1e1e   left panel
  border     #333333   dividers / input borders
  text-hi    #ececec   primary text
  text-lo    #888888   muted / secondary text
  accent     #d4774e   brand orange (Claude logo colour)
  user-msg   #2b3d55   user message bubble tint
  success    #4caf50
  danger     #e05252
"""

CSS = """
/* ─── Reset & Base ─────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #1a1a1a !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.gradio-container {
    max-width: 100% !important;
}

footer { display: none !important; }

/* ─── Login View ─────────────────────────────────────────────────────────── */
#login-view {
    min-height: 100vh;
    background: #0e0e0e;
    display: flex !important;
    align-items: flex-start;
    justify-content: center;
    padding: 40px 24px 60px;
}

/* Stop Gradio's flex children from stretching vertically */
#login-view > .gap,
#login-view > .block {
    width: 100% !important;
    flex: 0 0 auto !important;
    height: auto !important;
}

#login-card {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    width: 100% !important;
    max-width: 480px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    text-align: center;
    flex: 0 0 auto !important;
    height: auto !important;
}

/* Stop login-card inner gap from stretching */
#login-card > .gap,
#login-card > .block > .gap {
    flex: 0 0 auto !important;
    height: auto !important;
    gap: 0 !important;
}

/* Brand section */
.login-brand-section {
    padding-top: 16px;
    margin-bottom: 28px;
}

.login-brand-icon {
    font-size: 36px;
    color: #d4774e;
    display: block;
    margin: 0 auto 16px;
    line-height: 1;
}

.login-heading {
    font-size: 42px;
    font-weight: 400;
    color: #f2f2f2;
    font-family: Georgia, 'Times New Roman', serif;
    line-height: 1.15;
    margin: 0 0 10px;
    letter-spacing: -0.02em;
}

.login-subtitle {
    color: #999;
    font-size: 15px;
    margin: 0;
    line-height: 1.4;
}

/* ─── Form card — MUST be height:auto so it doesn't fill the page ── */
#login-form-container {
    background: #181818 !important;
    border: 1px solid #282828 !important;
    border-radius: 20px !important;
    padding: 24px 28px 20px !important;
    height: auto !important;
    flex: 0 0 auto !important;
    overflow: visible !important;
}

/* Inner gap: use column layout with 10px gap, no stretching */
#login-form-container > .gap,
#login-form-container > .block > .gap {
    flex: 0 0 auto !important;
    height: auto !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
}

/* Google button */
.google-btn-link {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 11px;
    background: #1e1e1e;
    border: 1px solid #363636;
    border-radius: 32px;
    color: #e8e8e8;
    font-size: 15px;
    font-weight: 500;
    padding: 13px 24px;
    text-decoration: none !important;
    width: 100%;
    cursor: pointer;
    transition: background 0.15s, border-color 0.15s;
    box-sizing: border-box;
}
.google-btn-link:hover {
    background: #262626;
    border-color: #555;
    color: #fff;
    text-decoration: none !important;
}
.google-btn-link svg { flex-shrink: 0; }

/* OR divider */
.or-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 4px 0;
    color: #555;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.or-divider::before, .or-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #282828;
}

/* ─── Login inputs — CRITICAL: target input/textarea only, NOT the label ─── */
/* In Gradio 6 the <label> wraps the entire input — never hide it              */
#login-email input,
#login-email textarea {
    background: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    color: #ececec !important;
    font-size: 15px !important;
    padding: 13px 16px !important;
    transition: border-color 0.15s !important;
    pointer-events: all !important;
    cursor: text !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
#login-password input,
#login-password textarea {
    background: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    color: #ececec !important;
    font-size: 15px !important;
    padding: 13px 44px 13px 16px !important;  /* right pad for eye icon */
    transition: border-color 0.15s !important;
    pointer-events: all !important;
    cursor: text !important;
    width: 100% !important;
    box-sizing: border-box !important;
}
#login-email input:focus,
#login-email textarea:focus,
#login-password input:focus,
#login-password textarea:focus {
    border-color: #555 !important;
    box-shadow: 0 0 0 3px rgba(212,119,78,0.12) !important;
    outline: none !important;
}
#login-email input::placeholder,
#login-email textarea::placeholder,
#login-password input::placeholder,
#login-password textarea::placeholder {
    color: #555 !important;
}

/* Hide the label text only (not the label element which wraps the input) */
#login-email .label-wrap,
#login-password .label-wrap { display: none !important; }
#login-email, #login-password { margin-bottom: 0 !important; }

/* Make the label wrapper itself transparent so the input is directly accessible */
#login-email label,
#login-password label {
    display: block !important;
    position: relative !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
}

/* Relative wrapper for the eye toggle */
#login-password .wrap,
#login-password [class*="wrap"] {
    position: relative !important;
}

/* Eye toggle button — injected by JavaScript */
.pwd-eye-btn {
    position: absolute !important;
    right: 13px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    background: none !important;
    border: none !important;
    color: #666 !important;
    cursor: pointer !important;
    padding: 4px !important;
    z-index: 20 !important;
    line-height: 1 !important;
    font-size: 15px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: color 0.15s !important;
}
.pwd-eye-btn:hover { color: #d4774e !important; }

/* "Continue with email" button — solid white pill */
#continue-email-btn {
    background: #f0f0f0 !important;
    color: #111 !important;
    border: none !important;
    border-radius: 32px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 13px 24px !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
#continue-email-btn:hover { background: #ddd !important; }

/* Sign In button — brand orange pill */
#signin-btn {
    background: #d4774e !important;
    color: #fff !important;
    border: none !important;
    border-radius: 32px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 13px 24px !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
#signin-btn:hover { background: #c06840 !important; }

/* Create Account button */
#reg-btn {
    background: #d4774e !important;
    color: #fff !important;
    border: none !important;
    border-radius: 32px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    padding: 13px 24px !important;
    width: 100% !important;
    transition: background 0.15s !important;
}
#reg-btn:hover { background: #c06840 !important; }

/* Login footer link-style buttons */
#create-account-btn, #back-signin-btn {
    background: transparent !important;
    border: none !important;
    color: #777 !important;
    font-size: 13px !important;
    padding: 4px 8px !important;
    min-width: unset !important;
    transition: color 0.15s !important;
    text-decoration: underline !important;
    text-underline-offset: 3px !important;
}
#create-account-btn:hover, #back-signin-btn:hover {
    color: #d4774e !important;
    background: transparent !important;
}

/* Login footer row — centered */
#login-footer {
    justify-content: center !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
}

/* Error / success messages */
.login-error {
    color: #e05252;
    font-size: 13px;
    text-align: center;
    min-height: 18px;
    margin: 2px 0 0;
}
.login-success {
    color: #4caf50;
    font-size: 13px;
    text-align: center;
    margin: 2px 0 0;
}

/* ─── Main View ─────────────────────────────────────────────────────────── */
#main-view {
    background: #1a1a1a;
    min-height: 100vh;
}

/* ─── App Layout Row ────────────────────────────────────────────────────── */
#app-row {
    min-height: 100vh;
    gap: 0 !important;
}

/* ─── Sidebar ───────────────────────────────────────────────────────────── */
#sidebar {
    background: #1e1e1e !important;
    border-right: 1px solid #2d2d2d !important;
    min-height: 100vh;
    padding: 16px 12px !important;
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 4px 16px;
    border-bottom: 1px solid #2d2d2d;
    margin-bottom: 8px;
}
.sidebar-brand .star { font-size: 20px; color: #d4774e; }
.sidebar-brand .brand-name { color: #ececec; font-size: 14px; font-weight: 600; line-height: 1.2; }
.sidebar-brand .brand-sub { color: #666; font-size: 11px; }

.section-header {
    color: #666 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    padding: 8px 4px 4px !important;
}

/* New Chat button */
#new-chat-btn {
    background: #2a2a2a !important;
    border: 1px solid #333 !important;
    color: #ececec !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
    width: 100% !important;
    text-align: left !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}
#new-chat-btn:hover { background: #333 !important; }

/* KB file list */
.kb-file-list {
    display: flex;
    flex-direction: column;
    gap: 2px;
    max-height: 200px;
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: #333 transparent;
}
.kb-file-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 8px;
    border-radius: 6px;
    background: #252525;
    cursor: default;
    transition: background 0.1s;
}
.kb-file-row:hover { background: #2f2f2f; }
.kb-icon { font-size: 13px; flex-shrink: 0; }
.kb-name { color: #ccc; font-size: 12px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.kb-size { color: #555; font-size: 11px; flex-shrink: 0; }
.kb-empty { color: #555; font-size: 12px; text-align: center; padding: 12px 0; font-style: italic; }

/* File upload dropzone */
#upload-zone .upload-container {
    background: #252525 !important;
    border: 1px dashed #383838 !important;
    border-radius: 8px !important;
}
#upload-zone label { color: #888 !important; font-size: 12px !important; }

/* Small buttons in sidebar */
#upload-btn, #delete-btn {
    font-size: 12px !important;
    padding: 5px 10px !important;
    border-radius: 6px !important;
}
#upload-btn { background: #2b3d55 !important; border: none !important; color: #7eb8f7 !important; }
#upload-btn:hover { background: #344d6a !important; }
#delete-btn { background: #3d2222 !important; border: none !important; color: #e05252 !important; }
#delete-btn:hover { background: #4d2828 !important; }

/* Status messages in sidebar */
.upload-status, .delete-status {
    font-size: 12px;
    padding: 4px 0;
    min-height: 18px;
}

/* User footer */
.sidebar-footer {
    margin-top: auto !important;
    padding-top: 12px;
    border-top: 1px solid #2d2d2d;
}
.user-chip {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 4px;
}
.user-avatar {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: #d4774e;
    color: #fff;
    font-size: 12px;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.user-email { color: #aaa; font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

#logout-btn {
    background: transparent !important;
    border: 1px solid #333 !important;
    color: #888 !important;
    font-size: 12px !important;
    border-radius: 6px !important;
    padding: 5px 10px !important;
    width: 100% !important;
}
#logout-btn:hover { border-color: #555 !important; color: #ccc !important; }

/* ─── Main Chat Area ────────────────────────────────────────────────────── */
#main-area {
    background: #212121 !important;
    min-height: 100vh;
    padding: 0 !important;
}

/* Welcome header */
.welcome-header {
    text-align: center;
    padding: 60px 20px 24px;
}
.welcome-header .welcome-star { font-size: 32px; color: #d4774e; display: block; margin-bottom: 10px; }
.welcome-header h2 { color: #ececec; font-size: 28px; font-weight: 400; margin: 0; letter-spacing: -0.02em; }
.welcome-header h2 strong { font-weight: 600; }

/* Chatbot */
#chatbot-window {
    flex: 1;
    margin: 0 !important;
}

#chatbot-window .chatbot {
    background: #212121 !important;
    border: none !important;
    border-radius: 0 !important;
}

/* Message bubbles */
#chatbot-window .message {
    border-radius: 10px !important;
    max-width: 85% !important;
    font-size: 15px !important;
    line-height: 1.6 !important;
    padding: 12px 16px !important;
}
#chatbot-window .user { background: #2b3d55 !important; color: #d4e8ff !important; margin-left: auto !important; }
#chatbot-window .bot  { background: #2a2a2a !important; color: #e8e8e8 !important; }

/* Transparency / response detail accordion */
#transparency-panel {
    background: #1e1e1e !important;
    border: 1px solid #2d2d2d !important;
    border-radius: 8px !important;
    margin: 0 16px 0 !important;
}
#transparency-panel .label { color: #666 !important; font-size: 12px !important; }
#transparency-panel .prose { color: #bbb !important; font-size: 13px !important; }

/* Feedback row */
#feedback-row {
    padding: 4px 16px;
    gap: 6px !important;
}
#thumbs-up-btn, #thumbs-down-btn {
    background: transparent !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    padding: 4px 10px !important;
    color: #666 !important;
    min-width: 40px !important;
    width: auto !important;
}
#thumbs-up-btn:hover   { border-color: #4caf50 !important; color: #4caf50 !important; }
#thumbs-down-btn:hover { border-color: #e05252 !important; color: #e05252 !important; }
.feedback-status { color: #666; font-size: 12px; align-self: center; }

/* Input row */
#input-row {
    padding: 12px 16px 20px !important;
    gap: 8px !important;
    align-items: flex-end !important;
}

#msg-input textarea {
    background: #2a2a2a !important;
    border: 1px solid #383838 !important;
    border-radius: 12px !important;
    color: #ececec !important;
    font-size: 15px !important;
    padding: 12px 16px !important;
    resize: none !important;
    transition: border-color 0.15s !important;
}
#msg-input textarea:focus { border-color: #555 !important; outline: none !important; }
#msg-input textarea::placeholder { color: #555 !important; }
#msg-input label { display: none !important; }

#send-btn {
    background: #d4774e !important;
    border: none !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-size: 18px !important;
    height: 44px !important;
    min-width: 44px !important;
    padding: 0 !important;
    flex-shrink: 0 !important;
}
#send-btn:hover { background: #c06840 !important; }
#send-btn:disabled { background: #444 !important; color: #666 !important; }

/* ─── Gradio Input/Button overrides ────────────────────────────────────── */
.gr-textbox input, .gr-textbox textarea {
    background: #2a2a2a !important;
    border: 1px solid #383838 !important;
    color: #ececec !important;
    border-radius: 8px !important;
}
.gr-textbox input:focus, .gr-textbox textarea:focus {
    border-color: #555 !important;
}

label.svelte-1b6s6xi, .label-wrap span {
    color: #888 !important;
    font-size: 13px !important;
}

/* Dropdown for file delete */
#delete-select select, #delete-select .wrap-inner {
    background: #252525 !important;
    border: 1px solid #333 !important;
    color: #ccc !important;
    font-size: 12px !important;
    border-radius: 6px !important;
}

/* Accordion */
.accordion { background: #1e1e1e !important; border: 1px solid #2d2d2d !important; border-radius: 8px !important; }
.accordion-header { color: #666 !important; font-size: 12px !important; }

/* Tabs */
.tabs { background: transparent !important; border: none !important; }
.tabitem { padding: 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #444; }
"""

# ── Show-password eye-toggle (injected via launch(js=...)) ──────────────────
# Uses MutationObserver so it fires whenever Gradio makes #login-password
# visible (e.g. after "Continue with email" is clicked).
SHOW_PASSWORD_JS = """
() => {
    function tryAddEyeToggle() {
        const container = document.getElementById('login-password');
        if (!container) return;

        // Don't add twice
        if (container.querySelector('.pwd-eye-btn')) return;

        // Gradio 6 may render <input> or <textarea> inside the password field
        const input = container.querySelector('input, textarea');
        if (!input) return;

        // Find the innermost wrapper to position relative to
        const wrap = (
            input.closest('.wrap-inner') ||
            input.closest('[class*="wrap"]') ||
            input.parentElement
        );
        if (!wrap) return;
        wrap.style.position = 'relative';

        const SVG_EYE = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>
            <circle cx="12" cy="12" r="3"/>
        </svg>`;
        const SVG_EYE_OFF = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M17.94 17.94A10.07 10.07 0 0112 20c-7 0-11-8-11-8
                a18.45 18.45 0 015.06-5.94M9.9 4.24A9.12 9.12 0 0112 4c7 0 11 8 11 8
                a18.5 18.5 0 01-2.16 3.19m-6.72-1.07a3 3 0 11-4.24-4.24"/>
            <line x1="1" y1="1" x2="23" y2="23"/>
        </svg>`;

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'pwd-eye-btn';
        btn.innerHTML = SVG_EYE;
        btn.title = 'Show password';
        btn.setAttribute('aria-label', 'Toggle password visibility');
        wrap.appendChild(btn);

        btn.addEventListener('mousedown', function(e) {
            e.preventDefault();
            e.stopPropagation();
            const inp = container.querySelector('input, textarea');
            if (!inp) return;
            if (inp.type === 'password') {
                inp.type = 'text';
                btn.innerHTML = SVG_EYE_OFF;
                btn.title = 'Hide password';
            } else {
                inp.type = 'password';
                btn.innerHTML = SVG_EYE;
                btn.title = 'Show password';
            }
        });
    }

    // Watch for the password container to appear or change visibility
    const obs = new MutationObserver(tryAddEyeToggle);
    obs.observe(document.body, { childList: true, subtree: true });

    // Also try immediately (in case already mounted)
    tryAddEyeToggle();
}
"""
