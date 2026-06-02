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

/* ─── Login View ────────────────────────────────────────────────────────── */
#login-view {
    min-height: 100vh;
    background: #1a1a1a;
    padding: 24px;
}

#login-card {
    background: #242424;
    border: 1px solid #333;
    border-radius: 16px;
    padding: 48px 40px;
    width: 100%;
    max-width: 440px;
    margin: 60px auto;
    box-shadow: 0 20px 60px rgba(0,0,0,0.5);
}

.login-logo {
    text-align: center;
    margin-bottom: 8px;
}

.login-logo .star {
    font-size: 36px;
    color: #d4774e;
    display: block;
    margin-bottom: 6px;
}

.login-logo h1 {
    color: #ececec;
    font-size: 20px;
    font-weight: 600;
    margin: 0 0 4px;
    line-height: 1.3;
}

.login-logo p {
    color: #888;
    font-size: 13px;
    margin: 0 0 32px;
}

.login-error {
    color: #e05252;
    font-size: 13px;
    text-align: center;
    min-height: 20px;
    margin-top: 8px;
}

.login-success {
    color: #4caf50;
    font-size: 13px;
    text-align: center;
    margin-top: 8px;
}

/* Tab styling inside login card */
#login-card .tabs { background: transparent !important; border: none !important; }
#login-card .tab-nav { background: #1e1e1e !important; border-radius: 8px !important; border: 1px solid #333 !important; padding: 3px !important; }
#login-card .tab-nav button { color: #888 !important; border-radius: 6px !important; padding: 7px 20px !important; font-size: 14px !important; border: none !important; background: transparent !important; }
#login-card .tab-nav button.selected { background: #2f2f2f !important; color: #ececec !important; }

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
