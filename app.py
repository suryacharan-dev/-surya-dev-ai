import streamlit as st
from groq import Groq
import google.generativeai as genai
import cohere
import requests
from streamlit_oauth import OAuth2Component
import jwt
import json
import os
import re
from datetime import datetime
from urllib.parse import quote

# ── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(page_title="Surya Dev AI", page_icon="🤖", layout="wide")

# ── API Keys & OAuth Config ──────────────────────────────────────────────────
GROQ_API_KEY    = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY  = st.secrets["GEMINI_API_KEY"]
COHERE_API_KEY  = st.secrets["COHERE_API_KEY"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
CLIENT_ID       = st.secrets["CLIENT_ID"]
CLIENT_SECRET   = st.secrets["CLIENT_SECRET"]

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL     = "https://oauth2.googleapis.com/token"
REDIRECT_URI  = "https://nz5ossng47a243vac5nrti.streamlit.app"
SCOPE         = "openid email profile"
CHATS_FILE    = "chats.json"

groq_client   = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

MODELS = {
    "Llama 3.3 (Groq)":      "llama-3.3-70b-versatile",
    "Gemini 1.5 Flash":      "gemini-1.5-flash",
    "Command R+ (Cohere)":   "command-r-plus",
    "Mistral Small":         "mistral-small-latest",
}

MODEL_ICONS = {
    "Llama 3.3 (Groq)":    "🦙",
    "Gemini 1.5 Flash":    "✨",
    "Command R+ (Cohere)": "🪸",
    "Mistral Small":       "💨",
}

# ── Robot Logo SVG ───────────────────────────────────────────────────────────
ROBOT_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="{size}" height="{size}">
  <rect x="14" y="16" width="36" height="28" rx="8" fill="#d97706"/>
  <circle cx="24" cy="27" r="5" fill="white"/>
  <circle cx="40" cy="27" r="5" fill="white"/>
  <circle cx="25" cy="28" r="2.5" fill="#1a1a1a"/>
  <circle cx="41" cy="28" r="2.5" fill="#1a1a1a"/>
  <circle cx="26" cy="27" r="1" fill="white"/>
  <circle cx="42" cy="27" r="1" fill="white"/>
  <rect x="22" y="36" width="20" height="3" rx="1.5" fill="white" opacity="0.8"/>
  <rect x="25" y="36" width="3" height="3" rx="1" fill="#d97706"/>
  <rect x="31" y="36" width="3" height="3" rx="1" fill="#d97706"/>
  <rect x="37" y="36" width="3" height="3" rx="1" fill="#d97706"/>
  <rect x="30" y="8" width="4" height="10" rx="2" fill="#d97706"/>
  <circle cx="32" cy="7" r="4" fill="#f59e0b"/>
  <circle cx="32" cy="7" r="2" fill="white"/>
  <rect x="7" y="22" width="7" height="12" rx="3" fill="#b45309"/>
  <rect x="50" y="22" width="7" height="12" rx="3" fill="#b45309"/>
  <rect x="27" y="44" width="10" height="6" rx="2" fill="#b45309"/>
  <rect x="16" y="50" width="32" height="10" rx="5" fill="#92400e"/>
  <circle cx="32" cy="55" r="3" fill="#fbbf24"/>
  <circle cx="32" cy="55" r="1.5" fill="white" opacity="0.8"/>
</svg>
"""

def get_robot_logo(size=40):
    return ROBOT_LOGO_SVG.replace("{size}", str(size))

# ── Global CSS (Claude-like UI) ──────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Reset & Base ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #1a1a1a;
        color: #e5e5e5;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
    .stDeployButton { display: none; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #111111 !important;
        border-right: 1px solid #2a2a2a;
        width: 260px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 0;
    }

    /* ── Sidebar header ── */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 18px 16px 14px;
        border-bottom: 1px solid #2a2a2a;
    }
    .sidebar-brand {
        font-size: 15px;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: -0.3px;
    }

    /* ── New chat button ── */
    .new-chat-btn {
        display: flex;
        align-items: center;
        gap: 8px;
        width: calc(100% - 24px);
        margin: 12px 12px 6px;
        padding: 9px 14px;
        background: transparent;
        border: 1px solid #2e2e2e;
        border-radius: 8px;
        color: #d1d1d1;
        font-size: 13.5px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s;
        text-align: left;
    }
    .new-chat-btn:hover {
        background: #1e1e1e;
        border-color: #3a3a3a;
    }

    /* ── Chat history items ── */
    .chat-section-label {
        padding: 10px 16px 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #666;
    }
    .chat-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 7px 14px;
        margin: 1px 6px;
        border-radius: 7px;
        cursor: pointer;
        transition: background 0.12s;
        font-size: 13px;
        color: #aaa;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .chat-item:hover { background: #1e1e1e; color: #ddd; }
    .chat-item.active { background: #242424; color: #fff; }
    .chat-item-title {
        flex: 1;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* ── Top bar ── */
    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 20px;
        border-bottom: 1px solid #2a2a2a;
        background: #1a1a1a;
        position: sticky;
        top: 0;
        z-index: 100;
    }
    .topbar-model {
        font-size: 13.5px;
        font-weight: 500;
        color: #ccc;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .topbar-actions {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .avatar-chip {
        width: 32px; height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #d97706, #f59e0b);
        display: flex; align-items: center; justify-content: center;
        font-size: 13px; font-weight: 700; color: white;
        cursor: pointer;
    }

    /* ── Chat container ── */
    .chat-wrapper {
        max-width: 760px;
        margin: 0 auto;
        padding: 24px 20px 160px;
    }

    /* ── Messages ── */
    .msg-row {
        display: flex;
        gap: 12px;
        margin-bottom: 24px;
        animation: fadeSlide 0.2s ease;
    }
    @keyframes fadeSlide {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .msg-avatar {
        width: 32px; height: 32px;
        border-radius: 50%;
        flex-shrink: 0;
        display: flex; align-items: center; justify-content: center;
        font-size: 14px;
        margin-top: 2px;
    }
    .msg-avatar.user {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white; font-weight: 700; font-size: 13px;
    }
    .msg-avatar.ai {
        background: #2a2a2a;
        border: 1px solid #333;
    }
    .msg-body { flex: 1; min-width: 0; }
    .msg-name {
        font-size: 12.5px;
        font-weight: 600;
        color: #888;
        margin-bottom: 5px;
    }
    .msg-text {
        font-size: 14.5px;
        line-height: 1.65;
        color: #e0e0e0;
    }
    .msg-text p { margin: 0 0 10px; }
    .msg-text p:last-child { margin-bottom: 0; }
    .msg-text code {
        background: #2a2a2a;
        border: 1px solid #333;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 13px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        color: #f0a050;
    }
    .msg-text pre {
        background: #222;
        border: 1px solid #2e2e2e;
        border-radius: 8px;
        padding: 14px 16px;
        overflow-x: auto;
        margin: 10px 0;
    }
    .msg-text pre code {
        background: none;
        border: none;
        padding: 0;
        color: #d4d4d4;
        font-size: 13px;
    }

    /* ── Source pills ── */
    .source-pill {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #222;
        border: 1px solid #333;
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 11.5px;
        color: #888;
        margin: 6px 4px 0 0;
        text-decoration: none;
    }
    .source-pill:hover { background: #2a2a2a; color: #bbb; }

    /* ── Input area (fixed bottom) ── */
    .input-wrapper {
        position: fixed;
        bottom: 0; left: 260px; right: 0;
        padding: 12px 20px 20px;
        background: linear-gradient(to top, #1a1a1a 70%, transparent);
        z-index: 50;
    }
    .input-box {
        max-width: 760px;
        margin: 0 auto;
        background: #262626;
        border: 1px solid #333;
        border-radius: 14px;
        padding: 4px 4px 4px 16px;
        display: flex;
        align-items: flex-end;
        gap: 8px;
        box-shadow: 0 0 0 1px #2e2e2e, 0 8px 32px rgba(0,0,0,0.4);
    }
    .input-box:focus-within {
        border-color: #d97706;
        box-shadow: 0 0 0 3px rgba(217,119,6,0.12), 0 8px 32px rgba(0,0,0,0.4);
    }

    /* ── Streamlit widget overrides ── */
    .stTextArea textarea {
        background: transparent !important;
        border: none !important;
        color: #e5e5e5 !important;
        font-size: 14.5px !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.6 !important;
        resize: none !important;
        box-shadow: none !important;
        padding: 8px 0 !important;
    }
    .stTextArea textarea:focus { box-shadow: none !important; border: none !important; }
    [data-testid="stTextArea"] label { display: none !important; }
    [data-testid="stTextArea"] { border: none !important; background: transparent !important; }

    .stButton button {
        background: #d97706 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: background 0.15s !important;
    }
    .stButton button:hover { background: #b45309 !important; }

    .stSelectbox select, [data-baseweb="select"] {
        background: #222 !important;
        border-color: #333 !important;
        color: #e5e5e5 !important;
        font-size: 13.5px !important;
    }
    [data-baseweb="select"] > div {
        background: #222 !important;
        border-color: #333 !important;
        color: #e5e5e5 !important;
    }

    .stToggle label { color: #aaa !important; font-size: 13px !important; }

    /* ── Welcome screen ── */
    .welcome-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        padding: 40px 20px;
        text-align: center;
    }
    .welcome-logo { margin-bottom: 20px; }
    .welcome-title {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    .welcome-sub {
        font-size: 15px;
        color: #888;
        margin-bottom: 32px;
    }
    .suggestion-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        width: 100%;
        max-width: 560px;
    }
    .suggestion-card {
        background: #222;
        border: 1px solid #2e2e2e;
        border-radius: 10px;
        padding: 14px 16px;
        text-align: left;
        cursor: pointer;
        transition: background 0.15s, border-color 0.15s;
    }
    .suggestion-card:hover { background: #2a2a2a; border-color: #3a3a3a; }
    .suggestion-icon { font-size: 18px; margin-bottom: 6px; }
    .suggestion-title { font-size: 13.5px; font-weight: 600; color: #ddd; margin-bottom: 3px; }
    .suggestion-desc  { font-size: 12px; color: #777; }

    /* ── Auth screen ── */
    .auth-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;
        background: #111;
        padding: 40px 20px;
    }
    .auth-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 40px 44px;
        max-width: 400px;
        width: 100%;
        text-align: center;
    }
    .auth-title {
        font-size: 24px;
        font-weight: 700;
        color: #fff;
        margin: 16px 0 8px;
    }
    .auth-sub {
        font-size: 14px;
        color: #888;
        margin-bottom: 28px;
        line-height: 1.5;
    }

    /* ── Dividers & labels ── */
    hr.divider { border: none; border-top: 1px solid #2a2a2a; margin: 8px 0; }
    .sidebar-section-label {
        padding: 12px 16px 4px;
        font-size: 10.5px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #555;
    }

    /* ── Spinner override ── */
    .stSpinner > div { border-top-color: #d97706 !important; }

    /* ── Search badge ── */
    .search-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(217,119,6,0.12);
        border: 1px solid rgba(217,119,6,0.3);
        border-radius: 20px;
        padding: 2px 8px;
        font-size: 11px;
        color: #d97706;
        margin-bottom: 8px;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: #444; }

    /* ── Mobile ── */
    @media (max-width: 768px) {
        .input-wrapper { left: 0; }
        .suggestion-grid { grid-template-columns: 1fr; }
    }
    </style>
    """, unsafe_allow_html=True)


# ── Live Search ──────────────────────────────────────────────────────────────
def search_wikipedia(query):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("extract"):
                return {
                    "source": "Wikipedia",
                    "title": data.get("title", query),
                    "content": data["extract"][:1000],
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
    except requests.RequestException:
        pass
    return None

def search_duckduckgo(query):
    try:
        url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results = []
            if data.get("Answer"):
                results.append(f"Quick Answer: {data['Answer']}")
            if data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText'][:800]}")
            for fact in (data.get("Infobox") or {}).get("content", [])[:3]:
                lbl, val = fact.get("label", ""), fact.get("value", "")
                if lbl and val:
                    results.append(f"{lbl}: {val}")
            if results:
                return {
                    "source": "DuckDuckGo",
                    "title": data.get("Heading", query),
                    "content": "\n".join(results),
                    "url": data.get("AbstractURL", ""),
                }
    except requests.RequestException:
        pass
    return None

LIVE_KEYWORDS = [
    "latest", "current", "today", "now", "recent", "news", "live",
    "2024", "2025", "2026", "who is", "what is", "when did", "where is",
    "how much", "price", "weather", "score", "stock", "wiki", "wikipedia",
    "tell me about", "search for", "find information", "look up",
    "what happened", "update", "trending", "population", "capital of",
    "president of", "ceo of", "founder of", "born", "died", "history of",
]

def needs_live_search(query):
    q = query.lower()
    return any(k in q for k in LIVE_KEYWORDS)

def get_live_context(query):
    results, seen = [], set()
    ddg = search_duckduckgo(query)
    if ddg and ddg["content"]:
        results.append(ddg)
        seen.add(ddg["content"][:100])
    topic = re.sub(
        r'^(what is|who is|tell me about|search for|find|look up|when did|where is)\s+',
        '', query.lower()
    ).strip()
    wiki = search_wikipedia(topic)
    if wiki and wiki["content"] and wiki["content"][:100] not in seen:
        results.append(wiki)
    return results


# ── AI Response ──────────────────────────────────────────────────────────────
def _build_messages(messages, use_search):
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not (use_search and needs_live_search(last_user)):
        return list(messages), False, []

    live = get_live_context(last_user)
    if not live:
        return list(messages), False, []

    ctx  = "\n\n".join(f"[{r['source']}] {r['title']}:\n{r['content']}" for r in live)
    srcs = "\n".join(f"{r['source']}: {r['url']}" for r in live if r["url"])
    inject = {
        "role": "user",
        "content": (
            "Here is live web information to help answer the next question:\n\n"
            f"--- LIVE WEB CONTEXT ---\n{ctx}\n--- END CONTEXT ---\n\n"
            f"Sources:\n{srcs}\n\n"
            "Answer using this context. Mention the source (Wikipedia/DuckDuckGo) "
            "when using live data. Supplement with your own knowledge if needed.\n\n"
            f"User's question: {last_user}"
        ),
    }
    return [m for m in messages[:-1]] + [inject], True, live

def get_ai_response(messages, model_choice, use_search=True):
    try:
        msgs, used_search, live_sources = _build_messages(messages, use_search)

        if model_choice == "Llama 3.3 (Groq)":
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=msgs)
            text = resp.choices[0].message.content

        elif model_choice == "Gemini 1.5 Flash":
            gmodel  = genai.GenerativeModel("gemini-1.5-flash")
            history = []
            for m in msgs[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})
            # Gemini requires history to start with a user turn
            while history and history[0]["role"] != "user":
                history.pop(0)
            chat = gmodel.start_chat(history=history)
            text = chat.send_message(msgs[-1]["content"]).text

        elif model_choice == "Command R+ (Cohere)":
            cohere_msgs = [
                {"role": "user" if m["role"] == "user" else "assistant",
                 "content": m["content"]}
                for m in msgs
            ]
            resp = cohere_client.chat(model="command-r-plus", messages=cohere_msgs)
            text = resp.message.content[0].text

        elif model_choice == "Mistral Small":
            headers = {
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {"model": "mistral-small-latest", "messages": msgs}
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]

        else:
            return "⚠️ Unknown model selected.", False, []

        return text, used_search, live_sources

    except Exception as e:
        return f"⚠️ Error: {str(e)}", False, []


# ── Persistence ──────────────────────────────────────────────────────────────
def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}

def save_chats(chats):
    try:
        with open(CHATS_FILE, "w") as f:
            json.dump(chats, f)
    except OSError:
        pass

def get_greeting():
    hour = datetime.now().hour
    if hour < 12:   return "Good morning"
    elif hour < 17: return "Good afternoon"
    else:           return "Good evening"

def count_user_messages(chats_dict):
    return sum(
        1
        for chat in chats_dict.values()
        for msg in chat.get("messages", [])
        if msg["role"] == "user"
    )

def persist_chat(all_chats, user_email, title):
    chat_id = st.session_state.current_chat_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.current_chat_id = chat_id
    if user_email not in all_chats:
        all_chats[user_email] = {}
    all_chats[user_email][chat_id] = {
        "title":    title[:40],
        "messages": st.session_state.messages,
        "model":    st.session_state.get("model_choice", list(MODELS.keys())[0]),
        "updated":  datetime.now().isoformat(),
    }
    st.session_state.all_chats = all_chats
    save_chats(all_chats)

def do_signout():
    for key in ["authenticated", "user_info", "messages", "current_chat_id",
                "show_account", "pending_input"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()


# ── Session State Init ───────────────────────────────────────────────────────
def init_state():
    defaults = {
        "authenticated":  False,
        "user_info":      None,
        "messages":       [],
        "current_chat_id": None,
        "all_chats":      load_chats(),
        "model_choice":   list(MODELS.keys())[0],
        "use_search":     True,
        "show_account":   False,
        "pending_input":  "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
inject_css()


# ── OAuth ────────────────────────────────────────────────────────────────────
oauth2 = OAuth2Component(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    authorize_endpoint=AUTHORIZE_URL,
    token_endpoint=TOKEN_URL,
)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    st.markdown(f"""
    <div class="auth-wrapper">
      <div class="auth-card">
        <div style="display:flex;justify-content:center;margin-bottom:4px;">
          {get_robot_logo(56)}
        </div>
        <div class="auth-title">Surya Dev AI</div>
        <div class="auth-sub">
          Sign in with your Google account to access your chats and start chatting.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    result = oauth2.authorize_button(
        name="Continue with Google",
        icon="https://www.google.com/favicon.ico",
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        key="google_oauth",
        use_container_width=True,
    )

    if result and "token" in result:
        try:
            id_token = result["token"].get("id_token", "")
            payload  = jwt.decode(id_token, options={"verify_signature": False})
            st.session_state.user_info      = payload
            st.session_state.authenticated  = True
            # Load this user's chats
            email = payload.get("email", "unknown")
            if email not in st.session_state.all_chats:
                st.session_state.all_chats[email] = {}
            st.rerun()
        except Exception as e:
            st.error(f"Authentication error: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP (Authenticated)
# ══════════════════════════════════════════════════════════════════════════════
user_info  = st.session_state.user_info or {}
user_email = user_info.get("email", "unknown")
user_name  = user_info.get("name", "User")
user_pic   = user_info.get("picture", "")
initials   = "".join(p[0].upper() for p in user_name.split()[:2]) if user_name else "U"
user_chats = st.session_state.all_chats.get(user_email, {})


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown(f"""
    <div class="sidebar-header">
      {get_robot_logo(28)}
      <span class="sidebar-brand">Surya Dev AI</span>
    </div>
    """, unsafe_allow_html=True)

    # New chat
    if st.button("✏️  New chat", use_container_width=True, key="new_chat_btn"):
        st.session_state.messages        = []
        st.session_state.current_chat_id = None
        st.rerun()

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Model selector
    st.markdown('<div class="sidebar-section-label">Model</div>', unsafe_allow_html=True)
    model_keys = list(MODELS.keys())
    chosen_idx = model_keys.index(st.session_state.model_choice) if st.session_state.model_choice in model_keys else 0
    selected_model = st.selectbox(
        "Model", model_keys,
        index=chosen_idx,
        label_visibility="collapsed",
        key="model_select",
    )
    st.session_state.model_choice = selected_model

    # Live search toggle
    st.session_state.use_search = st.toggle(
        "🔍 Live web search",
        value=st.session_state.use_search,
        key="search_toggle",
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Chat history
    if user_chats:
        # Sort by updated time, newest first
        sorted_chats = sorted(
            user_chats.items(),
            key=lambda x: x[1].get("updated", x[0]),
            reverse=True,
        )
        st.markdown('<div class="sidebar-section-label">Recent</div>', unsafe_allow_html=True)
        for cid, chat in sorted_chats[:30]:
            title  = chat.get("title", "Untitled chat")
            active = "active" if cid == st.session_state.current_chat_id else ""
            if st.button(
                f"{'💬' if active else '🗨️'}  {title}",
                key=f"chat_{cid}",
                use_container_width=True,
            ):
                st.session_state.messages        = chat.get("messages", [])
                st.session_state.current_chat_id = cid
                st.session_state.model_choice    = chat.get("model", model_keys[0])
                st.rerun()

    # Bottom: user profile
    st.markdown('<div style="flex:1"></div>', unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])
    with col1:
        if user_pic:
            st.image(user_pic, width=34)
        else:
            st.markdown(f'<div class="avatar-chip" style="width:34px;height:34px;">{initials}</div>',
                        unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="font-size:12.5px;font-weight:600;color:#ddd;line-height:1.2;">{user_name}</div>
        <div style="font-size:11px;color:#666;overflow:hidden;text-overflow:ellipsis;">{user_email}</div>
        """, unsafe_allow_html=True)

    if st.button("Sign out", use_container_width=True, key="signout_btn"):
        do_signout()


# ── Top bar ──────────────────────────────────────────────────────────────────
model_icon = MODEL_ICONS.get(st.session_state.model_choice, "🤖")
st.markdown(f"""
<div class="topbar">
  <div class="topbar-model">
    <span>{model_icon}</span>
    <span>{st.session_state.model_choice}</span>
    {"<span class='search-badge'>🔍 Search on</span>" if st.session_state.use_search else ""}
  </div>
  <div class="topbar-actions">
    <div class="avatar-chip">{initials}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Chat area ────────────────────────────────────────────────────────────────
messages = st.session_state.messages

# Welcome screen (no messages)
if not messages:
    greeting = get_greeting()
    total_msgs = count_user_messages(st.session_state.all_chats)
    name_first = user_name.split()[0] if user_name else "there"

    st.markdown(f"""
    <div class="chat-wrapper">
      <div class="welcome-wrapper">
        <div class="welcome-logo">{get_robot_logo(52)}</div>
        <div class="welcome-title">{greeting}, {name_first}!</div>
        <div class="welcome-sub">
          How can I help you today? Choose a suggestion or type your message below.
        </div>
        <div class="suggestion-grid">
          <div class="suggestion-card" onclick="document.querySelector('textarea').value='Explain quantum computing simply'">
            <div class="suggestion-icon">⚛️</div>
            <div class="suggestion-title">Explain a concept</div>
            <div class="suggestion-desc">Quantum computing in simple terms</div>
          </div>
          <div class="suggestion-card" onclick="document.querySelector('textarea').value='Write a Python web scraper'">
            <div class="suggestion-icon">💻</div>
            <div class="suggestion-title">Write code</div>
            <div class="suggestion-desc">Python, JS, SQL and more</div>
          </div>
          <div class="suggestion-card" onclick="document.querySelector('textarea').value='What is happening in AI news today?'">
            <div class="suggestion-icon">🌐</div>
            <div class="suggestion-title">Search the web</div>
            <div class="suggestion-desc">Latest news and current events</div>
          </div>
          <div class="suggestion-card" onclick="document.querySelector('textarea').value='Help me brainstorm startup ideas in EdTech'">
            <div class="suggestion-icon">💡</div>
            <div class="suggestion-title">Brainstorm ideas</div>
            <div class="suggestion-desc">Creative thinking and planning</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Render messages
    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            st.markdown(f"""
            <div class="msg-row">
              <div class="msg-avatar user">{initials}</div>
              <div class="msg-body">
                <div class="msg-name">You</div>
                <div class="msg-text"><p>{content}</p></div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Format markdown-like content
            source_pills = ""
            if sources:
                for s in sources:
                    if s.get("url"):
                        source_pills += f'<a class="source-pill" href="{s["url"]}" target="_blank">🔗 {s["source"]}</a>'

            st.markdown(f"""
            <div class="msg-row">
              <div class="msg-avatar ai">{get_robot_logo(22)}</div>
              <div class="msg-body">
                <div class="msg-name">{MODEL_ICONS.get(msg.get("model",""), "🤖")} {msg.get("model", "AI")}</div>
                {"<div class='search-badge'>🔍 Used live search</div>" if msg.get("used_search") else ""}
                <div class="msg-text">{_md_to_html(content)}</div>
                {source_pills}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Simple Markdown → HTML converter ────────────────────────────────────────
def _md_to_html(text: str) -> str:
    """Lightweight md→html: code blocks, inline code, bold, italic, lists."""
    import html as _html

    # Escape HTML first
    text = _html.escape(text)

    # Fenced code blocks
    def replace_code_block(m):
        lang    = m.group(1).strip() if m.group(1) else ""
        code    = m.group(2)
        return f'<pre><code class="language-{lang}">{code}</code></pre>'

    text = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)

    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # Bold & italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*',     r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*',         r'<em>\1</em>', text)

    # Headers
    text = re.sub(r'^#{3}\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^#{2}\s+(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^#{1}\s+(.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    # Bullet lists
    lines     = text.split('\n')
    out       = []
    in_ul     = False
    for line in lines:
        if re.match(r'^[\*\-]\s+', line):
            if not in_ul:
                out.append('<ul style="margin:6px 0 6px 20px;padding:0;">')
                in_ul = True
            item = re.sub(r'^[\*\-]\s+', '', line)
            out.append(f'<li style="margin:3px 0;">{item}</li>')
        else:
            if in_ul:
                out.append('</ul>')
                in_ul = False
            out.append(line)
    if in_ul:
        out.append('</ul>')
    text = '\n'.join(out)

    # Paragraphs
    paragraphs = re.split(r'\n{2,}', text)
    html_parts = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if para.startswith(('<h', '<ul', '<pre', '<ol')):
            html_parts.append(para)
        else:
            para = para.replace('\n', '<br>')
            html_parts.append(f'<p>{para}</p>')
    return ''.join(html_parts)


# ── Input area ───────────────────────────────────────────────────────────────
with st.container():
    st.markdown('<div class="input-wrapper"><div class="input-box">', unsafe_allow_html=True)
    col_input, col_send = st.columns([9, 1])
    with col_input:
        user_input = st.text_area(
            "Message",
            placeholder="Message Surya Dev AI…",
            key="chat_input",
            height=52,
            label_visibility="collapsed",
        )
    with col_send:
        send_clicked = st.button("➤", key="send_btn", use_container_width=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

# Handle Enter key (Ctrl+Enter) via keyboard shortcut hint
st.markdown("""
<script>
const ta = window.parent.document.querySelector('textarea[data-testid="stTextArea"]');
if(ta){
  ta.addEventListener('keydown', function(e){
    if((e.ctrlKey || e.metaKey) && e.key === 'Enter'){
      const btn = window.parent.document.querySelector('[data-testid="baseButton-secondary"]');
      if(btn) btn.click();
    }
  });
}
</script>
""", unsafe_allow_html=True)

# ── Process send ─────────────────────────────────────────────────────────────
if send_clicked and user_input and user_input.strip():
    prompt = user_input.strip()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response
    with st.spinner("Thinking…"):
        response_text, used_search, live_sources = get_ai_response(
            st.session_state.messages,
            st.session_state.model_choice,
            st.session_state.use_search,
        )

    # Add assistant message
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    response_text,
        "model":      st.session_state.model_choice,
        "used_search": used_search,
        "sources":    live_sources,
    })

    # Persist
    title     = prompt[:40]
    all_chats = st.session_state.all_chats
    if user_email not in all_chats:
        all_chats[user_email] = {}
    persist_chat(all_chats, user_email, title)

    st.rerun()
 
 
