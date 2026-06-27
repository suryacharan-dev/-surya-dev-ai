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
import base64
from datetime import datetime
from urllib.parse import quote

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY    = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY  = st.secrets["GEMINI_API_KEY"]
COHERE_API_KEY  = st.secrets["COHERE_API_KEY"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
CLIENT_ID       = st.secrets["CLIENT_ID"]
CLIENT_SECRET   = st.secrets["CLIENT_SECRET"]
AUTHORIZE_URL   = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL       = "https://oauth2.googleapis.com/token"
REDIRECT_URI    = "https://nz5ossng47a243vac5nrti.streamlit.app"
SCOPE           = "openid email profile"
CHATS_FILE      = "chats.json"

groq_client   = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

# ── Robot SVG (base64) ────────────────────────────────────────────────────────
_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <rect x="14" y="16" width="36" height="28" rx="8" fill="#10a37f"/>
  <circle cx="24" cy="27" r="5" fill="white"/><circle cx="40" cy="27" r="5" fill="white"/>
  <circle cx="25" cy="28" r="2.5" fill="#111"/><circle cx="41" cy="28" r="2.5" fill="#111"/>
  <circle cx="26" cy="27" r="1" fill="white"/><circle cx="42" cy="27" r="1" fill="white"/>
  <rect x="22" y="36" width="20" height="3" rx="1.5" fill="white" opacity="0.9"/>
  <rect x="25" y="36" width="3" height="3" rx="1" fill="#10a37f"/>
  <rect x="31" y="36" width="3" height="3" rx="1" fill="#10a37f"/>
  <rect x="37" y="36" width="3" height="3" rx="1" fill="#10a37f"/>
  <rect x="30" y="8" width="4" height="10" rx="2" fill="#10a37f"/>
  <circle cx="32" cy="7" r="4" fill="#1dc8a0"/><circle cx="32" cy="7" r="2" fill="white"/>
  <rect x="7" y="22" width="7" height="12" rx="3" fill="#0d8a6a"/>
  <rect x="50" y="22" width="7" height="12" rx="3" fill="#0d8a6a"/>
  <rect x="27" y="44" width="10" height="6" rx="2" fill="#0d8a6a"/>
  <rect x="16" y="50" width="32" height="10" rx="5" fill="#0a6b52"/>
  <circle cx="32" cy="55" r="3" fill="#1dc8a0"/>
  <circle cx="32" cy="55" r="1.5" fill="white" opacity="0.8"/>
</svg>"""
_B64 = base64.b64encode(_SVG.encode()).decode()
def robot_img(size):
    return f'<img src="data:image/svg+xml;base64,{_B64}" width="{size}" height="{size}" style="display:inline-block;vertical-align:middle;"/>'

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = {
    "GPT-4o (Llama 3.3)":   {"icon": "⚡", "color": "#10a37f", "desc": "Fast & powerful"},
    "Gemini 1.5 Flash":      {"icon": "✦", "color": "#4285f4", "desc": "Google AI"},
    "Command R (Cohere)":    {"icon": "◈", "color": "#d97706", "desc": "Great for analysis"},
    "Mistral Small":         {"icon": "◆", "color": "#7c3aed", "desc": "Efficient & smart"},
}

# ── Storage ───────────────────────────────────────────────────────────────────
def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE) as f:
            return json.load(f)
    return {}

def save_chats(c):
    with open(CHATS_FILE, "w") as f:
        json.dump(c, f)

def msg_count(chats):
    return sum(1 for ch in chats.values() for m in ch.get("messages",[]) if m["role"]=="user")

# ── Live search ───────────────────────────────────────────────────────────────
def wiki_search(q):
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(q)}", timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get("extract"):
                return {"src":"Wikipedia","title":d.get("title",q),"text":d["extract"][:900],"url":d.get("content_urls",{}).get("desktop",{}).get("page","")}
    except: pass
    return None

def ddg_search(q):
    try:
        r = requests.get(f"https://api.duckduckgo.com/?q={quote(q)}&format=json&no_html=1&skip_disambig=1", timeout=5)
        if r.status_code == 200:
            d = r.json()
            parts = []
            if d.get("Answer"): parts.append(d["Answer"])
            if d.get("AbstractText"): parts.append(d["AbstractText"][:700])
            if parts:
                return {"src":"DuckDuckGo","title":d.get("Heading",q),"text":"\n".join(parts),"url":d.get("AbstractURL","")}
    except: pass
    return None

LIVE_KEYWORDS = ["latest","current","today","now","recent","news","live","2024","2025","2026",
    "who is","what is","when did","where is","how much","price","weather","score","stock",
    "wikipedia","tell me about","search","look up","what happened","update","trending",
    "population","capital of","president of","ceo of","founder of","born","died","history of"]

def needs_search(q):
    return any(k in q.lower() for k in LIVE_KEYWORDS)

def get_context(q):
    results = []
    ddg = ddg_search(q)
    if ddg: results.append(ddg)
    topic = re.sub(r'^(what is|who is|tell me about|search for|find|look up|when did|where is)\s+','',q.lower()).strip()
    wk = wiki_search(topic)
    if wk: results.append(wk)
    return results

# ── AI Response ───────────────────────────────────────────────────────────────
def get_response(messages, model, use_live=True):
    try:
        last = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
        aug = list(messages)
        searched = False
        if use_live and needs_search(last):
            ctx = get_context(last)
            if ctx:
                searched = True
                parts = [f"[{r['src']}] {r['title']}:\n{r['text']}" for r in ctx]
                srcs  = [f"{r['src']}: {r['url']}" for r in ctx if r['url']]
                inject = {"role":"user","content":
                    f"Live web context:\n---\n{chr(10).join(parts)}\n---\nSources: {', '.join(srcs)}\n\nAnswer using this context (cite source). Question: {last}"}
                aug = list(messages[:-1]) + [inject]

        if model == "GPT-4o (Llama 3.3)":
            r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=aug)
            return r.choices[0].message.content, searched
        elif model == "Gemini 1.5 Flash":
            gm = genai.GenerativeModel("gemini-1.5-flash")
            hist = [{"role":"user" if m["role"]=="user" else "model","parts":[m["content"]]} for m in aug[:-1]]
            ch = gm.start_chat(history=hist)
            return ch.send_message(aug[-1]["content"]).text, searched
        elif model == "Command R (Cohere)":
            cm = [{"role":"user" if m["role"]=="user" else "assistant","content":m["content"]} for m in aug]
            r = cohere_client.chat(model="command-r", messages=cm)
            return r.message.content[0].text, searched
        elif model == "Mistral Small":
            h = {"Authorization":f"Bearer {MISTRAL_API_KEY}","Content-Type":"application/json"}
            r = requests.post("https://api.mistral.ai/v1/chat/completions",
                headers=h, json={"model":"mistral-small-latest","messages":aug}, timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], searched
    except Exception as e:
        return f"⚠️ Error: {str(e)}", False

def greet():
    h = datetime.now().hour
    return "Good morning" if h<12 else "Good afternoon" if h<17 else "Good evening"

# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Surya Dev AI", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { font-family: 'Inter', sans-serif !important; }

/* ── App background ── */
.stApp { background: #212121 !important; color: #ececec !important; }
#MainMenu, footer, header { visibility: hidden !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid #2e2e2e !important;
    width: 260px !important;
    min-width: 260px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 !important; }

[data-testid="collapsedControl"] {
    display: flex !important; visibility: visible !important; opacity: 1 !important;
    background: #171717 !important; border-radius: 0 8px 8px 0 !important;
    border: 1px solid #2e2e2e !important; color: #ececec !important;
}

/* ── Sidebar buttons ── */
.stButton > button {
    background: transparent !important; color: #c5c5c5 !important;
    border: none !important; border-radius: 8px !important;
    width: 100% !important; text-align: left !important;
    padding: 9px 12px !important; font-size: 13.5px !important;
    font-weight: 400 !important; transition: background 0.15s !important;
    white-space: nowrap !important; overflow: hidden !important;
    text-overflow: ellipsis !important;
}
.stButton > button:hover {
    background: #2a2a2a !important; color: #ffffff !important; border: none !important;
}

/* ── New Chat button ── */
div[data-new-chat="true"] button {
    background: #2a2a2a !important; color: #ececec !important;
    border: 1px solid #3a3a3a !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
}

/* ── Main content area ── */
.main .block-container {
    padding: 0 !important; max-width: 100% !important;
    background: #212121 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important; border: none !important;
    padding: 16px 0 !important; max-width: 760px !important;
    margin: 0 auto !important; width: 100% !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: transparent !important;
}
[data-testid="stChatMessageContent"] {
    background: transparent !important; color: #ececec !important;
    font-size: 15px !important; line-height: 1.7 !important;
    overflow: visible !important; white-space: normal !important;
    word-wrap: break-word !important; max-width: 100% !important;
}
[data-testid="stChatMessageContent"] p {
    color: #ececec !important; overflow: visible !important;
    white-space: normal !important; margin-bottom: 8px !important;
}
[data-testid="stChatMessageContent"] code {
    background: #2d2d2d !important; color: #ececec !important;
    border-radius: 4px !important; padding: 2px 6px !important;
}
[data-testid="stChatMessageContent"] pre {
    background: #1a1a1a !important; border: 1px solid #3a3a3a !important;
    border-radius: 8px !important; padding: 16px !important;
    overflow-x: auto !important;
}

/* ── Avatar ── */
[data-testid="stChatMessage"] .stChatMessageAvatar {
    width: 32px !important; height: 32px !important;
    min-width: 32px !important; border-radius: 6px !important;
}

/* ── Bottom input area ── */
[data-testid="stBottom"] {
    background: #212121 !important;
    border-top: none !important; padding: 12px 0 16px 0 !important;
}
[data-testid="stBottom"] > div { background: #212121 !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #2f2f2f !important;
    border: 1px solid #404040 !important;
    border-radius: 16px !important;
    max-width: 760px !important; margin: 0 auto !important;
}
[data-testid="stChatInput"] textarea {
    background: #2f2f2f !important; color: #ececec !important;
    border: none !important; border-radius: 16px !important;
    font-size: 15px !important; padding: 14px 18px !important;
    resize: none !important; box-shadow: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #8e8ea0 !important; }
[data-testid="stChatInput"] textarea:focus { outline: none !important; box-shadow: none !important; }
[data-testid="stChatInput"] button {
    background: #10a37f !important; border-radius: 10px !important;
    color: white !important; border: none !important;
}
[data-testid="stChatInput"] button:hover { background: #0d8a6a !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #2f2f2f !important; color: #ececec !important;
    border: 1px solid #3a3a3a !important; border-radius: 8px !important;
    font-size: 13px !important;
}

/* ── Toggle ── */
.stToggle label { color: #c5c5c5 !important; font-size: 13px !important; }

/* ── Divider ── */
hr { border-color: #2e2e2e !important; margin: 8px 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 3px; }

/* ── Model badge ── */
.m-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 600; padding: 2px 9px;
    border-radius: 20px; margin-bottom: 6px;
}
.m-badge.groq   { background: #0d3b30; color: #1dc8a0; border: 1px solid #0d8a6a; }
.m-badge.gemini { background: #0d2060; color: #7baaf7; border: 1px solid #2d5dbf; }
.m-badge.cohere { background: #3d2200; color: #f59e0b; border: 1px solid #92400e; }
.m-badge.mistral{ background: #2d1060; color: #c084fc; border: 1px solid #7c3aed; }
.m-badge.live   { background: #0d3020; color: #4ade80; border: 1px solid #166534; margin-left:4px; }

/* ── Sidebar section label ── */
.slabel {
    font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase;
    letter-spacing: 0.08em; padding: 6px 14px 3px 14px;
}

/* ── Account page ── */
.acc-header {
    background: linear-gradient(135deg,#1a3a30,#212121);
    border: 1px solid #2a5a42; border-radius: 16px;
    padding: 28px; display: flex; align-items: center; gap: 18px; margin-bottom: 20px;
}
.acc-name  { font-size: 20px; font-weight: 700; color: #ececec; }
.acc-email { font-size: 13px; color: #888; margin-top: 3px; }
.acc-badge { display: inline-flex; align-items: center; gap: 5px; background: #10a37f;
    color: white; font-size: 11px; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; margin-top: 8px; }
.stat-box  { background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 12px;
    padding: 18px; text-align: center; }
.stat-n    { font-size: 26px; font-weight: 700; color: #10a37f; }
.stat-l    { font-size: 12px; color: #888; margin-top: 4px; }
.info-box  { background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 12px;
    padding: 18px 22px; margin-bottom: 14px; }
.info-title{ font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 14px; }
.info-row  { display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0; border-bottom: 1px solid #333; }
.info-row:last-child { border-bottom: none; }
.info-k    { font-size: 13px; color: #888; }
.info-v    { font-size: 13px; color: #ececec; font-weight: 600; }

/* ── Login page ── */
.login-wrap {
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; justify-content: center; background: #212121;
}
.login-card {
    background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 18px;
    padding: 36px 32px; width: 100%; box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.feat-item { display: flex; align-items: center; gap: 10px;
    font-size: 13px; color: #a0a0a0; margin-bottom: 10px; }
.feat-icon { width: 28px; height: 28px; border-radius: 7px;
    display: flex; align-items: center; justify-content: center; font-size: 14px; flex-shrink: 0; }

/* ── Model selector card ── */
.model-card {
    background: #2a2a2a; border: 2px solid #3a3a3a; border-radius: 10px;
    padding: 10px 14px; cursor: pointer; transition: all 0.15s;
    display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
}
.model-card.active { border-color: #10a37f !important; background: #0d2d22 !important; }
.model-card:hover  { border-color: #555 !important; background: #2f2f2f !important; }

/* ── Suggestion chips ── */
.chip-btn button {
    background: #2a2a2a !important; color: #c5c5c5 !important;
    border: 1px solid #3a3a3a !important; border-radius: 12px !important;
    font-size: 13px !important; text-align: left !important;
    padding: 12px 16px !important; height: auto !important;
    white-space: normal !important; line-height: 1.4 !important;
}
.chip-btn button:hover {
    background: #333 !important; border-color: #10a37f !important;
    color: #fff !important;
}

/* ── Profile area ── */
.prof-card {
    display: flex; align-items: center; gap: 9px;
    padding: 9px 12px; border-radius: 8px; background: #2a2a2a;
    border: 1px solid #333; margin: 6px 0;
}
.prof-name  { font-size: 13px; font-weight: 600; color: #ececec; }
.prof-email { font-size: 11px; color: #777; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "auth": False, "user": None, "msgs": [], "chat_id": None,
    "chats": load_chats(), "model": "GPT-4o (Llama 3.3)",
    "page": "chat",   # chat | account | models
    "live": True, "login_time": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.auth:
    c1, c2, c3 = st.columns([1, 1.1, 1])
    with c2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)

        # Logo
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:24px;">
          <div style="display:inline-flex;align-items:center;justify-content:center;
              width:76px;height:76px;background:#1a3a30;border-radius:20px;
              border:2px solid #10a37f;margin-bottom:14px;
              box-shadow:0 4px 20px rgba(16,163,127,0.3);">
            {robot_img(52)}
          </div>
          <h1 style="font-size:1.9rem;font-weight:700;color:#ececec;margin:0 0 5px;letter-spacing:-0.5px;">
            Surya Dev AI
          </h1>
          <p style="color:#888;font-size:14px;margin:0;">Your intelligent AI assistant</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="login-card">
          <p style="font-size:16px;font-weight:600;color:#ececec;text-align:center;margin-bottom:4px;">
            Welcome back
          </p>
          <p style="font-size:13px;color:#888;text-align:center;margin-bottom:22px;">
            Sign in to continue to Surya Dev AI
          </p>
          <div class="feat-item"><div class="feat-icon" style="background:#0d3b30;">🤖</div>
            4 powerful AI models — Llama, Gemini, Cohere, Mistral</div>
          <div class="feat-item"><div class="feat-icon" style="background:#0d2060;">🌐</div>
            Live web search from Wikipedia &amp; DuckDuckGo</div>
          <div class="feat-item"><div class="feat-icon" style="background:#2d1060;">💬</div>
            Full chat history saved across sessions</div>
          <div class="feat-item"><div class="feat-icon" style="background:#3d2200;">🔒</div>
            Secure Google login</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, TOKEN_URL, TOKEN_URL)
        result = oauth2.authorize_button("Continue with Google", redirect_uri=REDIRECT_URI,
            scope=SCOPE, key="google",
            extras_params={"prompt":"consent","access_type":"offline"},
            use_container_width=True)
        st.markdown('<p style="text-align:center;font-size:11px;color:#555;margin-top:14px;">By continuing you agree to our terms of use</p>', unsafe_allow_html=True)

        if result and "token" in result:
            payload = jwt.decode(result["token"]["id_token"], options={"verify_signature":False})
            st.session_state.auth = True
            st.session_state.user = payload
            st.session_state.login_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATED PAGES
# ══════════════════════════════════════════════════════════════════════════════
else:
    u      = st.session_state.user
    email  = u.get("email","")
    name   = u.get("name","User") or "User"
    fname  = name.split()[0]
    pic    = u.get("picture","")
    init   = name[0].upper()
    chats  = st.session_state.chats
    if email not in chats: chats[email] = {}
    uchats = chats[email]

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)

        # Logo + title
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:9px;padding:4px 14px 14px;">
          <div style="width:30px;height:30px;background:#1a3a30;border-radius:8px;
              border:1px solid #10a37f;display:flex;align-items:center;justify-content:center;">
            {robot_img(22)}
          </div>
          <span style="font-size:15px;font-weight:700;color:#ececec;letter-spacing:-0.3px;">Surya Dev AI</span>
        </div>
        """, unsafe_allow_html=True)

        # New Chat
        if st.button("✏️   New chat", key="new_chat", use_container_width=True):
            st.session_state.msgs = []
            st.session_state.chat_id = None
            st.session_state.page = "chat"
            st.rerun()

        st.markdown("<div class='slabel'>Model</div>", unsafe_allow_html=True)
        cur_model_idx = list(MODELS.keys()).index(st.session_state.model)
        chosen = st.selectbox("Model", list(MODELS.keys()),
                              index=cur_model_idx, label_visibility="collapsed")
        if chosen != st.session_state.model:
            st.session_state.model = chosen
            st.rerun()

        # Live search toggle
        live_on = st.toggle("🌐 Live Web Search", value=st.session_state.live)
        if live_on != st.session_state.live:
            st.session_state.live = live_on
            st.rerun()

        st.divider()

        # Chat history
        if uchats:
            st.markdown("<div class='slabel'>Recent chats</div>", unsafe_allow_html=True)
            for cid, cdata in sorted(uchats.items(), reverse=True)[:20]:
                title = cdata.get("title", "Untitled")[:30]
                label = f"{'💬'} {title}"
                if st.button(label, key=f"h_{cid}", use_container_width=True):
                    st.session_state.chat_id = cid
                    st.session_state.msgs = cdata["messages"]
                    st.session_state.page = "chat"
                    st.rerun()

        # Push profile to bottom
        st.markdown("<br>" * 5, unsafe_allow_html=True)
        st.divider()

        # Profile
        av = f'<img src="{pic}" style="width:32px;height:32px;border-radius:50%;object-fit:cover;"/>' if pic else \
             f'<div style="width:32px;height:32px;border-radius:50%;background:#10a37f;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:13px;">{init}</div>'

        st.markdown(f"""
        <div class="prof-card">
          {av}
          <div style="flex:1;min-width:0;">
            <div class="prof-name">{name}</div>
            <div class="prof-email" title="{email}">{email[:26]}{'…' if len(email)>26 else ''}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👤 Account", use_container_width=True, key="btn_acc"):
                st.session_state.page = "account"
                st.rerun()
        with col2:
            if st.button("🚪 Sign out", use_container_width=True, key="btn_out"):
                for k in ["auth","user","msgs","chat_id","page"]:
                    st.session_state[k] = defaults[k]
                st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  ACCOUNT PAGE
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.page == "account":
        _, cc, _ = st.columns([1,3,1])
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back to chat", key="back"):
                st.session_state.page = "chat"
                st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)

            av_lg = f'<img src="{pic}" style="width:68px;height:68px;border-radius:50%;border:3px solid #10a37f;object-fit:cover;"/>' if pic else \
                    f'<div style="width:68px;height:68px;border-radius:50%;background:#10a37f;display:flex;align-items:center;justify-content:center;font-size:26px;color:white;font-weight:700;flex-shrink:0;">{init}</div>'

            tc = len(uchats); tm = msg_count(uchats)
            lt = st.session_state.login_time or datetime.now().strftime("%d %b %Y, %I:%M %p")

            st.markdown(f"""
            <div class="acc-header">
              {av_lg}
              <div>
                <div class="acc-name">{name}</div>
                <div class="acc-email">{email}</div>
                <div class="acc-badge">🤖 Surya Dev AI Member</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Stats
            s1,s2,s3,s4 = st.columns(4)
            for col, num, lbl in [(s1,tc,"Chats"),(s2,tm,"Messages"),(s3,4,"AI Models"),(s4,"ON" if live_on else "OFF","Live Search")]:
                col.markdown(f'<div class="stat-box"><div class="stat-n">{num}</div><div class="stat-l">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Account info
            st.markdown(f"""
            <div class="info-box">
              <div class="info-title">Account Details</div>
              <div class="info-row"><span class="info-k">Full Name</span><span class="info-v">{name}</span></div>
              <div class="info-row"><span class="info-k">Email</span><span class="info-v">{email}</span></div>
              <div class="info-row"><span class="info-k">Login Provider</span><span class="info-v">🔵 Google OAuth</span></div>
              <div class="info-row"><span class="info-k">Session Started</span><span class="info-v">{lt}</span></div>
              <div class="info-row"><span class="info-k">Status</span><span class="info-v" style="color:#10a37f;">✅ Active</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Model & preferences
            mdesc = MODELS[st.session_state.model]
            st.markdown(f"""
            <div class="info-box">
              <div class="info-title">Preferences</div>
              <div class="info-row"><span class="info-k">Active Model</span>
                <span class="info-v">{mdesc['icon']} {st.session_state.model}</span></div>
              <div class="info-row"><span class="info-k">Live Search</span>
                <span class="info-v" style="color:#10a37f;">{"✅ On" if live_on else "❌ Off"}</span></div>
              <div class="info-row"><span class="info-k">Search Sources</span>
                <span class="info-v">Wikipedia + DuckDuckGo</span></div>
              <div class="info-row"><span class="info-k">Theme</span>
                <span class="info-v">🌙 Dark (ChatGPT style)</span></div>
            </div>
            """, unsafe_allow_html=True)

            # All 4 models overview
            st.markdown('<div class="info-box"><div class="info-title">Available AI Models</div>', unsafe_allow_html=True)
            for mname, minfo in MODELS.items():
                active = "✅ Active" if mname == st.session_state.model else ""
                st.markdown(f"""
                <div class="info-row">
                  <span class="info-k">{minfo['icon']} {mname}</span>
                  <span class="info-v" style="font-size:12px;color:{'#10a37f' if active else '#666'};">
                    {active if active else minfo['desc']}
                  </span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            da, db = st.columns(2)
            with da:
                if st.button("🗑️ Clear all chats", use_container_width=True, key="clr"):
                    chats[email] = {}
                    st.session_state.chats = chats
                    st.session_state.msgs = []
                    st.session_state.chat_id = None
                    save_chats(chats)
                    st.success("All chats cleared!")
                    st.rerun()
            with db:
                if st.button("🚪 Sign out", use_container_width=True, key="so2"):
                    for k in ["auth","user","msgs","chat_id","page"]:
                        st.session_state[k] = defaults[k]
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  CHAT PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.page == "chat":

        # Model colour for badge
        badge_class = {"GPT-4o (Llama 3.3)":"groq","Gemini 1.5 Flash":"gemini",
                       "Command R (Cohere)":"cohere","Mistral Small":"mistral"}
        mdl = st.session_state.model
        bcls = badge_class.get(mdl,"groq")
        mico = MODELS[mdl]["icon"]

        # Centre column for chat
        _, cc, _ = st.columns([1,4,1])
        with cc:

            # ── Welcome screen ────────────────────────────────────────────────
            if not st.session_state.msgs:
                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="text-align:center;margin-bottom:32px;">
                  <div style="display:inline-flex;align-items:center;justify-content:center;
                      width:68px;height:68px;background:#1a3a30;border-radius:18px;
                      border:2px solid #10a37f;margin-bottom:14px;
                      box-shadow:0 4px 20px rgba(16,163,127,0.25);">
                    {robot_img(46)}
                  </div>
                  <h1 style="font-size:2rem;font-weight:700;color:#ececec;margin:0 0 6px;letter-spacing:-0.5px;">
                    {greet()}, {fname}!
                  </h1>
                  <p style="color:#888;font-size:15px;margin:0 0 10px;">
                    How can I help you today?
                  </p>
                  <span style="display:inline-flex;align-items:center;gap:5px;
                      background:#0d2d22;color:#4ade80;border:1px solid #166534;
                      font-size:12px;font-weight:600;padding:3px 12px;border-radius:20px;">
                    🌐 Live Search {"ON" if live_on else "OFF"} &nbsp;·&nbsp;
                    {mico} {mdl}
                  </span>
                </div>
                """, unsafe_allow_html=True)

                # Suggestion chips
                suggestions = [
                    ("What's happening in AI news today?",      "🌐"),
                    ("Explain quantum computing simply",         "⚛️"),
                    ("Write a Python web scraper",               "🐍"),
                    ("Give me a creative startup idea for 2025", "💡"),
                    ("What is the capital of Australia?",        "🗺️"),
                    ("Help me write a professional email",       "✉️"),
                ]
                c1, c2 = st.columns(2)
                for i,(txt,ico) in enumerate(suggestions):
                    col = c1 if i%2==0 else c2
                    with col:
                        st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                        if st.button(f"{ico}  {txt}", use_container_width=True, key=f"sug{i}"):
                            st.session_state.msgs.append({"role":"user","content":txt})
                            with st.spinner("Thinking…"):
                                reply, searched = get_response(st.session_state.msgs, mdl, live_on)
                            st.session_state.msgs.append({"role":"assistant","content":reply,"searched":searched})
                            cid = datetime.now().strftime("%Y%m%d%H%M%S%f")
                            st.session_state.chat_id = cid
                            chats[email][cid] = {"title":txt[:40],"messages":st.session_state.msgs}
                            st.session_state.chats = chats
                            save_chats(chats)
                            st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

            # ── Messages ─────────────────────────────────────────────────────
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                for i, msg in enumerate(st.session_state.msgs):
                    with st.chat_message(msg["role"]):
                        if msg["role"] == "assistant":
                            searched = msg.get("searched", False)
                            badge = f'<span class="m-badge {bcls}">{mico} {mdl}</span>'
                            if searched:
                                badge += ' <span class="m-badge live">🌐 Live</span>'
                            st.markdown(badge, unsafe_allow_html=True)
                        st.markdown(msg["content"])

        # ── Chat input (full width at bottom) ────────────────────────────────
        if prompt := st.chat_input(f"Message {mdl}…"):
            st.session_state.msgs.append({"role":"user","content":prompt})
            is_live = needs_search(prompt) and live_on
            spin_txt = "🔍 Searching & thinking…" if is_live else "💭 Thinking…"
            with st.spinner(spin_txt):
                reply, searched = get_response(st.session_state.msgs, mdl, live_on)
            st.session_state.msgs.append({"role":"assistant","content":reply,"searched":searched})
            cid = st.session_state.chat_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
            st.session_state.chat_id = cid
            chats[email][cid] = {
                "title": st.session_state.msgs[0]["content"][:40],
                "messages": st.session_state.msgs
            }
            st.session_state.chats = chats
            save_chats(chats)
            st.rerun()
