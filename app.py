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

# ── Robot SVG logo ────────────────────────────────────────────────────────────
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
    "Llama 3.3 (Groq)":    {"icon": "⚡", "tag": "groq"},
    "Gemini 1.5 Flash":    {"icon": "✦",  "tag": "gemini"},
    "Command R (Cohere)":  {"icon": "◈",  "tag": "cohere"},
    "Mistral Small":       {"icon": "◆",  "tag": "mistral"},
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

def total_msgs(chats):
    return sum(1 for ch in chats.values() for m in ch.get("messages",[]) if m["role"]=="user")

# ── Live search ───────────────────────────────────────────────────────────────
def wiki_search(q):
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(q)}", timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get("extract"):
                return d["extract"][:900]
    except: pass
    return ""

def ddg_search(q):
    try:
        r = requests.get(f"https://api.duckduckgo.com/?q={quote(q)}&format=json&no_html=1&skip_disambig=1", timeout=5)
        if r.status_code == 200:
            d = r.json()
            parts = []
            if d.get("Answer"): parts.append(d["Answer"])
            if d.get("AbstractText"): parts.append(d["AbstractText"][:700])
            return "\n".join(parts)
    except: pass
    return ""

LIVE_KW = ["latest","current","today","now","recent","news","live","2024","2025","2026",
    "who is","what is","when did","where is","how much","price","weather","score",
    "wikipedia","tell me about","search","look up","what happened","update","trending",
    "population","capital of","president of","ceo of","founder of","born","died","history of"]

def needs_search(q): return any(k in q.lower() for k in LIVE_KW)

# ── AI Response ───────────────────────────────────────────────────────────────
def get_response(messages, model, use_live=True):
    try:
        last = next((m["content"] for m in reversed(messages) if m["role"]=="user"), "")
        aug = list(messages)
        searched = False

        if use_live and needs_search(last):
            topic = re.sub(r'^(what is|who is|tell me about|search|find|look up|when did|where is)\s+','',last.lower()).strip()
            ctx = ddg_search(last) + "\n" + wiki_search(topic)
            if ctx.strip():
                searched = True
                inject = {"role":"user","content":
                    f"Live web info:\n---\n{ctx[:1500]}\n---\nUse this to answer. Question: {last}"}
                aug = list(messages[:-1]) + [inject]

        if model == "Llama 3.3 (Groq)":
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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Söhne:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body { margin:0; padding:0; }
.stApp { background:#212121 !important; color:#ececec !important; font-family:'Inter',sans-serif !important; }
#MainMenu, footer, header { visibility:hidden !important; }

/* ═══ SIDEBAR ═══ */
section[data-testid="stSidebar"] {
    background:#171717 !important;
    border-right:1px solid #2e2e2e !important;
    width:260px !important; min-width:260px !important;
}
section[data-testid="stSidebar"] > div { padding:0 !important; overflow-y:auto; }
[data-testid="collapsedControl"] {
    display:flex !important; visibility:visible !important; opacity:1 !important;
    background:#171717 !important; border:1px solid #2e2e2e !important;
    border-radius:0 6px 6px 0 !important; color:#ececec !important;
}

/* ═══ ALL BUTTONS BASE ═══ */
.stButton > button {
    background:transparent !important; color:#ececec !important;
    border:none !important; border-radius:8px !important;
    width:100% !important; text-align:left !important;
    padding:10px 14px !important; font-size:14px !important;
    font-weight:400 !important; cursor:pointer !important;
    transition:background 0.1s !important;
    white-space:nowrap !important; overflow:hidden !important;
    text-overflow:ellipsis !important;
}
.stButton > button:hover { background:#2a2a2a !important; }
.stButton > button:focus { box-shadow:none !important; border:none !important; }

/* ═══ MAIN AREA ═══ */
.main .block-container {
    padding:0 !important; max-width:100% !important; background:#212121 !important;
}

/* ═══ CHAT MESSAGES ═══ */
[data-testid="stChatMessage"] {
    background:transparent !important; border:none !important;
    padding:20px 0 !important; max-width:48rem !important;
    margin:0 auto !important; width:100% !important;
}
[data-testid="stChatMessageContent"] {
    background:transparent !important; color:#ececec !important;
    font-size:16px !important; line-height:1.75 !important;
    overflow:visible !important; white-space:normal !important;
    word-wrap:break-word !important; max-width:100% !important;
}
[data-testid="stChatMessageContent"] p {
    color:#ececec !important; margin-bottom:10px !important;
    overflow:visible !important; white-space:normal !important;
}
[data-testid="stChatMessageContent"] h1,
[data-testid="stChatMessageContent"] h2,
[data-testid="stChatMessageContent"] h3 { color:#ececec !important; }
[data-testid="stChatMessageContent"] code {
    background:#2d2d2d !important; color:#e2e2e2 !important;
    border-radius:4px !important; padding:2px 6px !important; font-size:14px !important;
}
[data-testid="stChatMessageContent"] pre {
    background:#1a1a1a !important; border:1px solid #333 !important;
    border-radius:8px !important; padding:16px !important; overflow-x:auto !important;
}
[data-testid="stChatMessageContent"] ul,
[data-testid="stChatMessageContent"] ol { color:#ececec !important; padding-left:20px !important; }
[data-testid="stChatMessageContent"] li { margin-bottom:4px !important; }
[data-testid="stChatMessageContent"] a { color:#10a37f !important; }

/* ═══ BOTTOM INPUT AREA ═══ */
[data-testid="stBottom"] {
    background:#212121 !important; border-top:none !important;
    padding:8px 0 20px 0 !important;
}
[data-testid="stBottom"] > div { background:#212121 !important; }

/* ═══ CHAT INPUT ═══ */
[data-testid="stChatInput"] {
    background:#2f2f2f !important; border:1px solid #404040 !important;
    border-radius:16px !important; max-width:48rem !important;
    margin:0 auto !important; box-shadow:0 0 0 1px transparent !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color:#10a37f !important;
}
[data-testid="stChatInput"] textarea {
    background:#2f2f2f !important; color:#ececec !important;
    border:none !important; border-radius:16px !important;
    font-size:16px !important; padding:14px 50px 14px 18px !important;
    resize:none !important; box-shadow:none !important; outline:none !important;
    font-family:'Inter',sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder { color:#8e8ea0 !important; }
[data-testid="stChatInput"] button {
    background:#10a37f !important; border-radius:10px !important;
    color:white !important; border:none !important; margin-right:8px !important;
}
[data-testid="stChatInput"] button:hover { background:#0d8a6a !important; }
[data-testid="stChatInput"] button:disabled { background:#444 !important; }

/* ═══ SELECT BOX ═══ */
.stSelectbox > div > div {
    background:#2f2f2f !important; color:#ececec !important;
    border:1px solid #3a3a3a !important; border-radius:10px !important;
    font-size:13px !important;
}
.stSelectbox label { color:#888 !important; font-size:11px !important; text-transform:uppercase; letter-spacing:0.06em; }

/* ═══ TOGGLE ═══ */
.stToggle { padding:4px 0 !important; }
.stToggle label { color:#c5c5c5 !important; font-size:13px !important; }
.stToggle p { color:#c5c5c5 !important; font-size:13px !important; }

/* ═══ DIVIDER ═══ */
hr { border:none !important; border-top:1px solid #2e2e2e !important; margin:6px 0 !important; }

/* ═══ SCROLLBAR ═══ */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#3a3a3a; border-radius:4px; }

/* ═══ SIDEBAR LABELS ═══ */
.s-label {
    font-size:11px; font-weight:600; color:#555; text-transform:uppercase;
    letter-spacing:0.08em; padding:8px 14px 4px;
}
.s-section-title {
    font-size:12px; color:#888; padding:10px 14px 2px;
    font-weight:500;
}

/* ═══ NEW CHAT BUTTON ═══ */
.new-chat-wrap button {
    background:#1c1c1c !important; color:#ececec !important;
    border:1px solid #333 !important; border-radius:10px !important;
    font-size:14px !important; font-weight:500 !important;
    padding:10px 14px !important; margin:8px 0 !important;
    display:flex !important; align-items:center !important; gap:8px !important;
}
.new-chat-wrap button:hover { background:#2a2a2a !important; border-color:#444 !important; }

/* ═══ PROFILE CARD ═══ */
.prof-wrap {
    padding:10px 12px 14px;
}
.prof-card {
    display:flex; align-items:center; gap:10px;
    padding:8px 10px; border-radius:10px;
    cursor:pointer; transition:background 0.15s;
}
.prof-card:hover { background:#2a2a2a; }
.prof-name  { font-size:13px; font-weight:600; color:#ececec; }
.prof-sub   { font-size:11px; color:#666; }

/* ═══ CHAT HISTORY ITEMS ═══ */
.hist-item button {
    background:transparent !important; color:#c5c5c5 !important;
    border:none !important; border-radius:8px !important;
    font-size:13.5px !important; text-align:left !important;
    padding:8px 14px !important; white-space:nowrap !important;
    overflow:hidden !important; text-overflow:ellipsis !important;
}
.hist-item button:hover { background:#2a2a2a !important; color:#fff !important; }

/* ═══ MODEL BADGES ═══ */
.mbadge {
    display:inline-flex; align-items:center; gap:4px;
    font-size:11px; font-weight:600; padding:2px 10px;
    border-radius:20px; margin-bottom:8px;
}
.mbadge.groq    { background:#0d3b30; color:#1dc8a0; border:1px solid #0d8a6a; }
.mbadge.gemini  { background:#0d2060; color:#7baaf7; border:1px solid #2d5dbf; }
.mbadge.cohere  { background:#3d2200; color:#f59e0b; border:1px solid #92400e; }
.mbadge.mistral { background:#2d1060; color:#c084fc; border:1px solid #7c3aed; }
.mbadge.live    { background:#0d2d1a; color:#4ade80; border:1px solid #166534; margin-left:5px; }

/* ═══ SUGGESTION CHIPS ═══ */
.chip button {
    background:#2a2a2a !important; color:#c5c5c5 !important;
    border:1px solid #383838 !important; border-radius:14px !important;
    font-size:13.5px !important; text-align:left !important;
    padding:14px 16px !important; height:auto !important;
    white-space:normal !important; line-height:1.45 !important;
    transition:all 0.15s !important;
}
.chip button:hover {
    background:#303030 !important; border-color:#10a37f !important; color:#fff !important;
}

/* ═══ ACCOUNT PAGE ═══ */
.acc-card {
    background:linear-gradient(135deg,#1a3a30,#1e2a25);
    border:1px solid #2a5a42; border-radius:16px;
    padding:28px; display:flex; align-items:center; gap:18px; margin-bottom:20px;
}
.acc-name  { font-size:22px; font-weight:700; color:#ececec; }
.acc-email { font-size:13px; color:#888; margin-top:4px; }
.acc-badge {
    display:inline-flex; align-items:center; gap:5px;
    background:#10a37f; color:white; font-size:11px; font-weight:600;
    padding:3px 12px; border-radius:20px; margin-top:8px;
}
.stat-box  {
    background:#2a2a2a; border:1px solid #333; border-radius:12px;
    padding:20px; text-align:center;
}
.stat-n    { font-size:28px; font-weight:700; color:#10a37f; }
.stat-l    { font-size:12px; color:#888; margin-top:4px; }
.info-box  {
    background:#2a2a2a; border:1px solid #333; border-radius:12px;
    padding:18px 22px; margin-bottom:14px;
}
.info-ttl  { font-size:11px; font-weight:600; color:#555; text-transform:uppercase;
    letter-spacing:0.06em; margin-bottom:14px; }
.irow      { display:flex; justify-content:space-between; align-items:center;
    padding:10px 0; border-bottom:1px solid #2e2e2e; }
.irow:last-child { border-bottom:none; }
.ik        { font-size:13px; color:#888; }
.iv        { font-size:13px; color:#ececec; font-weight:600; }

/* ═══ LOGIN ═══ */
.login-card {
    background:#2a2a2a; border:1px solid #333; border-radius:18px;
    padding:36px 32px; box-shadow:0 8px 32px rgba(0,0,0,0.5);
}
.fi { display:flex; align-items:center; gap:10px; font-size:14px; color:#999; margin-bottom:12px; }
.fi-icon { width:30px; height:30px; border-radius:8px; display:flex;
    align-items:center; justify-content:center; font-size:15px; flex-shrink:0; }

/* ═══ MODEL SELECTOR PAGE ═══ */
.model-opt {
    background:#2a2a2a; border:2px solid #333; border-radius:12px;
    padding:14px 18px; margin-bottom:10px; cursor:pointer;
    display:flex; align-items:center; gap:12px; transition:all 0.15s;
}
.model-opt.active { border-color:#10a37f !important; background:#0d2d22 !important; }
.model-opt-name { font-size:15px; font-weight:600; color:#ececec; }
.model-opt-desc { font-size:12px; color:#888; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "auth":False,"user":None,"msgs":[],"chat_id":None,
    "chats":load_chats(),"model":"Llama 3.3 (Groq)",
    "page":"chat","live":True,"login_time":"",
}
for k,v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.auth:
    _, mid, _ = st.columns([1,1.1,1])
    with mid:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="text-align:center;margin-bottom:22px;">
          <div style="display:inline-flex;align-items:center;justify-content:center;
            width:80px;height:80px;background:#1a3a30;border-radius:20px;
            border:2px solid #10a37f;margin-bottom:14px;
            box-shadow:0 4px 24px rgba(16,163,127,0.3);">
            {robot_img(54)}
          </div>
          <h1 style="font-size:2rem;font-weight:700;color:#ececec;margin:0 0 6px;letter-spacing:-0.5px;">
            Surya Dev AI
          </h1>
          <p style="color:#888;font-size:14px;margin:0;">AI chat powered by multiple models</p>
        </div>
        <div class="login-card">
          <p style="font-size:16px;font-weight:600;color:#ececec;text-align:center;margin-bottom:4px;">Welcome back</p>
          <p style="font-size:13px;color:#777;text-align:center;margin-bottom:22px;">Sign in to continue</p>
          <div class="fi"><div class="fi-icon" style="background:#0d3b30;">⚡</div>4 AI models — Llama, Gemini, Cohere, Mistral</div>
          <div class="fi"><div class="fi-icon" style="background:#0d2060;">🌐</div>Live web search (Wikipedia + DuckDuckGo)</div>
          <div class="fi"><div class="fi-icon" style="background:#2d1060;">💬</div>Full chat history across sessions</div>
          <div class="fi"><div class="fi-icon" style="background:#3d2200;">🔒</div>Secure Google account login</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, TOKEN_URL, TOKEN_URL)
        result = oauth2.authorize_button("Continue with Google", redirect_uri=REDIRECT_URI,
            scope=SCOPE, key="google",
            extras_params={"prompt":"consent","access_type":"offline"},
            use_container_width=True)
        st.markdown('<p style="text-align:center;font-size:11px;color:#444;margin-top:14px;">By continuing you agree to our terms</p>', unsafe_allow_html=True)
        if result and "token" in result:
            payload = jwt.decode(result["token"]["id_token"], options={"verify_signature":False})
            st.session_state.auth = True
            st.session_state.user = payload
            st.session_state.login_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  APP (AUTHENTICATED)
# ══════════════════════════════════════════════════════════════════════════════
else:
    u     = st.session_state.user
    email = u.get("email","")
    name  = u.get("name","User") or "User"
    fname = name.split()[0]
    pic   = u.get("picture","")
    init  = name[0].upper()
    chats = st.session_state.chats
    if email not in chats: chats[email] = {}
    uchats = chats[email]
    mdl    = st.session_state.model
    tag    = MODELS[mdl]["tag"]
    icon   = MODELS[mdl]["icon"]

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        # Top: logo + collapse icon row
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;padding:14px 14px 6px;">
          <div style="display:flex;align-items:center;gap:8px;">
            <div style="width:28px;height:28px;background:#1a3a30;border-radius:6px;
              border:1px solid #10a37f;display:flex;align-items:center;justify-content:center;">
              {robot_img(20)}
            </div>
            <span style="font-size:14px;font-weight:700;color:#ececec;">Surya Dev AI</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # New Chat
        st.markdown('<div class="new-chat-wrap">', unsafe_allow_html=True)
        if st.button("✏️   New chat", key="new_chat", use_container_width=True):
            st.session_state.msgs = []
            st.session_state.chat_id = None
            st.session_state.page = "chat"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Sidebar nav items (like ChatGPT)
        st.markdown('<div style="padding:0 6px;">', unsafe_allow_html=True)
        if st.button("🔍   Search chats", key="nav_search", use_container_width=True):
            st.session_state.page = "search"
            st.rerun()
        if st.button("🤖   AI Models", key="nav_models", use_container_width=True):
            st.session_state.page = "models"
            st.rerun()
        if st.button("👤   Account", key="nav_acc", use_container_width=True):
            st.session_state.page = "account"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        # Model picker
        st.markdown('<div class="s-label">Active Model</div>', unsafe_allow_html=True)
        new_model = st.selectbox("Model", list(MODELS.keys()),
            index=list(MODELS.keys()).index(mdl), label_visibility="collapsed")
        if new_model != mdl:
            st.session_state.model = new_model
            st.rerun()

        # Live search toggle
        live_on = st.toggle("🌐 Live Web Search", value=st.session_state.live)
        if live_on != st.session_state.live:
            st.session_state.live = live_on
            st.rerun()

        st.divider()

        # Chat history
        if uchats:
            st.markdown('<div class="s-section-title">Recent</div>', unsafe_allow_html=True)
            for cid, cdata in sorted(uchats.items(), reverse=True)[:25]:
                title = cdata.get("title","Untitled")
                disp  = title[:32] + ("…" if len(title)>32 else "")
                st.markdown('<div class="hist-item">', unsafe_allow_html=True)
                if st.button(f"💬 {disp}", key=f"h_{cid}", use_container_width=True):
                    st.session_state.chat_id = cid
                    st.session_state.msgs = cdata["messages"]
                    st.session_state.page = "chat"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        # Profile at bottom
        st.markdown('<div style="position:fixed;bottom:0;width:232px;background:#171717;padding:8px 0;border-top:1px solid #2e2e2e;">', unsafe_allow_html=True)
        av = f'<img src="{pic}" style="width:32px;height:32px;border-radius:50%;object-fit:cover;flex-shrink:0;"/>' if pic \
             else f'<div style="width:32px;height:32px;border-radius:50%;background:#10a37f;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:13px;flex-shrink:0;">{init}</div>'
        st.markdown(f"""
        <div class="prof-card" style="margin:0 8px;">
          {av}
          <div style="flex:1;min-width:0;">
            <div class="prof-name">{name}</div>
            <div class="prof-sub">Free</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Sign out", key="signout", use_container_width=True):
            for k in DEFAULTS: st.session_state[k] = DEFAULTS[k]
            st.session_state.chats = load_chats()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  ACCOUNT PAGE
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.page == "account":
        _, cc, _ = st.columns([1,3,1])
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back", key="back_acc"):
                st.session_state.page = "chat"; st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)

            av_lg = f'<img src="{pic}" style="width:70px;height:70px;border-radius:50%;border:3px solid #10a37f;object-fit:cover;"/>' if pic \
                    else f'<div style="width:70px;height:70px;border-radius:50%;background:#10a37f;display:flex;align-items:center;justify-content:center;font-size:28px;color:white;font-weight:700;flex-shrink:0;">{init}</div>'

            tc = len(uchats); tm = total_msgs(uchats)
            lt = st.session_state.login_time or "—"

            st.markdown(f"""
            <div class="acc-card">
              {av_lg}
              <div>
                <div class="acc-name">{name}</div>
                <div class="acc-email">{email}</div>
                <div class="acc-badge">🤖 Surya Dev AI Member</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            c1,c2,c3,c4 = st.columns(4)
            for col,num,lbl in [(c1,tc,"Chats"),(c2,tm,"Messages"),(c3,4,"AI Models"),(c4,"ON" if live_on else "OFF","Live Search")]:
                col.markdown(f'<div class="stat-box"><div class="stat-n">{num}</div><div class="stat-l">{lbl}</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
              <div class="info-ttl">Account Details</div>
              <div class="irow"><span class="ik">Full Name</span><span class="iv">{name}</span></div>
              <div class="irow"><span class="ik">Email</span><span class="iv">{email}</span></div>
              <div class="irow"><span class="ik">Provider</span><span class="iv">🔵 Google</span></div>
              <div class="irow"><span class="ik">Session Started</span><span class="iv">{lt}</span></div>
              <div class="irow"><span class="ik">Status</span><span class="iv" style="color:#10a37f;">✅ Active</span></div>
            </div>
            <div class="info-box">
              <div class="info-ttl">Preferences</div>
              <div class="irow"><span class="ik">Active Model</span><span class="iv">{icon} {mdl}</span></div>
              <div class="irow"><span class="ik">Live Search</span><span class="iv" style="color:#10a37f;">{"✅ Enabled" if live_on else "❌ Disabled"}</span></div>
              <div class="irow"><span class="ik">Sources</span><span class="iv">Wikipedia + DuckDuckGo</span></div>
              <div class="irow"><span class="ik">Theme</span><span class="iv">🌙 Dark</span></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            b1,b2 = st.columns(2)
            with b1:
                if st.button("🗑️ Clear all chats", use_container_width=True, key="clr_all"):
                    chats[email]={}; st.session_state.chats=chats
                    st.session_state.msgs=[]; st.session_state.chat_id=None
                    save_chats(chats); st.success("Cleared!"); st.rerun()
            with b2:
                if st.button("🚪 Sign out", use_container_width=True, key="so2"):
                    for k in DEFAULTS: st.session_state[k] = DEFAULTS[k]
                    st.session_state.chats = load_chats(); st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  MODELS PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.page == "models":
        _, cc, _ = st.columns([1,3,1])
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back", key="back_mdl"):
                st.session_state.page = "chat"; st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<h2 style="color:#ececec;font-size:1.5rem;margin-bottom:6px;">AI Models</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color:#888;font-size:14px;margin-bottom:24px;">Choose which AI model to chat with</p>', unsafe_allow_html=True)

            MODEL_DETAILS = {
                "Llama 3.3 (Groq)":   {"icon":"⚡","color":"#10a37f","desc":"Fastest model. Powered by Groq's LPU. Great for general tasks, coding, and quick answers.","tag":"groq","speed":"🚀 Very Fast"},
                "Gemini 1.5 Flash":   {"icon":"✦","color":"#4285f4","desc":"Google's multimodal AI. Excellent at reasoning, analysis, and following complex instructions.","tag":"gemini","speed":"⚡ Fast"},
                "Command R (Cohere)": {"icon":"◈","color":"#d97706","desc":"Cohere's Command R model. Best for retrieval-augmented generation and document analysis.","tag":"cohere","speed":"🏃 Moderate"},
                "Mistral Small":      {"icon":"◆","color":"#7c3aed","desc":"Mistral's efficient small model. Great balance of speed and quality for everyday tasks.","tag":"mistral","speed":"⚡ Fast"},
            }
            for mname, minfo in MODEL_DETAILS.items():
                is_active = mname == st.session_state.model
                border = "#10a37f" if is_active else "#333"
                bg = "#0d2d22" if is_active else "#2a2a2a"
                st.markdown(f"""
                <div style="background:{bg};border:2px solid {border};border-radius:14px;
                  padding:18px 20px;margin-bottom:12px;transition:all 0.15s;">
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    <span style="font-size:24px;">{minfo['icon']}</span>
                    <div>
                      <div style="font-size:15px;font-weight:600;color:#ececec;">{mname}</div>
                      <div style="font-size:12px;color:#888;">{minfo['speed']}</div>
                    </div>
                    {"<span style='margin-left:auto;background:#10a37f;color:white;font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;'>✓ Active</span>" if is_active else ""}
                  </div>
                  <p style="font-size:13px;color:#aaa;margin:0;">{minfo['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                if not is_active:
                    if st.button(f"Use {mname}", key=f"sel_{mname}", use_container_width=True):
                        st.session_state.model = mname
                        st.session_state.page = "chat"
                        st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  SEARCH PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.page == "search":
        _, cc, _ = st.columns([1,3,1])
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back", key="back_srch"):
                st.session_state.page = "chat"; st.rerun()
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<h2 style="color:#ececec;font-size:1.5rem;margin-bottom:16px;">Search chats</h2>', unsafe_allow_html=True)
            query = st.text_input("", placeholder="Search your conversations…",
                label_visibility="collapsed")
            if query:
                results = [(cid, cd) for cid,cd in uchats.items()
                           if query.lower() in cd.get("title","").lower()
                           or any(query.lower() in m.get("content","").lower() for m in cd.get("messages",[]))]
                if results:
                    st.markdown(f'<p style="color:#888;font-size:13px;margin-bottom:12px;">{len(results)} result(s)</p>', unsafe_allow_html=True)
                    for cid, cdata in sorted(results, reverse=True):
                        title = cdata.get("title","Untitled")
                        if st.button(f"💬 {title[:50]}", key=f"sr_{cid}", use_container_width=True):
                            st.session_state.chat_id = cid
                            st.session_state.msgs = cdata["messages"]
                            st.session_state.page = "chat"
                            st.rerun()
                else:
                    st.markdown('<p style="color:#666;font-size:14px;">No chats found.</p>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  CHAT PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.page == "chat":

        _, cc, _ = st.columns([1,5,1])
        with cc:

            # Welcome screen
            if not st.session_state.msgs:
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="text-align:center;margin-bottom:36px;">
                  <h1 style="font-size:2.2rem;font-weight:600;color:#ececec;
                    margin:0 0 28px;letter-spacing:-0.5px;">
                    What's on the agenda today?
                  </h1>
                </div>
                """, unsafe_allow_html=True)

                suggestions = [
                    ("What's happening in AI news today?", "🌐"),
                    ("Explain quantum computing in simple terms", "⚛️"),
                    ("Write me a Python web scraper", "🐍"),
                    ("Give me a creative startup idea for 2025", "💡"),
                    ("What is the population of India?", "🗺️"),
                    ("Help me write a professional email", "✉️"),
                ]
                c1, c2 = st.columns(2)
                for i,(txt,ico) in enumerate(suggestions):
                    col = c1 if i%2==0 else c2
                    with col:
                        st.markdown('<div class="chip">', unsafe_allow_html=True)
                        if st.button(f"{ico}  {txt}", use_container_width=True, key=f"s{i}"):
                            st.session_state.msgs.append({"role":"user","content":txt})
                            with st.spinner("Thinking…"):
                                reply, searched = get_response(st.session_state.msgs, mdl, live_on)
                            st.session_state.msgs.append({"role":"assistant","content":reply,"searched":searched})
                            cid = datetime.now().strftime("%Y%m%d%H%M%S%f")
                            st.session_state.chat_id = cid
                            chats[email][cid] = {"title":txt[:40],"messages":st.session_state.msgs}
                            st.session_state.chats = chats; save_chats(chats); st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)

            # Messages
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                for msg in st.session_state.msgs:
                    with st.chat_message(msg["role"]):
                        if msg["role"] == "assistant":
                            badge = f'<span class="mbadge {tag}">{icon} {mdl}</span>'
                            if msg.get("searched"):
                                badge += ' <span class="mbadge live">🌐 Live</span>'
                            st.markdown(badge, unsafe_allow_html=True)
                        st.markdown(msg["content"])

                st.markdown("<br><br><br>", unsafe_allow_html=True)

        # Chat input pinned at bottom
        prompt = st.chat_input("Ask anything…")
        if prompt:
            st.session_state.msgs.append({"role":"user","content":prompt})
            is_live = needs_search(prompt) and live_on
            with st.spinner("🔍 Searching…" if is_live else "💭 Thinking…"):
                reply, searched = get_response(st.session_state.msgs, mdl, live_on)
            st.session_state.msgs.append({"role":"assistant","content":reply,"searched":searched})
            cid = st.session_state.chat_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
            st.session_state.chat_id = cid
            chats[email][cid] = {"title":st.session_state.msgs[0]["content"][:40],"messages":st.session_state.msgs}
            st.session_state.chats = chats; save_chats(chats); st.rerun()
            save_chats(chats)
            st.rerun()
