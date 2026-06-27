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
import html as _html_mod
from datetime import datetime
from urllib.parse import quote

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  — must be the very first Streamlit call
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Surya Dev AI", page_icon="🤖", layout="wide")

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS & CLIENTS
# ════════════════════════════════════════════════════════════════════════════
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
    "Llama 3.3 (Groq)":    "llama-3.3-70b-versatile",
    "Gemini 1.5 Flash":    "gemini-1.5-flash",
    "Command R+ (Cohere)": "command-r-plus",
    "Mistral Small":       "mistral-small-latest",
}
MODEL_ICONS = {
    "Llama 3.3 (Groq)":    "🦙",
    "Gemini 1.5 Flash":    "✨",
    "Command R+ (Cohere)": "🪸",
    "Mistral Small":       "💨",
}

ROBOT_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" '
    'width="{sz}" height="{sz}">'
    '<rect x="14" y="16" width="36" height="28" rx="8" fill="#d97706"/>'
    '<circle cx="24" cy="27" r="5" fill="white"/>'
    '<circle cx="40" cy="27" r="5" fill="white"/>'
    '<circle cx="25" cy="28" r="2.5" fill="#1a1a1a"/>'
    '<circle cx="41" cy="28" r="2.5" fill="#1a1a1a"/>'
    '<circle cx="26" cy="27" r="1" fill="white"/>'
    '<circle cx="42" cy="27" r="1" fill="white"/>'
    '<rect x="22" y="36" width="20" height="3" rx="1.5" fill="white" opacity="0.8"/>'
    '<rect x="25" y="36" width="3" height="3" rx="1" fill="#d97706"/>'
    '<rect x="31" y="36" width="3" height="3" rx="1" fill="#d97706"/>'
    '<rect x="37" y="36" width="3" height="3" rx="1" fill="#d97706"/>'
    '<rect x="30" y="8" width="4" height="10" rx="2" fill="#d97706"/>'
    '<circle cx="32" cy="7" r="4" fill="#f59e0b"/>'
    '<circle cx="32" cy="7" r="2" fill="white"/>'
    '<rect x="7" y="22" width="7" height="12" rx="3" fill="#b45309"/>'
    '<rect x="50" y="22" width="7" height="12" rx="3" fill="#b45309"/>'
    '<rect x="27" y="44" width="10" height="6" rx="2" fill="#b45309"/>'
    '<rect x="16" y="50" width="32" height="10" rx="5" fill="#92400e"/>'
    '<circle cx="32" cy="55" r="3" fill="#fbbf24"/>'
    '<circle cx="32" cy="55" r="1.5" fill="white" opacity="0.8"/>'
    '</svg>'
)

def robot(sz=36):
    return ROBOT_SVG.replace("{sz}", str(sz))

LIVE_KEYWORDS = [
    "latest","current","today","now","recent","news","live",
    "2024","2025","2026","who is","what is","when did","where is",
    "how much","price","weather","score","stock","wiki","wikipedia",
    "tell me about","search for","find information","look up",
    "what happened","update","trending","population","capital of",
    "president of","ceo of","founder of","born","died","history of",
]

# ════════════════════════════════════════════════════════════════════════════
# HELPERS  — defined BEFORE any st.* rendering code uses them
# ════════════════════════════════════════════════════════════════════════════

def md_to_html(text: str) -> str:
    """Lightweight Markdown → HTML (safe for st.markdown unsafe_allow_html)."""
    text = _html_mod.escape(text)

    def code_block(m):
        lang = (m.group(1) or "").strip()
        return f'<pre><code class="lang-{lang}">{m.group(2)}</code></pre>'

    text = re.sub(r"```(\w*)\n(.*?)```", code_block, text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*\*(.+?)\*\*\*", r"<strong><em>\1</em></strong>", text)
    text = re.sub(r"\*\*(.+?)\*\*",     r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*",         r"<em>\1</em>", text)
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$",  r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$",   r"<h1>\1</h1>", text, flags=re.MULTILINE)

    lines, out, in_ul = text.split("\n"), [], False
    for line in lines:
        if re.match(r"^[*\-] ", line):
            if not in_ul:
                out.append('<ul style="margin:6px 0 6px 20px;padding:0">')
                in_ul = True
            out.append(f'<li style="margin:3px 0">{line[2:]}</li>')
        else:
            if in_ul:
                out.append("</ul>")
                in_ul = False
            out.append(line)
    if in_ul:
        out.append("</ul>")
    text = "\n".join(out)

    parts = []
    for para in re.split(r"\n{2,}", text):
        para = para.strip()
        if not para:
            continue
        if para.startswith(("<h", "<ul", "<pre", "<ol")):
            parts.append(para)
        else:
            parts.append(f'<p style="margin:0 0 10px">{para.replace(chr(10), "<br>")}</p>')
    return "".join(parts)


def needs_live_search(q: str) -> bool:
    q = q.lower()
    return any(k in q for k in LIVE_KEYWORDS)


def search_wikipedia(query: str):
    try:
        r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}",
            timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get("extract"):
                return {
                    "source": "Wikipedia",
                    "title":  d.get("title", query),
                    "content": d["extract"][:1000],
                    "url": d.get("content_urls", {}).get("desktop", {}).get("page", ""),
                }
    except requests.RequestException:
        pass
    return None


def search_duckduckgo(query: str):
    try:
        r = requests.get(
            f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1",
            timeout=5)
        if r.status_code == 200:
            d    = r.json()
            rows = []
            if d.get("Answer"):       rows.append(f"Quick Answer: {d['Answer']}")
            if d.get("AbstractText"): rows.append(f"Summary: {d['AbstractText'][:800]}")
            for f in (d.get("Infobox") or {}).get("content", [])[:3]:
                if f.get("label") and f.get("value"):
                    rows.append(f"{f['label']}: {f['value']}")
            if rows:
                return {
                    "source":  "DuckDuckGo",
                    "title":   d.get("Heading", query),
                    "content": "\n".join(rows),
                    "url":     d.get("AbstractURL", ""),
                }
    except requests.RequestException:
        pass
    return None


def get_live_context(query: str):
    results, seen = [], set()
    ddg = search_duckduckgo(query)
    if ddg and ddg["content"]:
        results.append(ddg)
        seen.add(ddg["content"][:100])
    topic = re.sub(
        r"^(what is|who is|tell me about|search for|find|look up|when did|where is)\s+",
        "", query.lower()).strip()
    wiki = search_wikipedia(topic)
    if wiki and wiki["content"] and wiki["content"][:100] not in seen:
        results.append(wiki)
    return results


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
            f"LIVE WEB CONTEXT:\n{ctx}\n\nSources:\n{srcs}\n\n"
            "Use this context to answer. Mention source when citing live data.\n\n"
            f"User question: {last_user}"
        ),
    }
    return [m for m in messages[:-1]] + [inject], True, live


def get_ai_response(messages, model_choice, use_search=True):
    try:
        msgs, used, sources = _build_messages(messages, use_search)

        if model_choice == "Llama 3.3 (Groq)":
            r    = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs)
            text = r.choices[0].message.content

        elif model_choice == "Gemini 1.5 Flash":
            gm   = genai.GenerativeModel("gemini-1.5-flash")
            hist = []
            for m in msgs[:-1]:
                hist.append({"role": "user" if m["role"]=="user" else "model",
                             "parts": [m["content"]]})
            while hist and hist[0]["role"] != "user":
                hist.pop(0)
            text = gm.start_chat(history=hist).send_message(msgs[-1]["content"]).text

        elif model_choice == "Command R+ (Cohere)":
            cohere_msgs = [
                {"role": "user" if m["role"]=="user" else "assistant", "content": m["content"]}
                for m in msgs
            ]
            r    = cohere_client.chat(model="command-r-plus", messages=cohere_msgs)
            text = r.message.content[0].text

        elif model_choice == "Mistral Small":
            r = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": "mistral-small-latest", "messages": msgs},
                timeout=30)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]

        else:
            return "⚠️ Unknown model.", False, []

        return text, used, sources
    except Exception as e:
        return f"⚠️ Error: {e}", False, []


def load_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_chats(chats):
    try:
        with open(CHATS_FILE, "w") as f:
            json.dump(chats, f)
    except OSError:
        pass


def get_greeting():
    h = datetime.now().hour
    return "Good morning" if h < 12 else ("Good afternoon" if h < 17 else "Good evening")


def count_user_messages(chats_dict):
    return sum(
        1 for chat in chats_dict.values()
        for msg in chat.get("messages", []) if msg["role"] == "user"
    )


def persist_chat(all_chats, user_email, title):
    cid = st.session_state.current_chat_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.current_chat_id = cid
    all_chats.setdefault(user_email, {})[cid] = {
        "title":    title[:40],
        "messages": st.session_state.messages,
        "model":    st.session_state.get("model_choice", list(MODELS)[0]),
        "updated":  datetime.now().isoformat(),
    }
    st.session_state.all_chats = all_chats
    save_chats(all_chats)


def do_signout():
    for k in ["authenticated","user_info","messages","current_chat_id","pending_input"]:
        st.session_state.pop(k, None)
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════════════════
_defaults = {
    "authenticated":   False,
    "user_info":       None,
    "messages":        [],
    "current_chat_id": None,
    "all_chats":       load_chats(),
    "model_choice":    list(MODELS)[0],
    "use_search":      True,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── base ── */
html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"]{
    background:#1a1a1a!important;
    color:#e5e5e5;
    font-family:'Inter',-apple-system,BlinkMacSystemFont,sans-serif;
}
#MainMenu,footer,header,[data-testid="stToolbar"],.stDeployButton{display:none!important;}

/* ── sidebar ── */
[data-testid="stSidebar"]{
    background:#111!important;
    border-right:1px solid #222;
}
[data-testid="stSidebar"] section{padding:0!important;}

/* ── buttons in sidebar ── */
[data-testid="stSidebar"] .stButton button{
    background:#1a1a1a!important;
    color:#bbb!important;
    border:1px solid #2a2a2a!important;
    border-radius:7px!important;
    font-size:13px!important;
    font-weight:400!important;
    text-align:left!important;
    padding:7px 12px!important;
    transition:background .12s,color .12s!important;
}
[data-testid="stSidebar"] .stButton button:hover{
    background:#232323!important;color:#eee!important;
}

/* ── new-chat button override ── */
[data-testid="stSidebar"] .stButton:first-of-type button{
    background:#222!important;
    border:1px solid #333!important;
    font-weight:600!important;
    color:#ddd!important;
}

/* ── main content ── */
.main .block-container{
    padding:0!important;
    max-width:100%!important;
}

/* ── top bar ── */
.topbar{
    display:flex;align-items:center;justify-content:space-between;
    padding:10px 24px;
    border-bottom:1px solid #222;
    background:#1a1a1a;
    position:sticky;top:0;z-index:100;
}
.topbar-left{display:flex;align-items:center;gap:8px;font-size:13.5px;font-weight:500;color:#ccc;}
.topbar-right{display:flex;align-items:center;gap:8px;}
.avatar-chip{
    width:32px;height:32px;border-radius:50%;
    background:linear-gradient(135deg,#d97706,#f59e0b);
    display:flex;align-items:center;justify-content:center;
    font-size:13px;font-weight:700;color:#fff;
}
.search-on-badge{
    display:inline-flex;align-items:center;gap:4px;
    background:rgba(217,119,6,.12);
    border:1px solid rgba(217,119,6,.3);
    border-radius:20px;padding:2px 9px;font-size:11px;color:#d97706;
}

/* ── chat scroll area ── */
.chat-scroll{
    max-width:760px;margin:0 auto;
    padding:28px 20px 180px;
}

/* ── welcome ── */
.welcome-wrap{
    display:flex;flex-direction:column;align-items:center;
    justify-content:center;min-height:62vh;text-align:center;padding:40px 20px;
}
.welcome-title{font-size:28px;font-weight:700;color:#fff;letter-spacing:-.5px;margin:16px 0 6px;}
.welcome-sub{font-size:15px;color:#888;margin-bottom:30px;}
.sug-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:560px;width:100%;}
.sug-card{
    background:#222;border:1px solid #2e2e2e;border-radius:10px;
    padding:14px 16px;text-align:left;cursor:pointer;
    transition:background .15s,border-color .15s;
}
.sug-card:hover{background:#2a2a2a;border-color:#3a3a3a;}
.sug-icon{font-size:18px;margin-bottom:5px;}
.sug-title{font-size:13.5px;font-weight:600;color:#ddd;margin-bottom:2px;}
.sug-desc{font-size:12px;color:#777;}

/* ── messages ── */
.msg-row{display:flex;gap:12px;margin-bottom:24px;animation:fs .2s ease;}
@keyframes fs{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.msg-av{
    width:32px;height:32px;border-radius:50%;flex-shrink:0;
    display:flex;align-items:center;justify-content:center;
    margin-top:2px;font-size:13px;font-weight:700;
}
.msg-av.user{background:linear-gradient(135deg,#d97706,#f59e0b);color:#fff;}
.msg-av.ai{background:#252525;border:1px solid #333;}
.msg-body{flex:1;min-width:0;}
.msg-name{font-size:12px;font-weight:600;color:#777;margin-bottom:5px;}
.msg-text{font-size:14.5px;line-height:1.68;color:#e0e0e0;}
.msg-text p{margin:0 0 10px}
.msg-text p:last-child{margin-bottom:0}
.msg-text code{
    background:#2a2a2a;border:1px solid #333;border-radius:4px;
    padding:2px 5px;font-size:13px;color:#f0a050;
    font-family:'JetBrains Mono','Fira Code',monospace;
}
.msg-text pre{background:#1e1e1e;border:1px solid #2e2e2e;border-radius:8px;
    padding:14px 16px;overflow-x:auto;margin:10px 0;}
.msg-text pre code{background:none;border:none;padding:0;color:#d4d4d4;font-size:13px;}
.msg-text h1,.msg-text h2,.msg-text h3{color:#eee;margin:14px 0 6px}
.msg-text ul{margin:6px 0 6px 20px;padding:0}
.msg-text li{margin:3px 0}
.search-used-badge{
    display:inline-flex;align-items:center;gap:4px;
    background:rgba(217,119,6,.1);border:1px solid rgba(217,119,6,.25);
    border-radius:20px;padding:2px 8px;font-size:11px;color:#d97706;margin-bottom:7px;
}
.src-pill{
    display:inline-flex;align-items:center;gap:4px;
    background:#222;border:1px solid #2e2e2e;border-radius:20px;
    padding:3px 9px;font-size:11.5px;color:#888;
    text-decoration:none;margin:5px 4px 0 0;
}
.src-pill:hover{background:#2a2a2a;color:#bbb;}

/* ── input area ── */
.input-outer{
    position:fixed;bottom:0;left:var(--sidebar-width,260px);right:0;
    padding:10px 20px 18px;
    background:linear-gradient(to top,#1a1a1a 65%,transparent);
    z-index:50;
}
.input-inner{
    max-width:760px;margin:0 auto;
    background:#252525;border:1px solid #333;border-radius:14px;
    padding:6px 6px 6px 14px;
    display:flex;align-items:flex-end;gap:8px;
    box-shadow:0 8px 32px rgba(0,0,0,.4);
    transition:border-color .15s,box-shadow .15s;
}
.input-inner:focus-within{
    border-color:#d97706;
    box-shadow:0 0 0 3px rgba(217,119,6,.12),0 8px 32px rgba(0,0,0,.4);
}

/* textarea */
.stTextArea textarea{
    background:transparent!important;border:none!important;
    color:#e5e5e5!important;font-size:14.5px!important;
    font-family:'Inter',sans-serif!important;line-height:1.6!important;
    resize:none!important;box-shadow:none!important;padding:6px 0!important;
}
.stTextArea textarea:focus{box-shadow:none!important;border:none!important;}
[data-testid="stTextArea"] label{display:none!important;}
[data-testid="stTextArea"]{border:none!important;background:transparent!important;}

/* send button */
.send-wrap .stButton button{
    background:#d97706!important;color:#fff!important;
    border:none!important;border-radius:9px!important;
    padding:9px 14px!important;font-size:16px!important;
    font-weight:700!important;line-height:1!important;
    transition:background .15s!important;
}
.send-wrap .stButton button:hover{background:#b45309!important;}

/* selectbox */
[data-baseweb="select"]>div{
    background:#1e1e1e!important;border-color:#2e2e2e!important;color:#ddd!important;
}
[data-baseweb="select"] span,[data-baseweb="select"] div{color:#ddd!important;}
[data-baseweb="menu"]{background:#1e1e1e!important;border:1px solid #333!important;}
[data-baseweb="option"]{background:#1e1e1e!important;color:#ddd!important;}
[data-baseweb="option"]:hover{background:#2a2a2a!important;}

/* toggle */
[data-testid="stToggle"] label span{color:#aaa!important;font-size:13px!important;}

/* divider */
hr{border:none;border-top:1px solid #222;margin:6px 0;}

/* scrollbar */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#333;border-radius:10px}
::-webkit-scrollbar-thumb:hover{background:#444}

/* spinner */
.stSpinner>div{border-top-color:#d97706!important;}

@media(max-width:768px){
    .input-outer{left:0!important;}
    .sug-grid{grid-template-columns:1fr!important;}
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# OAUTH
# ════════════════════════════════════════════════════════════════════════════
oauth2 = OAuth2Component(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    authorize_endpoint=AUTHORIZE_URL,
    token_endpoint=TOKEN_URL,
)

# ════════════════════════════════════════════════════════════════════════════
# AUTH SCREEN
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    # Center the auth card using columns
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("<br><br>", unsafe_allow_html=True)
        # Logo
        st.markdown(
            f'<div style="display:flex;justify-content:center;margin-bottom:4px">'
            f'{robot(60)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<h1 style="text-align:center;font-size:26px;font-weight:700;'
            'color:#fff;letter-spacing:-.4px;margin:8px 0 6px">Surya Dev AI</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="text-align:center;font-size:14px;color:#888;'
            'margin-bottom:26px;line-height:1.5">'
            'Sign in with Google to access your chats.</p>',
            unsafe_allow_html=True,
        )
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
                payload = jwt.decode(
                    result["token"].get("id_token",""),
                    options={"verify_signature": False})
                st.session_state.user_info     = payload
                st.session_state.authenticated = True
                email = payload.get("email","unknown")
                st.session_state.all_chats.setdefault(email, {})
                st.rerun()
            except Exception as e:
                st.error(f"Auth error: {e}")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# AUTHENTICATED APP
# ════════════════════════════════════════════════════════════════════════════
user_info  = st.session_state.user_info or {}
user_email = user_info.get("email", "unknown")
user_name  = user_info.get("name", "User")
user_pic   = user_info.get("picture", "")
initials   = "".join(p[0].upper() for p in user_name.split()[:2]) or "U"
user_chats = st.session_state.all_chats.get(user_email, {})

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'padding:16px 14px 12px;border-bottom:1px solid #222">'
        f'{robot(26)}'
        f'<span style="font-size:15px;font-weight:700;color:#fff">Surya Dev AI</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("✏️  New chat", use_container_width=True, key="new_chat"):
        st.session_state.messages        = []
        st.session_state.current_chat_id = None
        st.rerun()

    st.markdown("---")

    # Model
    st.markdown(
        '<div style="font-size:10.5px;font-weight:600;text-transform:uppercase;'
        'letter-spacing:.1em;color:#555;padding:4px 2px 4px">Model</div>',
        unsafe_allow_html=True,
    )
    model_keys = list(MODELS)
    sel_idx    = model_keys.index(st.session_state.model_choice) \
                 if st.session_state.model_choice in model_keys else 0
    st.session_state.model_choice = st.selectbox(
        "model", model_keys, index=sel_idx, label_visibility="collapsed")

    # Web search toggle
    st.session_state.use_search = st.toggle(
        "🔍 Live web search", value=st.session_state.use_search)

    st.markdown("---")

    # Chat history
    if user_chats:
        st.markdown(
            '<div style="font-size:10.5px;font-weight:600;text-transform:uppercase;'
            'letter-spacing:.1em;color:#555;padding:4px 2px">Recent</div>',
            unsafe_allow_html=True)
        sorted_chats = sorted(user_chats.items(),
                              key=lambda x: x[1].get("updated", x[0]), reverse=True)
        for cid, chat in sorted_chats[:30]:
            title  = chat.get("title", "Untitled")
            is_cur = cid == st.session_state.current_chat_id
            label  = f"{'▶ ' if is_cur else ''}{title}"
            if st.button(label, key=f"ch_{cid}", use_container_width=True):
                st.session_state.messages        = chat.get("messages", [])
                st.session_state.current_chat_id = cid
                st.session_state.model_choice    = chat.get("model", model_keys[0])
                st.rerun()

    # Bottom user row
    st.markdown("---")
    c1, c2 = st.columns([1, 3])
    with c1:
        if user_pic:
            st.image(user_pic, width=32)
        else:
            st.markdown(
                f'<div class="avatar-chip">{initials}</div>',
                unsafe_allow_html=True)
    with c2:
        st.markdown(
            f'<div style="font-size:12px;font-weight:600;color:#ddd">{user_name}</div>'
            f'<div style="font-size:11px;color:#555;overflow:hidden;'
            f'text-overflow:ellipsis">{user_email}</div>',
            unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True, key="signout"):
        do_signout()

# ── TOP BAR ──────────────────────────────────────────────────────────────────
model_icon = MODEL_ICONS.get(st.session_state.model_choice, "🤖")
search_badge = (
    '<span class="search-on-badge">🔍 Search on</span>'
    if st.session_state.use_search else ""
)
st.markdown(
    f'<div class="topbar">'
    f'  <div class="topbar-left">'
    f'    <span>{model_icon}</span>'
    f'    <span>{st.session_state.model_choice}</span>'
    f'    {search_badge}'
    f'  </div>'
    f'  <div class="topbar-right">'
    f'    <div class="avatar-chip">{initials}</div>'
    f'  </div>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── CHAT AREA ─────────────────────────────────────────────────────────────────
messages = st.session_state.messages

if not messages:
    # Welcome / suggestions
    name_first = user_name.split()[0] if user_name else "there"
    greeting   = get_greeting()
    suggestions = [
        ("⚛️", "Explain a concept",    "Quantum computing in simple terms",
         "Explain quantum computing simply"),
        ("💻", "Write code",           "Python, JS, SQL and more",
         "Write a Python web scraper"),
        ("🌐", "Search the web",       "Latest news and current events",
         "What is happening in AI news today?"),
        ("💡", "Brainstorm ideas",     "Creative thinking and planning",
         "Help me brainstorm EdTech startup ideas"),
    ]
    sug_html = "".join(
        f'<div class="sug-card" onclick="'
        f'window.parent.document.querySelectorAll(\'textarea\')[0].value=\'{prompt}\''
        f'">'
        f'<div class="sug-icon">{icon}</div>'
        f'<div class="sug-title">{title}</div>'
        f'<div class="sug-desc">{desc}</div>'
        f'</div>'
        for icon, title, desc, prompt in suggestions
    )
    st.markdown(
        f'<div class="chat-scroll">'
        f'<div class="welcome-wrap">'
        f'{robot(54)}'
        f'<div class="welcome-title">{greeting}, {name_first}!</div>'
        f'<div class="welcome-sub">How can I help you today?</div>'
        f'<div class="sug-grid">{sug_html}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        if role == "user":
            safe_content = _html_mod.escape(content).replace("\n", "<br>")
            st.markdown(
                f'<div class="msg-row">'
                f'  <div class="msg-av user">{initials}</div>'
                f'  <div class="msg-body">'
                f'    <div class="msg-name">You</div>'
                f'    <div class="msg-text"><p>{safe_content}</p></div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            src_pills = "".join(
                f'<a class="src-pill" href="{s["url"]}" target="_blank">🔗 {s["source"]}</a>'
                for s in sources if s.get("url")
            )
            search_used = (
                '<div class="search-used-badge">🔍 Used live search</div>'
                if msg.get("used_search") else ""
            )
            m_icon  = MODEL_ICONS.get(msg.get("model",""), "🤖")
            m_label = msg.get("model", "AI")
            st.markdown(
                f'<div class="msg-row">'
                f'  <div class="msg-av ai">{robot(22)}</div>'
                f'  <div class="msg-body">'
                f'    <div class="msg-name">{m_icon} {m_label}</div>'
                f'    {search_used}'
                f'    <div class="msg-text">{md_to_html(content)}</div>'
                f'    {src_pills}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ── INPUT AREA ───────────────────────────────────────────────────────────────
st.markdown(
    '<div class="input-outer"><div class="input-inner">',
    unsafe_allow_html=True,
)
col_ta, col_btn = st.columns([10, 1])
with col_ta:
    user_input = st.text_area(
        "msg", placeholder="Message Surya Dev AI…",
        key="chat_input", height=52, label_visibility="collapsed")
with col_btn:
    st.markdown('<div class="send-wrap">', unsafe_allow_html=True)
    send = st.button("➤", key="send_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div></div>', unsafe_allow_html=True)

# Ctrl+Enter to send
st.markdown("""
<script>
(function(){
  var tries=0;
  var iv=setInterval(function(){
    var ta=window.parent.document.querySelector('textarea');
    if(ta){
      clearInterval(iv);
      ta.addEventListener('keydown',function(e){
        if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){
          var btn=window.parent.document.querySelector('button[kind="secondary"]');
          if(btn)btn.click();
        }
      });
    }
    if(++tries>20)clearInterval(iv);
  },300);
})();
</script>
""", unsafe_allow_html=True)

# ── SEND HANDLER ─────────────────────────────────────────────────────────────
if send and user_input and user_input.strip():
    prompt = user_input.strip()
    st.session_state.messages.append({"role":"user","content":prompt})

    with st.spinner("Thinking…"):
        resp_text, used_search, live_sources = get_ai_response(
            st.session_state.messages,
            st.session_state.model_choice,
            st.session_state.use_search,
        )

    st.session_state.messages.append({
        "role":        "assistant",
        "content":     resp_text,
        "model":       st.session_state.model_choice,
        "used_search": used_search,
        "sources":     live_sources,
    })
    persist_chat(st.session_state.all_chats, user_email, prompt[:40])
    st.rerun()
