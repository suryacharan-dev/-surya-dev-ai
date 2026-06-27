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
 
# ── API Keys & OAuth Config ───────────────────────────────────────────────────
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
 
# ── Robot Logo SVG ────────────────────────────────────────────────────────────
ROBOT_LOGO_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="{size}" height="{size}">
  <!-- Head -->
  <rect x="14" y="16" width="36" height="28" rx="8" fill="#d97706"/>
  <!-- Eyes -->
  <circle cx="24" cy="27" r="5" fill="white"/>
  <circle cx="40" cy="27" r="5" fill="white"/>
  <circle cx="25" cy="28" r="2.5" fill="#1a1a1a"/>
  <circle cx="41" cy="28" r="2.5" fill="#1a1a1a"/>
  <circle cx="26" cy="27" r="1" fill="white"/>
  <circle cx="42" cy="27" r="1" fill="white"/>
  <!-- Mouth -->
  <rect x="22" y="36" width="20" height="3" rx="1.5" fill="white" opacity="0.8"/>
  <rect x="25" y="36" width="3" height="3" rx="1" fill="#d97706"/>
  <rect x="31" y="36" width="3" height="3" rx="1" fill="#d97706"/>
  <rect x="37" y="36" width="3" height="3" rx="1" fill="#d97706"/>
  <!-- Antenna -->
  <rect x="30" y="8" width="4" height="10" rx="2" fill="#d97706"/>
  <circle cx="32" cy="7" r="4" fill="#f59e0b"/>
  <circle cx="32" cy="7" r="2" fill="white"/>
  <!-- Ears -->
  <rect x="7" y="22" width="7" height="12" rx="3" fill="#b45309"/>
  <rect x="50" y="22" width="7" height="12" rx="3" fill="#b45309"/>
  <!-- Neck -->
  <rect x="27" y="44" width="10" height="6" rx="2" fill="#b45309"/>
  <!-- Body -->
  <rect x="16" y="50" width="32" height="10" rx="5" fill="#92400e"/>
  <!-- Chest light -->
  <circle cx="32" cy="55" r="3" fill="#fbbf24"/>
  <circle cx="32" cy="55" r="1.5" fill="white" opacity="0.8"/>
</svg>
"""
 
def get_robot_logo(size=40):
    return ROBOT_LOGO_SVG.replace("{size}", str(size))
 
# ── Live Search Functions ─────────────────────────────────────────────────────
def search_wikipedia(query):
    """Search Wikipedia for live information"""
    try:
        search_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)}"
        resp = requests.get(search_url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if "extract" in data:
                return {
                    "source": "Wikipedia",
                    "title": data.get("title", query),
                    "content": data["extract"][:1000],
                    "url": data.get("content_urls", {}).get("desktop", {}).get("page", "")
                }
    except:
        pass
    return None
 
def search_duckduckgo(query):
    """Search DuckDuckGo Instant Answers API"""
    try:
        url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            abstract = data.get("AbstractText", "")
            answer = data.get("Answer", "")
            infobox = data.get("Infobox", {})
            results = []
            if answer:
                results.append(f"Quick Answer: {answer}")
            if abstract:
                results.append(f"Summary: {abstract[:800]}")
            if infobox and infobox.get("content"):
                facts = infobox["content"][:3]
                for fact in facts:
                    results.append(f"{fact.get('label','')}: {fact.get('value','')}")
            if results:
                return {
                    "source": "DuckDuckGo",
                    "title": data.get("Heading", query),
                    "content": "\n".join(results),
                    "url": data.get("AbstractURL", "")
                }
    except:
        pass
    return None
 
def needs_live_search(query):
    """Detect if a query needs live/current information"""
    live_keywords = [
        "latest", "current", "today", "now", "recent", "news", "live",
        "2024", "2025", "2026", "who is", "what is", "when did", "where is",
        "how much", "price", "weather", "score", "stock", "wiki", "wikipedia",
        "tell me about", "search for", "find information", "look up",
        "what happened", "update", "trending", "population", "capital of",
        "president of", "ceo of", "founder of", "born", "died", "history of"
    ]
    q_lower = query.lower()
    return any(keyword in q_lower for keyword in live_keywords)
 
def get_live_context(query):
    """Get live context from Wikipedia and DuckDuckGo"""
    results = []
 
    # Try DuckDuckGo first (faster)
    ddg = search_duckduckgo(query)
    if ddg and ddg["content"]:
        results.append(ddg)
 
    # Try Wikipedia
    # Extract main topic from query
    topic = re.sub(r'^(what is|who is|tell me about|search for|find|look up|when did|where is)\s+', '', query.lower()).strip()
    wiki = search_wikipedia(topic)
    if wiki and wiki["content"]:
        results.append(wiki)
 
    return results
 
def get_ai_response(messages, model_choice):
    """Get AI response with optional live web search"""
    try:
        last_user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
 
        # Build augmented messages with live search context
        augmented_messages = list(messages)
 
        if needs_live_search(last_user_msg):
            live_results = get_live_context(last_user_msg)
            if live_results:
                context_parts = []
                sources_used = []
                for r in live_results:
                    context_parts.append(f"[{r['source']}] {r['title']}:\n{r['content']}")
                    if r['url']:
                        sources_used.append(f"{r['source']}: {r['url']}")
 
                context_text = "\n\n".join(context_parts)
                sources_text = "\n".join(sources_used) if sources_used else ""
 
                system_inject = {
                    "role": "user",
                    "content": f"""Here is live information I fetched from the web to help answer the next question:
 
--- LIVE WEB CONTEXT ---
{context_text}
--- END CONTEXT ---
 
Sources:
{sources_text}
 
Now answer the user's question using this live context. Always mention the source (Wikipedia/DuckDuckGo) when using live data. If the context doesn't fully answer the question, supplement with your own knowledge.
 
User's question: {last_user_msg}"""
                }
                # Replace last user message with augmented version
                augmented_messages = [m for m in messages[:-1]] + [system_inject]
 
        if model_choice == "Llama 3.3 (Groq)":
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=augmented_messages)
            return response.choices[0].message.content
 
        elif model_choice == "Gemini 1.5 Flash":
            model = genai.GenerativeModel("gemini-1.5-flash")
            history = []
            for m in augmented_messages[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})
            chat = model.start_chat(history=history)
            return chat.send_message(augmented_messages[-1]["content"]).text
 
        elif model_choice == "Command R (Cohere)":
            cohere_messages = [
                {"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]}
                for m in augmented_messages
            ]
            response = cohere_client.chat(model="command-r", messages=cohere_messages)
            return response.message.content[0].text
 
        elif model_choice == "Mistral Small":
            headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
            data = {"model": "mistral-small-latest", "messages": augmented_messages}
            resp = requests.post("https://api.mistral.ai/v1/chat/completions",
                                 headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
 
    except Exception as e:
        return f"⚠️ Error: {str(e)}"
 
def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r") as f:
            return json.load(f)
    return {}
 
def save_chats(chats):
    with open(CHATS_FILE, "w") as f:
        json.dump(chats, f)
 
def get_greeting():
    hour = datetime.now().hour
    if hour < 12:   return "Good morning"
    elif hour < 17: return "Good afternoon"
    else:           return "Good evening"
 
def count_user_messages(chats_dict):
    total = 0
    for chat in chats_dict.values():
        for msg in chat.get("messages", []):
            if msg["role"] == "user":
                total += 1
    return total
 
# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Surya Dev AI", page_icon="🤖", layout="wide")
 
# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
 
#MainMenu, footer, header { visibility: hidden; }
* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; box-sizing: border-box; }
.stApp { background-color: #faf9f7; color: #1a1a1a; }
 
/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: #f0ede8;
    border-right: 1px solid #e5e1da;
    width: 260px !important;
}
section[data-testid="stSidebar"] > div { padding: 0 12px; }
 
[data-testid="collapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    color: #1a1a1a !important;
    background-color: #f0ede8 !important;
    border-radius: 0 8px 8px 0 !important;
    border: 1px solid #e5e1da !important;
}
 
.stButton > button {
    background: transparent;
    color: #44403c;
    border: none;
    border-radius: 8px;
    width: 100%;
    text-align: left;
    padding: 8px 10px;
    font-size: 13px;
    font-weight: 400;
    transition: background 0.15s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.stButton > button:hover {
    background-color: #e5e1da !important;
    color: #1a1a1a !important;
    border: none !important;
    box-shadow: none !important;
}
 
.new-chat-btn > div > button {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border: 1px solid #d6d3cc !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
}
.new-chat-btn > div > button:hover {
    background-color: #f0ede8 !important;
    border-color: #b8b3ab !important;
}
 
.stSelectbox > div > div {
    background-color: #ffffff !important;
    border: 1px solid #d6d3cc !important;
    border-radius: 8px !important;
    color: #1a1a1a !important;
    font-size: 13px !important;
}
 
.stChatInput { border-top: 1px solid #e5e1da; padding-top: 12px; }
.stChatInput textarea {
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    border-radius: 16px !important;
    border: 1px solid #d6d3cc !important;
    font-size: 15px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    resize: none !important;
}
.stChatInput textarea:focus {
    border-color: #d97706 !important;
    box-shadow: 0 0 0 3px rgba(217,119,6,0.1) !important;
}
 
.stChatMessage {
    background: transparent !important;
    border: none !important;
    max-width: 720px;
    margin: 0 auto;
}
[data-testid="stChatMessageContent"] { padding: 0 !important; }
 
.model-tag {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    font-size: 11px;
    font-weight: 500;
    border-radius: 6px;
    padding: 2px 8px;
    margin-bottom: 4px;
    border: 1px solid #fde68a;
}
 
.search-tag {
    display: inline-block;
    background: #eff6ff;
    color: #1d4ed8;
    font-size: 11px;
    font-weight: 500;
    border-radius: 6px;
    padding: 2px 8px;
    margin-bottom: 4px;
    margin-left: 4px;
    border: 1px solid #bfdbfe;
}
 
.profile-card {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px;
    border-radius: 10px;
    background: #ffffff;
    border: 1px solid #e5e1da;
    margin-top: 8px;
}
.profile-pic { width: 34px; height: 34px; border-radius: 50%; object-fit: cover; }
.profile-name { font-weight: 600; font-size: 13px; color: #1a1a1a; }
.profile-email { font-size: 11px; color: #78716c; }
 
.signout-btn > div > button {
    background: transparent !important;
    color: #78716c !important;
    border: 1px solid #d6d3cc !important;
    border-radius: 8px !important;
    font-size: 12px !important;
}
.signout-btn > div > button:hover { color: #dc2626 !important; border-color: #dc2626 !important; }
 
.section-label {
    font-size: 11px;
    font-weight: 600;
    color: #a8a29e;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 2px;
    margin-top: 8px;
}
 
hr { border: none; border-top: 1px solid #e5e1da !important; margin: 10px 0 !important; }
 
/* ── Login page ── */
.login-card {
    background: #ffffff;
    border: 1px solid #e5e1da;
    border-radius: 20px;
    padding: 36px 32px;
    width: 100%;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06);
}
.login-feature-item {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: #44403c;
    margin-bottom: 10px;
}
.login-feature-icon {
    width: 28px; height: 28px;
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px;
    flex-shrink: 0;
}
 
/* ── Account page ── */
.account-header {
    background: linear-gradient(135deg, #fef3c7 0%, #fffbf5 100%);
    border: 1px solid #fde68a;
    border-radius: 20px;
    padding: 32px;
    display: flex;
    align-items: center;
    gap: 20px;
    margin-bottom: 24px;
}
.account-avatar {
    width: 72px; height: 72px;
    border-radius: 50%;
    border: 3px solid #d97706;
    object-fit: cover;
    flex-shrink: 0;
}
.account-name { font-size: 22px; font-weight: 700; color: #1a1a1a; margin: 0 0 4px 0; }
.account-email { font-size: 14px; color: #78716c; margin: 0; }
.account-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #d97706; color: white;
    font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px;
    margin-top: 8px;
}
.stat-card {
    background: #ffffff;
    border: 1px solid #e5e1da;
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.stat-number { font-size: 28px; font-weight: 700; color: #d97706; margin: 0; }
.stat-label { font-size: 12px; color: #78716c; margin: 4px 0 0 0; font-weight: 500; }
.info-card {
    background: #ffffff;
    border: 1px solid #e5e1da;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.info-card-title {
    font-size: 13px; font-weight: 600; color: #a8a29e;
    text-transform: uppercase; letter-spacing: 0.06em;
    margin: 0 0 16px 0;
}
.info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid #f5f2ed;
}
.info-row:last-child { border-bottom: none; }
.info-key { font-size: 13px; color: #78716c; font-weight: 500; }
.info-value { font-size: 13px; color: #1a1a1a; font-weight: 600; }
.model-badge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 3px 10px; border-radius: 6px;
    font-size: 12px; font-weight: 500;
    background: #fef3c7; color: #92400e; border: 1px solid #fde68a;
}
.back-btn > div > button {
    background: transparent !important;
    color: #44403c !important;
    border: 1px solid #d6d3cc !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}
.back-btn > div > button:hover { background: #f0ede8 !important; }
.danger-btn > div > button {
    background: transparent !important;
    color: #dc2626 !important;
    border: 1px solid #fca5a5 !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}
.danger-btn > div > button:hover { background: #fef2f2 !important; }
 
/* Live search indicator */
.live-badge {
    display: inline-flex; align-items: center; gap: 4px;
    background: #f0fdf4; color: #16a34a;
    border: 1px solid #bbf7d0;
    font-size: 11px; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)
 
# ── Session State ─────────────────────────────────────────────────────────────
for key, default in [
    ("authenticated", False),
    ("user_info", None),
    ("messages", []),
    ("current_chat_id", None),
    ("all_chats", load_chats()),
    ("model_choice", "Llama 3.3 (Groq)"),
    ("show_account", False),
    ("last_used_search", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default
 
# ════════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ════════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 1.1, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
 
        # Robot logo + headline
        st.markdown(f"""
            <div style='text-align:center; margin-bottom:28px;'>
                <div style='display:inline-flex; align-items:center; justify-content:center;
                    width:80px; height:80px; background:linear-gradient(135deg,#fef3c7,#fffbf5);
                    border-radius:20px; border:2px solid #fde68a; margin-bottom:16px;
                    box-shadow:0 4px 20px rgba(217,119,6,0.2);'>
                    {get_robot_logo(52)}
                </div>
                <h1 style='font-size:2rem; font-weight:700; color:#1a1a1a; margin:0 0 6px 0; letter-spacing:-0.5px;'>
                    Surya Dev AI
                </h1>
                <p style='color:#78716c; font-size:15px; margin:0;'>
                    Your intelligent AI assistant with live web search
                </p>
            </div>
        """, unsafe_allow_html=True)
 
        st.markdown("""
            <div class='login-card'>
                <p style='font-size:15px; font-weight:600; color:#1a1a1a; margin:0 0 4px 0; text-align:center;'>
                    Welcome back
                </p>
                <p style='font-size:13px; color:#78716c; text-align:center; margin:0 0 20px 0;'>
                    Sign in to continue to Surya Dev AI
                </p>
                <div style='margin-bottom:20px;'>
                    <div class='login-feature-item'>
                        <div class='login-feature-icon' style='background:#fef3c7;'>🤖</div>
                        <span>4 powerful AI models — Llama, Gemini, Cohere, Mistral</span>
                    </div>
                    <div class='login-feature-item'>
                        <div class='login-feature-icon' style='background:#eff6ff;'>🌐</div>
                        <span>Live web search from Wikipedia & DuckDuckGo</span>
                    </div>
                    <div class='login-feature-item'>
                        <div class='login-feature-icon' style='background:#f0fdf4;'>💬</div>
                        <span>Chat history saved across sessions</span>
                    </div>
                    <div class='login-feature-item'>
                        <div class='login-feature-icon' style='background:#fdf4ff;'>🔒</div>
                        <span>Secure login with your Google account</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, TOKEN_URL, TOKEN_URL)
        result = oauth2.authorize_button(
            "Continue with Google",
            redirect_uri=REDIRECT_URI,
            scope=SCOPE,
            key="google",
            extras_params={"prompt": "consent", "access_type": "offline"},
            use_container_width=True
        )
 
        st.markdown("""
            <p style='text-align:center; font-size:11px; color:#a8a29e; margin-top:16px;'>
                By continuing, you agree to Surya Dev AI's terms of use
            </p>
        """, unsafe_allow_html=True)
 
        if result and "token" in result:
            payload = jwt.decode(result["token"]["id_token"], options={"verify_signature": False})
            st.session_state.authenticated = True
            st.session_state.user_info = payload
            st.session_state.login_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
            st.rerun()
 
# ════════════════════════════════════════════════════════════════════════════════
# ACCOUNT PAGE
# ════════════════════════════════════════════════════════════════════════════════
elif st.session_state.show_account:
    user       = st.session_state.user_info
    user_email = user.get("email", "")
    user_name  = user.get("name", "User") or "User"
    user_pic   = user.get("picture", "")
    all_chats  = st.session_state.all_chats
    user_chats = all_chats.get(user_email, {})
 
    total_chats    = len(user_chats)
    total_messages = count_user_messages(user_chats)
    member_since   = st.session_state.get("login_time", datetime.now().strftime("%d %b %Y, %I:%M %p"))
    first_initial  = user_name[0].upper() if user_name else "U"
 
    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='back-btn'>", unsafe_allow_html=True)
        if st.button("← Back to chat"):
            st.session_state.show_account = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
 
        if user_pic:
            avatar_html = f"<img src='{user_pic}' class='account-avatar'/>"
        else:
            avatar_html = f"<div style='width:72px;height:72px;border-radius:50%;background:linear-gradient(135deg,#d97706,#f59e0b);display:flex;align-items:center;justify-content:center;font-size:28px;color:white;font-weight:700;flex-shrink:0;'>{first_initial}</div>"
 
        st.markdown(f"""
            <div class='account-header'>
                {avatar_html}
                <div>
                    <p class='account-name'>{user_name}</p>
                    <p class='account-email'>{user_email}</p>
                    <div class='account-badge'>🤖 Surya Dev AI Member</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
 
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='stat-card'><p class='stat-number'>{total_chats}</p><p class='stat-label'>Total Chats</p></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='stat-card'><p class='stat-number'>{total_messages}</p><p class='stat-label'>Messages Sent</p></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='stat-card'><p class='stat-number'>4</p><p class='stat-label'>AI Models</p></div>", unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        st.markdown(f"""
            <div class='info-card'>
                <p class='info-card-title'>Account Details</p>
                <div class='info-row'><span class='info-key'>Full Name</span><span class='info-value'>{user_name}</span></div>
                <div class='info-row'><span class='info-key'>Email Address</span><span class='info-value'>{user_email}</span></div>
                <div class='info-row'><span class='info-key'>Login Provider</span><span class='info-value'>🔵 Google</span></div>
                <div class='info-row'><span class='info-key'>Session Started</span><span class='info-value'>{member_since}</span></div>
                <div class='info-row'><span class='info-key'>Account Status</span><span class='info-value' style='color:#16a34a;'>✅ Active</span></div>
            </div>
        """, unsafe_allow_html=True)
 
        current_model = st.session_state.get("model_choice", "Llama 3.3 (Groq)")
        st.markdown(f"""
            <div class='info-card'>
                <p class='info-card-title'>Preferences</p>
                <div class='info-row'><span class='info-key'>Active AI Model</span><span class='model-badge'>⚡ {current_model}</span></div>
                <div class='info-row'><span class='info-key'>Live Web Search</span><span class='info-value' style='color:#16a34a;'>✅ Enabled</span></div>
                <div class='info-row'><span class='info-key'>Search Sources</span><span class='info-value'>🌐 Wikipedia + DuckDuckGo</span></div>
                <div class='info-row'><span class='info-key'>Chat History</span><span class='info-value'>Saved locally</span></div>
            </div>
        """, unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='danger-btn'>", unsafe_allow_html=True)
            if st.button("🗑️ Clear all chats", use_container_width=True):
                all_chats[user_email] = {}
                st.session_state.all_chats = all_chats
                st.session_state.messages = []
                st.session_state.current_chat_id = None
                save_chats(all_chats)
                st.success("All chats cleared!")
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            st.markdown("<div class='danger-btn'>", unsafe_allow_html=True)
            if st.button("🚪 Sign out", use_container_width=True):
                for key in ["authenticated", "user_info", "messages", "current_chat_id", "show_account"]:
                    st.session_state[key] = False if key == "authenticated" else None if "info" in key or "id" in key else (False if key == "show_account" else [])
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
 
        st.markdown("<br><br>", unsafe_allow_html=True)
 
# ════════════════════════════════════════════════════════════════════════════════
# MAIN CHAT APP
# ════════════════════════════════════════════════════════════════════════════════
else:
    user       = st.session_state.user_info
    user_email = user.get("email", "default")
    user_name  = user.get("name", "User") or "User"
    first_name = user_name.split()[0] if user_name.split() else "there"
    user_pic   = user.get("picture", "")
    first_initial = user_name[0].upper() if user_name else "U"
 
    all_chats = st.session_state.all_chats
    if user_email not in all_chats:
        all_chats[user_email] = {}
    user_chats = all_chats[user_email]
 
    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
 
        # Robot logo + title
        st.markdown(f"""
            <div style='display:flex; align-items:center; gap:8px; padding:0 2px; margin-bottom:16px;'>
                <div style='width:32px;height:32px;background:linear-gradient(135deg,#fef3c7,#fff);
                    border-radius:8px;border:1px solid #fde68a;
                    display:flex;align-items:center;justify-content:center;'>
                    {get_robot_logo(22)}
                </div>
                <span style='font-size:15px;font-weight:700;color:#1a1a1a;letter-spacing:-0.3px;'>Surya Dev AI</span>
            </div>
        """, unsafe_allow_html=True)
 
        st.markdown("<div class='new-chat-btn'>", unsafe_allow_html=True)
        if st.button("✏️  New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        st.markdown("<div class='section-label'>Model</div>", unsafe_allow_html=True)
        st.session_state.model_choice = st.selectbox(
            "Model",
            ["Llama 3.3 (Groq)", "Gemini 1.5 Flash", "Command R (Cohere)", "Mistral Small"],
            label_visibility="collapsed"
        )
 
        # Live search toggle
        st.markdown("<div class='section-label' style='margin-top:12px;'>Live Search</div>", unsafe_allow_html=True)
        live_search_on = st.toggle("🌐 Wikipedia & Web", value=True)
 
        st.divider()
 
        if user_chats:
            st.markdown("<div class='section-label'>Recent</div>", unsafe_allow_html=True)
            for chat_id, chat_data in sorted(user_chats.items(), reverse=True):
                title = chat_data.get("title", "Untitled chat")
                if st.button(f"{title[:28]}…" if len(title) > 28 else title, key=chat_id):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.messages = chat_data["messages"]
                    st.rerun()
 
        st.markdown("<br>" * 4, unsafe_allow_html=True)
        st.divider()
 
        if user_pic:
            pic_html = f"<img src='{user_pic}' class='profile-pic'/>"
        else:
            pic_html = f"<div style='width:34px;height:34px;border-radius:50%;background:#d97706;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:14px;'>{first_initial}</div>"
 
        st.markdown(f"""
            <div class='profile-card'>
                {pic_html}
                <div style='flex:1;min-width:0;'>
                    <div class='profile-name'>{user_name}</div>
                    <div class='profile-email'>{user_email}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
        col_acc, col_out = st.columns(2)
        with col_acc:
            if st.button("👤 Account", use_container_width=True):
                st.session_state.show_account = True
                st.rerun()
        with col_out:
            st.markdown("<div class='signout-btn'>", unsafe_allow_html=True)
            if st.button("Sign out", use_container_width=True):
                for key in ["authenticated", "user_info", "messages", "current_chat_id"]:
                    st.session_state[key] = False if key == "authenticated" else None if "info" in key or "id" in key else []
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)
 
    # ── Main content ──────────────────────────────────────────────────────────
    _, center, _ = st.columns([1, 3, 1])
 
    with center:
        if len(st.session_state.messages) == 0:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"""
                <div style='text-align:center; margin-bottom:36px;'>
                    <div style='display:inline-flex;align-items:center;justify-content:center;
                        width:72px;height:72px;background:linear-gradient(135deg,#fef3c7,#fff);
                        border-radius:18px;border:2px solid #fde68a;margin-bottom:16px;
                        box-shadow:0 4px 20px rgba(217,119,6,0.15);'>
                        {get_robot_logo(48)}
                    </div>
                    <h1 style='font-size:2.2rem; font-weight:700; color:#1a1a1a; margin:0 0 6px 0; letter-spacing:-0.5px;'>
                        {get_greeting()}, {first_name}!
                    </h1>
                    <p style='color:#78716c; font-size:16px; margin:0 0 8px 0;'>How can I help you today?</p>
                    <span style='display:inline-flex;align-items:center;gap:5px;background:#f0fdf4;color:#16a34a;border:1px solid #bbf7d0;font-size:12px;font-weight:600;padding:3px 12px;border-radius:20px;'>
                        🌐 Live search {"ON" if live_search_on else "OFF"}
                    </span>
                </div>
            """, unsafe_allow_html=True)
 
            suggestions = [
                ("What is the latest news about AI?", "🌐"),
                ("Tell me about the history of India", "📚"),
                ("Explain machine learning like I'm 10", "🧠"),
                ("Help me plan a productive morning routine", "📋"),
            ]
            c1, c2 = st.columns(2)
            for i, (suggestion, icon) in enumerate(suggestions):
                col = c1 if i % 2 == 0 else c2
                if col.button(f"{icon}  {suggestion}", use_container_width=True, key=f"sug_{i}"):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    with st.spinner("🔍 Searching & thinking..."):
                        reply = get_ai_response(st.session_state.messages, st.session_state.model_choice) if live_search_on else get_ai_response_no_search(st.session_state.messages, st.session_state.model_choice)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    chat_id = st.session_state.current_chat_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
                    st.session_state.current_chat_id = chat_id
                    all_chats[user_email][chat_id] = {"title": suggestion[:40], "messages": st.session_state.messages}
                    st.session_state.all_chats = all_chats
                    save_chats(all_chats)
                    st.rerun()
 
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    if message["role"] == "assistant":
                        used_search = needs_live_search(
                            st.session_state.messages[i-1]["content"] if i > 0 else ""
                        ) and live_search_on
                        tags = f"<div class='model-tag'>{st.session_state.model_choice}</div>"
                        if used_search:
                            tags += " <div class='search-tag'>🌐 Live Search</div>"
                        st.markdown(tags, unsafe_allow_html=True)
                    st.markdown(message["content"])
 
        if prompt := st.chat_input("Message Surya Dev AI… (try 'what is' or 'latest news about')"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
 
            is_live = needs_live_search(prompt) and live_search_on
            spinner_msg = "🔍 Searching Wikipedia & web..." if is_live else "🤔 Thinking..."
 
            with st.spinner(spinner_msg):
                reply = get_ai_response(st.session_state.messages, st.session_state.model_choice)
 
            with st.chat_message("assistant"):
                tags = f"<div class='model-tag'>{st.session_state.model_choice}</div>"
                if is_live:
                    tags += " <div class='search-tag'>🌐 Live Search</div>"
                st.markdown(tags, unsafe_allow_html=True)
                st.markdown(reply)
 
            st.session_state.messages.append({"role": "assistant", "content": reply})
            chat_id = st.session_state.current_chat_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
            st.session_state.current_chat_id = chat_id
            all_chats[user_email][chat_id] = {
                "title": st.session_state.messages[0]["content"][:40],
                "messages": st.session_state.messages
            }
            st.session_state.all_chats = all_chats
            save_chats(all_chats)
            st.rerun()
 
