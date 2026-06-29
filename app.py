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
import io
from datetime import datetime
from urllib.parse import quote

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY    = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY  = st.secrets["GEMINI_API_KEY"]
COHERE_API_KEY  = st.secrets["COHERE_API_KEY"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
CLIENT_ID       = st.secrets["CLIENT_ID"]
CLIENT_SECRET   = st.secrets["CLIENT_SECRET"]
AUTHORIZE_URL   = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL       = "https://oauth2.googleapis.com/token"
REDIRECT_URI    = st.secrets.get("REDIRECT_URI", "https://nz5ossng47a243vac5nrti.streamlit.app")
SCOPE           = "openid email profile"
CHATS_FILE      = "chats.json"

groq_client   = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

# ── Robot SVG logo ─────────────────────────────────────────────────────────────
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

# ── Models ─────────────────────────────────────────────────────────────────────
MODELS = {
    "GPT-4o (Llama 3.3)":    {"icon": "⚡", "tag": "groq",    "provider": "Groq"},
    "Gemini 1.5 Flash":       {"icon": "✦",  "tag": "gemini",  "provider": "Google"},
    "Command R (Cohere)":     {"icon": "◈",  "tag": "cohere",  "provider": "Cohere"},
    "Mistral Small":          {"icon": "◆",  "tag": "mistral", "provider": "Mistral"},
}

# ── Storage ────────────────────────────────────────────────────────────────────
def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE) as f:
            return json.load(f)
    return {}

def save_chats(c):
    with open(CHATS_FILE, "w") as f:
        json.dump(c, f)

def total_msgs(chats):
    return sum(1 for ch in chats.values() for m in ch.get("messages", []) if m["role"] == "user")

# ── Live search ────────────────────────────────────────────────────────────────
def wiki_search(q):
    try:
        r = requests.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(q)}", timeout=5)
        if r.status_code == 200:
            d = r.json()
            if d.get("extract"):
                return d["extract"][:900]
    except:
        pass
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
    except:
        pass
    return ""

LIVE_KW = [
    "latest", "current", "today", "now", "recent", "news", "live", "2024", "2025", "2026",
    "who is", "what is", "when did", "where is", "how much", "price", "weather", "score",
    "wikipedia", "tell me about", "search", "look up", "what happened", "update", "trending",
    "population", "capital of", "president of", "ceo of", "founder of", "born", "died", "history of"
]

def needs_search(q):
    return any(k in q.lower() for k in LIVE_KW)

# ── Image Generation via Pollinations ────────────────────────────────────────────
def generate_image(prompt):
    """Generate image using Pollinations.ai (free, no API key needed)"""
    try:
        encoded = quote(prompt)
        # Use pollinations.ai for free image generation
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&enhance=true"
        return url, None
    except Exception as e:
        return None, str(e)

def is_image_request(prompt):
    """Detect if user wants image generation"""
    img_kw = [
        "generate image", "create image", "draw", "make image", "paint",
        "generate a picture", "create a picture", "image of", "picture of",
        "illustration of", "sketch of", "photo of", "generate photo",
        "create art", "make art", "design image", "visualize",
        "generate an image", "create an image", "make a picture",
        "draw me", "show me an image", "generate artwork", "create artwork"
    ]
    p_lower = prompt.lower()
    return any(k in p_lower for k in img_kw)

# ── AI Response ────────────────────────────────────────────────────────────────
def clean_msgs(messages):
    return [{"role": m["role"], "content": m["content"]} for m in messages]

def get_response(messages, model, use_live=True):
    try:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        aug = clean_msgs(messages)
        searched = False
        if use_live and needs_search(last):
            topic = re.sub(r'^(what is|who is|tell me about|search|find|look up|when did|where is)\s+', '', last.lower()).strip()
            ctx = ddg_search(last) + "\n" + wiki_search(topic)
            if ctx.strip():
                searched = True
                inject = {"role": "user", "content":
                    f"Live web info:\n---\n{ctx[:1500]}\n---\nUse this to answer accurately. Question: {last}"}
                aug = clean_msgs(messages[:-1]) + [inject]

        if model == "GPT-4o (Llama 3.3)":
            r = groq_client.chat.completions.create(model="llama-3.3-70b-versatile", messages=aug)
            return r.choices[0].message.content, searched
        elif model == "Gemini 1.5 Flash":
            gm = genai.GenerativeModel("gemini-1.5-flash")
            hist = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in aug[:-1]]
            ch = gm.start_chat(history=hist)
            return ch.send_message(aug[-1]["content"]).text, searched
        elif model == "Command R (Cohere)":
            cm = [{"role": "user" if m["role"] == "user" else "assistant", "content": m["content"]} for m in aug]
            r = cohere_client.chat(model="command-r", messages=cm)
            return r.message.content[0].text, searched
        elif model == "Mistral Small":
            h = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
            r = requests.post("https://api.mistral.ai/v1/chat/completions",
                headers=h, json={"model": "mistral-small-latest", "messages": aug}, timeout=30)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"], searched
    except Exception as e:
        return f"⚠️ Error: {str(e)}", False

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Surya Dev AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── VOICE CHAT + IMAGE CREATOR JS ─────────────────────────────────────────────
VOICE_JS = """
<script>
// ── Voice Recognition ──────────────────────────────────────────────────────
let recognition = null;
let isListening = false;

function startVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        alert('Voice recognition not supported in this browser. Please use Chrome.');
        return;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    const btn = document.getElementById('voice-btn');
    const status = document.getElementById('voice-status');

    recognition.onstart = () => {
        isListening = true;
        if (btn) { btn.classList.add('listening'); btn.textContent = '🔴 Listening...'; }
        if (status) status.textContent = 'Listening... speak now';
    };

    recognition.onresult = (e) => {
        let transcript = '';
        for (let i = e.resultIndex; i < e.results.length; i++) {
            transcript += e.results[i][0].transcript;
        }
        if (status) status.textContent = '📝 ' + transcript;
        if (e.results[e.results.length - 1].isFinal) {
            // Put text into Streamlit chat input
            const chatInput = document.querySelector('[data-testid="stChatInput"] textarea');
            if (chatInput) {
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
                nativeInputValueSetter.call(chatInput, transcript);
                chatInput.dispatchEvent(new Event('input', { bubbles: true }));
            }
        }
    };

    recognition.onerror = (e) => {
        isListening = false;
        if (btn) { btn.classList.remove('listening'); btn.textContent = '🎤'; }
        if (status) status.textContent = 'Error: ' + e.error;
    };

    recognition.onend = () => {
        isListening = false;
        if (btn) { btn.classList.remove('listening'); btn.textContent = '🎤'; }
        if (status) status.textContent = '';
    };

    recognition.start();
}

function stopVoice() {
    if (recognition && isListening) {
        recognition.stop();
    }
}

// ── Text to Speech ─────────────────────────────────────────────────────────
function speakText(text) {
    if (!('speechSynthesis' in window)) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    // Try to pick a good voice
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v => v.name.includes('Google') || v.name.includes('Natural')) || voices[0];
    if (preferred) utterance.voice = preferred;
    window.speechSynthesis.speak(utterance);
}

// Auto-speak last assistant message if voice mode is on
window.addEventListener('load', () => {
    setTimeout(() => {
        window.speechSynthesis.getVoices(); // pre-load voices
    }, 100);
});
</script>
"""

# ── FULL CSS ──────────────────────────────────────────────────────────────────
MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Söhne:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

/* ═══ RESET & BASE ═══ */
html, body, [class*="css"], .stApp {
    background-color: #212121 !important;
    color: #ececec !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ═══ CSS VARIABLES ═══ */
:root {
    --text-color: #ececec !important;
    --background-color: #212121 !important;
    --secondary-background-color: #171717 !important;
    --primary-color: #10a37f !important;
    --sidebar-bg: #171717;
    --surface: #2f2f2f;
    --border: #383838;
    --green: #10a37f;
    --green-dark: #0d8a6a;
    --muted: #8e8ea0;
}

/* ═══ SIDEBAR ═══ */
[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #2d2d2d !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child {
    background-color: #171717 !important;
    padding-top: 0 !important;
}

/* ── FIX: Sidebar collapse/expand button ── */
[data-testid="collapsedControl"] {
    background-color: #10a37f !important;
    color: white !important;
    border-radius: 0 6px 6px 0 !important;
    border: none !important;
    width: 26px !important;
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 99999 !important;
    top: 20px !important;
    box-shadow: 2px 0 12px rgba(16,163,127,0.4) !important;
    transition: background 0.2s !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] svg path {
    color: white !important;
    fill: white !important;
    stroke: white !important;
}
[data-testid="collapsedControl"]:hover {
    background-color: #0d8a6a !important;
}
/* Sidebar expand arrow (inside open sidebar top) */
button[data-testid="baseButton-header"] {
    color: #ececec !important;
}

/* ═══ SIDEBAR TEXT ═══ */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {
    color: #ececec !important;
}

/* ═══ SIDEBAR BUTTONS ═══ */
[data-testid="stSidebar"] .stButton button {
    background-color: transparent !important;
    color: #d0d0d0 !important;
    border: none !important;
    border-radius: 8px !important;
    width: 100% !important;
    text-align: left !important;
    padding: 10px 12px !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    font-family: 'Inter', sans-serif !important;
    margin-bottom: 2px !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background-color: #2e2e2e !important;
    color: #ffffff !important;
}

/* ═══ SELECTBOX ═══ */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #2a2a2a !important;
    color: #ececec !important;
    border: 1px solid #383838 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] { background-color: #2a2a2a !important; }
[data-testid="stSidebar"] [data-baseweb="select"] div { color: #ececec !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #888 !important; font-size: 11px !important; }

/* ═══ TOGGLE FIX ─ This is the key fix ═══ */
/* Streamlit renders toggle as a checkbox-like element */
[data-testid="stSidebar"] .stToggle {
    display: flex !important;
    align-items: center !important;
    padding: 6px 12px !important;
}
[data-testid="stSidebar"] .stToggle > label {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    cursor: pointer !important;
    color: #d0d0d0 !important;
    font-size: 13.5px !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stToggle > label > div:first-child {
    /* The track */
    width: 40px !important;
    height: 22px !important;
    border-radius: 11px !important;
    background-color: #3a3a3a !important;
    position: relative !important;
    transition: background 0.2s !important;
    flex-shrink: 0 !important;
    display: block !important;
    visibility: visible !important;
}
/* When checked — track turns green */
[data-testid="stSidebar"] input[type="checkbox"]:checked + div,
[data-testid="stSidebar"] .stToggle input:checked ~ label > div:first-child {
    background-color: #10a37f !important;
}
/* Actual toggle input hidden but not display:none so it still works */
[data-testid="stSidebar"] .stToggle input[type="checkbox"] {
    position: absolute !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
}
/* Streamlit 1.x toggle selectors */
div[data-testid="stToggle"] {
    background: transparent !important;
    padding: 4px 0 !important;
}
div[data-testid="stToggle"] label {
    color: #d0d0d0 !important;
    font-size: 13.5px !important;
}
div[data-testid="stToggle"] > div {
    display: flex !important;
    align-items: center !important;
}
/* The actual toggle pill */
div[data-testid="stToggle"] span[role="switch"] {
    background-color: #3a3a3a !important;
    border-color: #555 !important;
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
div[data-testid="stToggle"] span[role="switch"][aria-checked="true"] {
    background-color: #10a37f !important;
    border-color: #10a37f !important;
}
div[data-testid="stToggle"] span[role="switch"] > span {
    background-color: white !important;
}
/* Also target any generic toggle patterns */
[role="switch"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}
[role="switch"][aria-checked="true"] {
    background-color: #10a37f !important;
}

/* ═══ DIVIDERS ═══ */
[data-testid="stSidebar"] hr {
    border: none !important;
    border-top: 1px solid #2a2a2a !important;
    margin: 4px 0 !important;
}

/* ═══ CHAT MESSAGES ═══ */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 20px 0 !important;
    max-width: 48rem !important;
    margin: 0 auto !important;
}
[data-testid="stChatMessageContent"],
[data-testid="stChatMessageContent"] *,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] ol,
[data-testid="stMarkdownContainer"] ul,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #ececec !important;
    background-color: transparent !important;
}
[data-testid="stChatMessageContent"] strong,
[data-testid="stMarkdownContainer"] strong {
    color: #ffffff !important;
    font-weight: 600 !important;
}
[data-testid="stMarkdownContainer"] ol li,
[data-testid="stMarkdownContainer"] ul li {
    color: #ececec !important;
    margin-bottom: 6px !important;
    line-height: 1.7 !important;
}
[data-testid="stMarkdownContainer"] code {
    background-color: #2d2d2d !important;
    color: #e2e2e2 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
    font-size: 13px !important;
}
[data-testid="stMarkdownContainer"] pre {
    background-color: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    padding: 16px !important;
}
[data-testid="stMarkdownContainer"] pre code {
    color: #e2e2e2 !important;
    background: transparent !important;
}
/* User message bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: transparent !important;
}
[data-testid="chatAvatarIcon-user"] { background-color: #19c37d !important; }
[data-testid="chatAvatarIcon-assistant"] { background-color: #19c37d !important; }

/* ═══ GENERAL TEXT ═══ */
.stMarkdown p, .stMarkdown span, .stMarkdown li,
.element-container p, .element-container span {
    color: #ececec !important;
}

/* ═══ CHAT INPUT ═══ */
[data-testid="stBottom"] {
    background: linear-gradient(to top, #212121 80%, transparent) !important;
    border-top: none !important;
    padding: 8px 0 20px !important;
}
[data-testid="stBottom"] > div { background-color: transparent !important; }
[data-testid="stChatInput"] {
    background-color: #2f2f2f !important;
    border: 1px solid #444 !important;
    border-radius: 16px !important;
    max-width: 48rem !important;
    margin: 0 auto !important;
    box-shadow: 0 0 0 1px #3a3a3a !important;
    transition: box-shadow 0.2s !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #555 !important;
    box-shadow: 0 0 0 1px #555 !important;
}
[data-testid="stChatInput"] textarea {
    background-color: #2f2f2f !important;
    color: #ececec !important;
    border: none !important;
    font-size: 15px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 14px 50px 14px 18px !important;
    line-height: 1.5 !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #8e8ea0 !important; }
[data-testid="stChatInput"] button {
    background-color: #10a37f !important;
    border-radius: 10px !important;
    color: white !important;
    border: none !important;
    margin-right: 8px !important;
    transition: background 0.2s !important;
}
[data-testid="stChatInput"] button:hover { background-color: #0d8a6a !important; }
[data-testid="stChatInput"] button:disabled { background-color: #3a3a3a !important; }

/* ═══ SCROLLBAR ═══ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3a3a3a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4a4a4a; }

/* ═══ MODEL BADGES ═══ */
.mbadge {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 11px; font-weight: 600; padding: 2px 10px;
    border-radius: 20px; margin-bottom: 8px;
}
.mbadge.groq    { background: #0d3b30; color: #1dc8a0 !important; border: 1px solid #0d8a6a; }
.mbadge.gemini  { background: #0d2060; color: #7baaf7 !important; border: 1px solid #2d5dbf; }
.mbadge.cohere  { background: #3d2200; color: #f59e0b !important; border: 1px solid #92400e; }
.mbadge.mistral { background: #2d1060; color: #c084fc !important; border: 1px solid #7c3aed; }
.mbadge.live    { background: #0d2d1a; color: #4ade80 !important; border: 1px solid #166534; margin-left: 5px; }
.mbadge.img     { background: #1a0d3b; color: #a78bfa !important; border: 1px solid #5b21b6; margin-left: 5px; }
.mbadge.voice   { background: #0d1a3b; color: #60a5fa !important; border: 1px solid #1e40af; margin-left: 5px; }

/* ═══ SUGGESTION CHIPS ═══ */
.chip .stButton button {
    background-color: #2a2a2a !important;
    color: #d0d0d0 !important;
    border: 1px solid #383838 !important;
    border-radius: 12px !important;
    font-size: 13.5px !important;
    text-align: left !important;
    padding: 14px 16px !important;
    height: auto !important;
    white-space: normal !important;
    line-height: 1.45 !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
.chip .stButton button:hover {
    background-color: #333 !important;
    border-color: #555 !important;
    color: #fff !important;
}

/* ═══ NEW CHAT BUTTON ═══ */
.new-chat-wrap .stButton button {
    background-color: transparent !important;
    color: #ececec !important;
    border: 1px solid #383838 !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    padding: 10px 14px !important;
    transition: all 0.15s !important;
}
.new-chat-wrap .stButton button:hover {
    background-color: #2a2a2a !important;
    border-color: #555 !important;
}

/* ═══ SIGN OUT ═══ */
.signout-wrap .stButton button {
    background-color: transparent !important;
    color: #888 !important;
    border: 1px solid transparent !important;
    font-size: 13px !important;
    padding: 8px 12px !important;
}
.signout-wrap .stButton button:hover {
    color: #ff6b6b !important;
    background-color: #2a1a1a !important;
}

/* ═══ BACK BUTTON ═══ */
.back-btn .stButton button {
    background-color: #2a2a2a !important;
    color: #ececec !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
    width: auto !important;
}

/* ═══ HISTORY ITEMS ═══ */
.hist-item .stButton button {
    background-color: transparent !important;
    color: #b0b0b0 !important;
    border: none !important;
    font-size: 13px !important;
    text-align: left !important;
    padding: 7px 12px !important;
    border-radius: 6px !important;
}
.hist-item .stButton button:hover {
    background-color: #2a2a2a !important;
    color: #ffffff !important;
}

/* ═══ ACCOUNT & INFO BOXES ═══ */
.acc-card {
    background: linear-gradient(135deg, #1a3a30, #1e2a25);
    border: 1px solid #2a5a42; border-radius: 16px;
    padding: 28px; display: flex; align-items: center; gap: 18px; margin-bottom: 20px;
}
.acc-name  { font-size: 22px; font-weight: 700; color: #ececec; }
.acc-email { font-size: 13px; color: #888; margin-top: 4px; }
.acc-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: #10a37f; color: white; font-size: 11px; font-weight: 600;
    padding: 3px 12px; border-radius: 20px; margin-top: 8px;
}
.stat-box  {
    background: #2a2a2a; border: 1px solid #333; border-radius: 12px;
    padding: 20px; text-align: center;
}
.stat-n    { font-size: 28px; font-weight: 700; color: #10a37f; }
.stat-l    { font-size: 12px; color: #888; margin-top: 4px; }
.info-box  {
    background: #2a2a2a; border: 1px solid #333; border-radius: 12px;
    padding: 18px 22px; margin-bottom: 14px;
}
.info-ttl  { font-size: 11px; font-weight: 600; color: #666; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 14px; }
.irow      { display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0; border-bottom: 1px solid #2e2e2e; }
.irow:last-child { border-bottom: none; }
.ik { font-size: 13px; color: #888; }
.iv { font-size: 13px; color: #ececec; font-weight: 600; }

/* ═══ LOGIN ═══ */
.login-card {
    background: #2a2a2a; border: 1px solid #333; border-radius: 18px;
    padding: 36px 32px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}
.fi { display: flex; align-items: center; gap: 10px; font-size: 14px; color: #999; margin-bottom: 12px; }
.fi-icon { width: 30px; height: 30px; border-radius: 8px; display: flex;
    align-items: center; justify-content: center; font-size: 15px; flex-shrink: 0; }

/* ═══ VOICE BUTTON ═══ */
.voice-btn-wrap {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    margin: 4px 0;
}
.voice-btn {
    width: 42px; height: 42px; border-radius: 50%;
    background: #2a2a2a; border: 2px solid #444;
    color: #ececec; font-size: 18px; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    transition: all 0.2s; flex-shrink: 0;
}
.voice-btn:hover { background: #333; border-color: #10a37f; }
.voice-btn.listening { background: #3a0d0d; border-color: #ef4444; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(239,68,68,0.4)} 50%{box-shadow:0 0 0 8px rgba(239,68,68,0)} }
.voice-status { font-size: 12px; color: #888; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ═══ IMAGE GENERATION DISPLAY ═══ */
.img-container {
    border-radius: 16px; overflow: hidden;
    border: 1px solid #333; margin: 12px 0;
    max-width: 512px;
}
.img-container img { width: 100%; display: block; }
.img-actions {
    display: flex; gap: 8px; padding: 12px;
    background: #1e1e1e; border-top: 1px solid #333;
}
.img-action-btn {
    background: #2a2a2a; border: 1px solid #383838;
    color: #d0d0d0; border-radius: 8px; padding: 6px 14px;
    font-size: 12px; cursor: pointer; transition: all 0.15s;
}
.img-action-btn:hover { background: #333; color: #fff; border-color: #10a37f; }

/* ═══ VOICE MODE PANEL ═══ */
.voice-mode-panel {
    background: linear-gradient(135deg, #0d1a3b, #1a0d3b);
    border: 1px solid #1e3a6e;
    border-radius: 16px;
    padding: 28px;
    text-align: center;
    margin: 20px auto;
    max-width: 480px;
}
.voice-orb {
    width: 100px; height: 100px;
    border-radius: 50%;
    background: radial-gradient(circle, #10a37f, #0d5a47);
    margin: 0 auto 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 40px;
    box-shadow: 0 0 40px rgba(16,163,127,0.4);
    cursor: pointer;
    transition: all 0.2s;
}
.voice-orb:hover { transform: scale(1.05); box-shadow: 0 0 60px rgba(16,163,127,0.6); }
.voice-orb.active { animation: breathe 2s infinite; }
@keyframes breathe {
    0%,100% { transform: scale(1); box-shadow: 0 0 40px rgba(16,163,127,0.4); }
    50% { transform: scale(1.08); box-shadow: 0 0 80px rgba(16,163,127,0.7); }
}

/* ═══ IMAGE CREATOR PAGE ═══ */
.img-creator-header {
    text-align: center;
    padding: 40px 20px 20px;
}
.img-gallery {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
    margin-top: 16px;
}
.gallery-img {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #333;
    cursor: pointer;
    transition: transform 0.2s;
}
.gallery-img:hover { transform: scale(1.02); }
.gallery-img img { width: 100%; display: block; aspect-ratio: 1; object-fit: cover; }

/* ═══ TABS (for image creator styles) ═══ */
.style-chips {
    display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0;
}
.style-chip {
    background: #2a2a2a; border: 1px solid #383838;
    color: #d0d0d0; border-radius: 20px; padding: 6px 14px;
    font-size: 12px; cursor: pointer; transition: all 0.15s;
}
.style-chip:hover, .style-chip.active {
    background: #0d3b30; border-color: #10a37f; color: #1dc8a0;
}

/* ═══ MODE TABS ═══ */
.mode-tabs {
    display: flex; gap: 4px; padding: 6px 12px;
    border-bottom: 1px solid #2a2a2a; margin-bottom: 4px;
}
.mode-tab {
    padding: 6px 16px; border-radius: 8px;
    font-size: 13px; color: #888; cursor: pointer;
    transition: all 0.15s; border: none; background: transparent;
}
.mode-tab.active {
    background: #2a2a2a; color: #ececec;
}

/* Fix stImage display */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    max-width: 512px !important;
    width: 100% !important;
}

/* ═══ COPY BUTTON ON CODE ═══ */
pre { position: relative; }

/* ═══ THINKING INDICATOR ═══ */
.thinking {
    display: flex; align-items: center; gap: 8px;
    color: #888; font-size: 13px; padding: 8px 0;
}
.thinking-dots span {
    display: inline-block; width: 6px; height: 6px; border-radius: 50%;
    background: #10a37f; animation: blink 1.4s infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%,80%,100%{opacity:0.2} 40%{opacity:1} }

/* ═══ SIDEBAR SECTION LABELS ═══ */
.sidebar-section-label {
    font-size: 10px; font-weight: 700; color: #555 !important;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 8px 14px 4px; margin: 0;
    display: block;
}

/* ═══ BOTTOM USER AREA ═══ */
.user-bottom {
    position: fixed; bottom: 0; left: 0; width: 258px;
    background: #171717; border-top: 1px solid #2a2a2a;
    padding: 10px 14px 6px; z-index: 100;
}
</style>
"""

st.markdown(MAIN_CSS, unsafe_allow_html=True)
st.markdown(VOICE_JS, unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "auth": False, "user": None, "msgs": [], "chat_id": None,
    "chats": load_chats(), "model": "GPT-4o (Llama 3.3)",
    "page": "chat", "live": True, "login_time": "",
    "voice_mode": False, "tts_enabled": False,
    "img_style": "Realistic", "img_size": "1024×1024",
    "generated_images": [],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.auth:
    _, mid, _ = st.columns([1, 1.1, 1])
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
          <p style="color:#888;font-size:14px;margin:0;">Your personal AI assistant — text, voice & images</p>
        </div>
        <div class="login-card">
          <p style="font-size:16px;font-weight:600;color:#ececec;text-align:center;margin-bottom:4px;">Welcome back</p>
          <p style="font-size:13px;color:#777;text-align:center;margin-bottom:22px;">Sign in to continue</p>
          <div class="fi"><div class="fi-icon" style="background:#0d3b30;">⚡</div>4 AI models — Llama 3.3, Gemini, Cohere, Mistral</div>
          <div class="fi"><div class="fi-icon" style="background:#1a0d3b;">🎤</div>Voice Chat — speak and hear responses</div>
          <div class="fi"><div class="fi-icon" style="background:#0d2060;">🎨</div>Image Creator — AI-powered image generation</div>
          <div class="fi"><div class="fi-icon" style="background:#3d2200;">🌐</div>Live web search built in</div>
          <div class="fi"><div class="fi-icon" style="background:#2d1060;">🔒</div>Secure Google login</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, TOKEN_URL, TOKEN_URL)
        result = oauth2.authorize_button(
            "Continue with Google", redirect_uri=REDIRECT_URI, scope=SCOPE, key="google",
            extras_params={"prompt": "consent", "access_type": "offline"},
            use_container_width=True
        )
        st.markdown('<p style="text-align:center;font-size:11px;color:#444;margin-top:14px;">By continuing you agree to our terms</p>', unsafe_allow_html=True)
        if result and "token" in result:
            payload = jwt.decode(result["token"]["id_token"], options={"verify_signature": False})
            st.session_state.auth = True
            st.session_state.user = payload
            st.session_state.login_time = datetime.now().strftime("%d %b %Y, %I:%M %p")
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
#  AUTHENTICATED APP
# ══════════════════════════════════════════════════════════════════════════════
else:
    u       = st.session_state.user
    email   = u.get("email", "")
    name    = u.get("name", "User") or "User"
    fname   = name.split()[0]
    pic     = u.get("picture", "")
    init    = name[0].upper()
    chats   = st.session_state.chats
    if email not in chats:
        chats[email] = {}
    uchats  = chats[email]
    mdl     = st.session_state.model
    tag     = MODELS[mdl]["tag"]
    icon    = MODELS[mdl]["icon"]
    live_on = st.session_state.live

    # ── SIDEBAR ────────────────────────────────────────────────────────────────
    with st.sidebar:
        # Logo + Title
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:16px 14px 12px;
            border-bottom:1px solid #2a2a2a;margin-bottom:4px;">
          <div style="width:32px;height:32px;background:#1a3a30;border-radius:8px;
            border:1px solid #10a37f;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
            {robot_img(22)}
          </div>
          <span style="font-size:15px;font-weight:700;color:#ececec;letter-spacing:-0.3px;">Surya Dev AI</span>
        </div>
        """, unsafe_allow_html=True)

        # New Chat
        st.markdown('<div class="new-chat-wrap" style="padding:6px 10px 4px;">', unsafe_allow_html=True)
        if st.button("✏️  New Chat", key="new_chat", use_container_width=True):
            st.session_state.msgs = []
            st.session_state.chat_id = None
            st.session_state.page = "chat"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Nav buttons
        st.markdown('<div style="padding:2px 10px 4px;">', unsafe_allow_html=True)
        if st.button("🎤  Voice Chat", key="nav_voice", use_container_width=True):
            st.session_state.page = "voice"; st.rerun()
        if st.button("🎨  Image Creator", key="nav_img", use_container_width=True):
            st.session_state.page = "images"; st.rerun()
        if st.button("🔍  Search Chats", key="nav_search", use_container_width=True):
            st.session_state.page = "search"; st.rerun()
        if st.button("🤖  AI Models", key="nav_models", use_container_width=True):
            st.session_state.page = "models"; st.rerun()
        if st.button("👤  Account", key="nav_acc", use_container_width=True):
            st.session_state.page = "account"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)

        # Model selector
        st.markdown('<span class="sidebar-section-label">Active Model</span>', unsafe_allow_html=True)
        new_model = st.selectbox("Model", list(MODELS.keys()),
            index=list(MODELS.keys()).index(mdl), label_visibility="collapsed")
        if new_model != mdl:
            st.session_state.model = new_model; st.rerun()

        # Toggles
        st.markdown('<div style="padding:4px 2px 2px;">', unsafe_allow_html=True)
        new_live = st.toggle("🌐  Live Web Search", value=st.session_state.live, key="toggle_live")
        if new_live != st.session_state.live:
            st.session_state.live = new_live; st.rerun()

        new_tts = st.toggle("🔊  Read Responses Aloud", value=st.session_state.tts_enabled, key="toggle_tts")
        if new_tts != st.session_state.tts_enabled:
            st.session_state.tts_enabled = new_tts; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr>', unsafe_allow_html=True)

        # Recent Chats
        if uchats:
            st.markdown('<span class="sidebar-section-label">Recent Chats</span>', unsafe_allow_html=True)
            for cid, cdata in sorted(uchats.items(), reverse=True)[:20]:
                title = cdata.get("title", "Untitled")
                disp  = title[:26] + ("…" if len(title) > 26 else "")
                st.markdown('<div class="hist-item">', unsafe_allow_html=True)
                if st.button(f"💬 {disp}", key=f"h_{cid}", use_container_width=True):
                    st.session_state.chat_id = cid
                    st.session_state.msgs = cdata["messages"]
                    st.session_state.page = "chat"; st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div style="height:80px;"></div>', unsafe_allow_html=True)

        # Bottom user card
        av = (f'<img src="{pic}" style="width:30px;height:30px;border-radius:50%;object-fit:cover;border:2px solid #10a37f;flex-shrink:0;"/>'
              if pic else
              f'<div style="width:30px;height:30px;border-radius:50%;background:#10a37f;'
              f'display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:13px;flex-shrink:0;">{init}</div>')

        st.markdown(f"""
        <div class="user-bottom">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">
            {av}
            <div>
              <div style="font-size:13px;font-weight:600;color:#ececec;">{name}</div>
              <div style="font-size:11px;color:#555;">Free Plan</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="signout-wrap" style="padding:0 10px 72px;">', unsafe_allow_html=True)
        if st.button("🚪  Sign Out", key="signout", use_container_width=True):
            for k in DEFAULTS:
                st.session_state[k] = DEFAULTS[k]
            st.session_state.chats = load_chats(); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  VOICE CHAT PAGE
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.page == "voice":
        _, cc, _ = st.columns([1, 3, 1])
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button("← Back to Chat", key="back_voice"):
                st.session_state.page = "chat"; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align:center;padding:30px 0 10px;">
              <h2 style="color:#ececec;font-size:1.6rem;font-weight:600;margin-bottom:6px;">Voice Chat</h2>
              <p style="color:#888;font-size:14px;">Speak your question — get an AI response read aloud</p>
            </div>
            """, unsafe_allow_html=True)

            # Voice orb + recognition UI
            st.markdown("""
            <div class="voice-mode-panel">
              <div class="voice-orb" id="voice-btn" onclick="startVoiceRecognition()">🎤</div>
              <p style="color:#ececec;font-size:15px;font-weight:500;margin:0 0 4px;">Tap to speak</p>
              <p id="voice-status" class="voice-status" style="color:#888;font-size:13px;min-height:20px;"></p>
              <p style="color:#555;font-size:12px;margin-top:12px;">Works best in Chrome • Speak clearly</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p style="color:#888;font-size:13px;text-align:center;">Or type your question below and it will be read back to you:</p>', unsafe_allow_html=True)

            # Voice chat uses same chat input but auto-speaks response
            for msg in st.session_state.msgs[-6:]:  # show last few messages
                with st.chat_message(msg["role"]):
                    if msg["role"] == "assistant":
                        badge = f'<span class="mbadge {tag}">{icon} {mdl}</span>'
                        badge += f' <span class="mbadge voice">🎤 Voice</span>'
                        st.markdown(badge, unsafe_allow_html=True)
                    st.markdown(msg["content"])

            voice_prompt = st.chat_input("Type or use voice above…", key="voice_input")
            if voice_prompt:
                st.session_state.msgs.append({"role": "user", "content": voice_prompt})
                with st.spinner("💭 Thinking…"):
                    reply, searched = get_response(st.session_state.msgs, mdl, live_on)
                st.session_state.msgs.append({"role": "assistant", "content": reply, "searched": searched, "voice": True})
                cid = st.session_state.chat_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
                st.session_state.chat_id = cid
                chats[email][cid] = {
                    "title": f"[Voice] {st.session_state.msgs[0]['content'][:35]}",
                    "messages": st.session_state.msgs
                }
                st.session_state.chats = chats
                save_chats(chats)
                # Auto-speak the response
                safe_reply = reply.replace("`", "").replace('"', "'")[:500]
                st.markdown(f"<script>speakText(`{safe_reply}`);</script>", unsafe_allow_html=True)
                st.rerun()

            # TTS controls
            if st.session_state.msgs:
                last_assistant = next((m for m in reversed(st.session_state.msgs) if m["role"] == "assistant"), None)
                if last_assistant:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔊 Play Last Response", use_container_width=True, key="play_tts"):
                            safe_text = last_assistant["content"].replace("`", "").replace('"', "'")[:500]
                            st.markdown(f"<script>speakText(`{safe_text}`);</script>", unsafe_allow_html=True)
                    with col2:
                        if st.button("⏹️ Stop Speaking", use_container_width=True, key="stop_tts"):
                            st.markdown("<script>window.speechSynthesis.cancel();</script>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  IMAGE CREATOR PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.page == "images":
        _, cc, _ = st.columns([1, 3, 1])
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button("← Back to Chat", key="back_img"):
                st.session_state.page = "chat"; st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="text-align:center;padding:20px 0 16px;">
              <h2 style="color:#ececec;font-size:1.6rem;font-weight:600;margin-bottom:6px;">🎨 Image Creator</h2>
              <p style="color:#888;font-size:14px;">Describe any image and AI will create it for you</p>
            </div>
            """, unsafe_allow_html=True)

            # Style selection
            styles = ["Realistic", "Digital Art", "Oil Painting", "Anime", "Watercolor",
                      "3D Render", "Sketch", "Cyberpunk", "Fantasy", "Minimalist", "Photographic", "Abstract"]
            st.markdown('<p style="color:#aaa;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;margin:0 0 8px;">Style</p>', unsafe_allow_html=True)
            selected_style = st.selectbox("Style", styles,
                index=styles.index(st.session_state.img_style) if st.session_state.img_style in styles else 0,
                label_visibility="collapsed")
            st.session_state.img_style = selected_style

            # Quality / size
            col_a, col_b = st.columns(2)
            with col_a:
                sizes = ["512×512", "768×768", "1024×1024", "1024×768"]
                img_size = st.selectbox("Size", sizes,
                    index=sizes.index(st.session_state.img_size) if st.session_state.img_size in sizes else 2,
                    label_visibility="visible")
                st.session_state.img_size = img_size
            with col_b:
                quality = st.selectbox("Quality", ["Standard", "HD", "Ultra"],
                    label_visibility="visible")

            # Prompt input
            img_prompt = st.text_area(
                "Describe your image",
                placeholder="E.g.: A futuristic city at sunset, neon lights reflecting on wet streets, cinematic, ultra-detailed…",
                height=100, key="img_prompt_input"
            )

            # Negative prompt
            with st.expander("⚙️ Advanced Options"):
                neg_prompt = st.text_area("Negative Prompt (what to avoid)",
                    placeholder="blurry, low quality, watermark, distorted…", height=60)
                seed = st.number_input("Seed (0 = random)", min_value=0, max_value=999999, value=0)

            col1, col2 = st.columns([3, 1])
            with col1:
                generate_btn = st.button("✨ Generate Image", use_container_width=True, key="gen_img_btn",
                    type="primary")
            with col2:
                surprise_btn = st.button("🎲 Surprise Me", use_container_width=True, key="surprise_btn")

            # Surprise prompts
            SURPRISE_PROMPTS = [
                "A dragon made of crystal flying over an enchanted forest, magical lighting, fantasy art",
                "Astronaut relaxing on the moon with a coffee mug, Earth visible in background, cinematic",
                "Ancient Japanese temple surrounded by cherry blossoms in fog, golden hour, serene",
                "Cyberpunk robot samurai in neon-lit Tokyo alley, rain, ultra detailed",
                "Underwater city with glowing bioluminescent buildings, deep ocean, surreal",
                "Portrait of an elderly wizard with stars in his eyes, photorealistic, detailed",
                "A tiny cozy cabin in a giant snow globe, magical winter scene",
                "Futuristic farm on Mars with transparent biodomes, sci-fi concept art",
            ]

            if surprise_btn:
                import random
                img_prompt = random.choice(SURPRISE_PROMPTS)
                st.session_state["surprise_prompt"] = img_prompt
                generate_btn = True

            # Use surprise prompt if set
            if "surprise_prompt" in st.session_state and not img_prompt:
                img_prompt = st.session_state.pop("surprise_prompt")

            if generate_btn and img_prompt:
                # Build full prompt with style
                style_suffix = f", {selected_style.lower()} style" if selected_style != "Realistic" else ""
                quality_suffix = ", 8k ultra detailed" if quality == "Ultra" else (", high quality" if quality == "HD" else "")
                full_prompt = f"{img_prompt}{style_suffix}{quality_suffix}"
                if neg_prompt:
                    full_prompt += f", avoid: {neg_prompt}"

                with st.spinner("🎨 Generating your image…"):
                    try:
                        # Parse size
                        w, h = (img_size.split("×") + ["1024", "1024"])[:2]
                        encoded = quote(full_prompt)
                        seed_param = f"&seed={seed}" if seed > 0 else ""
                        image_url = f"https://image.pollinations.ai/prompt/{encoded}?width={w}&height={h}&nologo=true&enhance=true{seed_param}"

                        st.markdown("---")
                        st.markdown(f'<span class="mbadge img">🎨 AI Generated • {selected_style}</span>', unsafe_allow_html=True)

                        # Display image
                        img_col, _ = st.columns([2, 1])
                        with img_col:
                            st.image(image_url, caption=img_prompt[:80] + ("…" if len(img_prompt) > 80 else ""), use_column_width=True)

                        # Action buttons
                        dl_col, share_col, _ = st.columns([1, 1, 2])
                        with dl_col:
                            st.markdown(f'<a href="{image_url}" download="surya-ai-image.jpg" target="_blank"><button style="background:#2a2a2a;border:1px solid #383838;color:#d0d0d0;border-radius:8px;padding:8px 16px;cursor:pointer;font-size:13px;width:100%;">⬇️ Download</button></a>', unsafe_allow_html=True)
                        with share_col:
                            st.markdown(f'<a href="{image_url}" target="_blank"><button style="background:#2a2a2a;border:1px solid #383838;color:#d0d0d0;border-radius:8px;padding:8px 16px;cursor:pointer;font-size:13px;width:100%;">🔗 Open Full</button></a>', unsafe_allow_html=True)

                        # Save to history
                        st.session_state.generated_images.append({
                            "prompt": img_prompt, "url": image_url,
                            "style": selected_style, "time": datetime.now().strftime("%H:%M")
                        })
                    except Exception as e:
                        st.error(f"Image generation failed: {str(e)}")

            elif generate_btn and not img_prompt:
                st.warning("Please describe the image you want to create.")

            # Gallery of recent generations
            if st.session_state.generated_images:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p style="color:#888;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;">Recent Generations</p>', unsafe_allow_html=True)
                recent = list(reversed(st.session_state.generated_images[-6:]))
                cols = st.columns(3)
                for i, img_data in enumerate(recent):
                    with cols[i % 3]:
                        st.image(img_data["url"], caption=f"{img_data['style']} • {img_data['time']}", use_column_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  ACCOUNT PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.page == "account":
        _, cc, _ = st.columns([1, 3, 1])
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button("← Back to Chat", key="back_acc"):
                st.session_state.page = "chat"; st.rerun()
            st.markdown('</div><br>', unsafe_allow_html=True)

            av_lg = (f'<img src="{pic}" style="width:70px;height:70px;border-radius:50%;border:3px solid #10a37f;object-fit:cover;"/>'
                     if pic else
                     f'<div style="width:70px;height:70px;border-radius:50%;background:#10a37f;display:flex;'
                     f'align-items:center;justify-content:center;font-size:28px;color:white;font-weight:700;flex-shrink:0;">{init}</div>')

            tc = len(uchats)
            tm = total_msgs(uchats)
            ti = len(st.session_state.generated_images)
            lt = st.session_state.login_time or "—"

            st.markdown(f"""
            <div class="acc-card">{av_lg}
              <div><div class="acc-name">{name}</div>
              <div class="acc-email">{email}</div>
              <div class="acc-badge">🤖 Surya Dev AI Member</div></div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            for col, num, lbl in [(c1, tc, "Chats"), (c2, tm, "Messages"), (c3, ti, "Images"), (c4, 4, "AI Models")]:
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
              <div class="irow"><span class="ik">Read Aloud</span><span class="iv">{"✅ Enabled" if st.session_state.tts_enabled else "❌ Disabled"}</span></div>
              <div class="irow"><span class="ik">Theme</span><span class="iv">🌙 Dark</span></div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            b1, b2 = st.columns(2)
            with b1:
                if st.button("🗑️ Clear All Chats", use_container_width=True, key="clr_all"):
                    chats[email] = {}
                    st.session_state.chats = chats
                    st.session_state.msgs = []
                    st.session_state.chat_id = None
                    save_chats(chats)
                    st.success("All chats cleared!")
                    st.rerun()
            with b2:
                if st.button("🚪 Sign Out", use_container_width=True, key="so2"):
                    for k in DEFAULTS:
                        st.session_state[k] = DEFAULTS[k]
                    st.session_state.chats = load_chats()
                    st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  MODELS PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.page == "models":
        _, cc, _ = st.columns([1, 3, 1])
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button("← Back to Chat", key="back_mdl"):
                st.session_state.page = "chat"; st.rerun()
            st.markdown('</div><br>', unsafe_allow_html=True)
            st.markdown('<h2 style="color:#ececec;font-size:1.5rem;margin-bottom:6px;">AI Models</h2>', unsafe_allow_html=True)
            st.markdown('<p style="color:#888;font-size:14px;margin-bottom:24px;">Switch between powerful AI models</p>', unsafe_allow_html=True)

            MODEL_DETAILS = {
                "GPT-4o (Llama 3.3)":  {"icon": "⚡", "desc": "Fastest & most capable. Llama 3.3 70B via Groq LPU — blazing speed for any task.", "speed": "🚀 Very Fast", "best": "General, Coding, Math"},
                "Gemini 1.5 Flash":     {"icon": "✦", "desc": "Google's multimodal AI. Excellent reasoning, long context, image understanding.", "speed": "⚡ Fast", "best": "Analysis, Research"},
                "Command R (Cohere)":   {"icon": "◈", "desc": "Cohere Command R — retrieval-augmented generation champion.", "speed": "🏃 Moderate", "best": "Documents, RAG"},
                "Mistral Small":        {"icon": "◆", "desc": "Mistral's efficient model — great balance of speed and quality.", "speed": "⚡ Fast", "best": "Writing, Summarization"},
            }
            for mname, minfo in MODEL_DETAILS.items():
                is_active = mname == st.session_state.model
                border = "#10a37f" if is_active else "#333"
                bg     = "#0d2d22" if is_active else "#2a2a2a"
                st.markdown(f"""
                <div style="background:{bg};border:2px solid {border};border-radius:14px;
                  padding:18px 20px;margin-bottom:12px;">
                  <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    <span style="font-size:22px;">{minfo['icon']}</span>
                    <div style="flex:1;">
                      <div style="font-size:15px;font-weight:600;color:#ececec;">{mname}</div>
                      <div style="font-size:12px;color:#888;">{minfo['speed']} • Best for: {minfo['best']}</div>
                    </div>
                    {"<span style='background:#10a37f;color:white;font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;'>✓ Active</span>" if is_active else ""}
                  </div>
                  <p style="font-size:13px;color:#aaa;margin:0;">{minfo['desc']}</p>
                </div>""", unsafe_allow_html=True)
                if not is_active:
                    if st.button(f"Switch to {mname}", key=f"sel_{mname}", use_container_width=True):
                        st.session_state.model = mname
                        st.session_state.page = "chat"; st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  SEARCH PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.page == "search":
        _, cc, _ = st.columns([1, 3, 1])
        with cc:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button("← Back to Chat", key="back_srch"):
                st.session_state.page = "chat"; st.rerun()
            st.markdown('</div><br>', unsafe_allow_html=True)
            st.markdown('<h2 style="color:#ececec;font-size:1.5rem;margin-bottom:16px;">Search Chats</h2>', unsafe_allow_html=True)
            query = st.text_input("", placeholder="Search your conversations…", label_visibility="collapsed")
            if query:
                results = [
                    (cid, cd) for cid, cd in uchats.items()
                    if query.lower() in cd.get("title", "").lower()
                    or any(query.lower() in m.get("content", "").lower() for m in cd.get("messages", []))
                ]
                if results:
                    st.markdown(f'<p style="color:#888;font-size:13px;margin-bottom:12px;">{len(results)} result(s)</p>', unsafe_allow_html=True)
                    for cid, cdata in sorted(results, reverse=True):
                        title = cdata.get("title", "Untitled")
                        if st.button(f"💬 {title[:50]}", key=f"sr_{cid}", use_container_width=True):
                            st.session_state.chat_id = cid
                            st.session_state.msgs = cdata["messages"]
                            st.session_state.page = "chat"; st.rerun()
                else:
                    st.markdown('<p style="color:#666;font-size:14px;">No chats found.</p>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  CHAT PAGE
    # ══════════════════════════════════════════════════════════════════════════
    elif st.session_state.page == "chat":
        _, cc, _ = st.columns([1, 5, 1])
        with cc:
            if not st.session_state.msgs:
                # Welcome screen
                st.markdown("<br><br><br>", unsafe_allow_html=True)
                st.markdown(f"""
                <div style="text-align:center;margin-bottom:36px;">
                  <h1 style="font-size:2.2rem;font-weight:600;color:#ececec;
                    margin:0 0 8px;letter-spacing:-0.5px;">
                    Good day, {fname}
                  </h1>
                  <p style="color:#888;font-size:15px;margin:0;">What can I help you with today?</p>
                </div>""", unsafe_allow_html=True)

                suggestions = [
                    ("What's happening in AI news today?",   "🌐"),
                    ("Explain quantum computing simply",       "⚛️"),
                    ("Write me a Python web scraper",          "🐍"),
                    ("Creative startup idea for 2025",         "💡"),
                    ("What is the population of India?",       "🗺️"),
                    ("Help me write a professional email",     "✉️"),
                    ("🎨 Generate an image of a sunset",       "🎨"),
                    ("🎤 Help me practice a speech",           "🎤"),
                ]
                c1, c2 = st.columns(2)
                for i, (txt, ico) in enumerate(suggestions):
                    col = c1 if i % 2 == 0 else c2
                    with col:
                        st.markdown('<div class="chip">', unsafe_allow_html=True)
                        if st.button(f"{ico}  {txt}", use_container_width=True, key=f"s{i}"):
                            # Route image/voice suggestions
                            if "Generate an image" in txt or txt.startswith("🎨"):
                                st.session_state.page = "images"; st.rerun()
                            elif "speech" in txt or txt.startswith("🎤"):
                                st.session_state.page = "voice"; st.rerun()
                            else:
                                st.session_state.msgs.append({"role": "user", "content": txt})
                                with st.spinner("Thinking…"):
                                    reply, searched = get_response(st.session_state.msgs, mdl, live_on)
                                st.session_state.msgs.append({"role": "assistant", "content": reply, "searched": searched})
                                cid = datetime.now().strftime("%Y%m%d%H%M%S%f")
                                st.session_state.chat_id = cid
                                chats[email][cid] = {"title": txt[:40], "messages": st.session_state.msgs}
                                st.session_state.chats = chats
                                save_chats(chats)
                                st.rerun()
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown("<br>", unsafe_allow_html=True)
                for msg in st.session_state.msgs:
                    with st.chat_message(msg["role"]):
                        if msg["role"] == "assistant":
                            badge = f'<span class="mbadge {tag}">{icon} {mdl}</span>'
                            if msg.get("searched"):
                                badge += f' <span class="mbadge live">🌐 Live</span>'
                            if msg.get("image_url"):
                                badge += f' <span class="mbadge img">🎨 Image</span>'
                            if msg.get("voice"):
                                badge += f' <span class="mbadge voice">🎤 Voice</span>'
                            st.markdown(badge, unsafe_allow_html=True)
                            # Show generated image if any
                            if msg.get("image_url"):
                                st.image(msg["image_url"], use_column_width=True)
                        st.markdown(msg["content"])

                        # TTS button on assistant messages
                        if msg["role"] == "assistant" and st.session_state.tts_enabled:
                            safe_text = msg["content"].replace("`", "").replace('"', "'")[:500]
                            st.markdown(
                                f'<button onclick="speakText(`{safe_text}`)" style="'
                                f'background:transparent;border:1px solid #333;color:#888;'
                                f'border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;'
                                f'margin-top:6px;" title="Read aloud">🔊 Read</button>',
                                unsafe_allow_html=True
                            )

                st.markdown("<br><br><br>", unsafe_allow_html=True)

        # Main chat input (outside columns so it sticks to bottom)
        prompt = st.chat_input("Message Surya Dev AI…")
        if prompt:
            st.session_state.msgs.append({"role": "user", "content": prompt})

            # Check if user wants image generation
            if is_image_request(prompt):
                # Extract the image subject
                img_subject = re.sub(
                    r'^(generate|create|draw|make|paint|show me|produce|design)\s+(an?\s+)?(image|picture|photo|illustration|artwork|art|painting|sketch)\s*(of|showing|depicting|about)?\s*',
                    '', prompt, flags=re.IGNORECASE
                ).strip() or prompt

                with st.spinner("🎨 Creating your image…"):
                    style_suffix = f", {st.session_state.img_style} style, high quality"
                    full_prompt = img_subject + style_suffix
                    image_url, err = generate_image(full_prompt)

                if err:
                    reply = f"⚠️ Couldn't generate the image: {err}\n\nYou can also try the **Image Creator** page from the sidebar for more options."
                    st.session_state.msgs.append({"role": "assistant", "content": reply})
                else:
                    reply = f"Here's your image of **{img_subject}**!\n\nGenerated in {st.session_state.img_style} style. Visit the 🎨 **Image Creator** page for more styles and options."
                    st.session_state.msgs.append({
                        "role": "assistant", "content": reply,
                        "image_url": image_url, "searched": False
                    })
            else:
                is_live = needs_search(prompt) and live_on
                with st.spinner("🔍 Searching the web…" if is_live else "💭 Thinking…"):
                    reply, searched = get_response(st.session_state.msgs, mdl, live_on)
                st.session_state.msgs.append({"role": "assistant", "content": reply, "searched": searched})

                # Auto-TTS if enabled
                if st.session_state.tts_enabled:
                    safe_reply = reply.replace("`", "").replace('"', "'")[:500]
                    st.markdown(f"<script>setTimeout(()=>speakText(`{safe_reply}`),300);</script>", unsafe_allow_html=True)

            cid = st.session_state.chat_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
            st.session_state.chat_id = cid
            chats[email][cid] = {
                "title": st.session_state.msgs[0]["content"][:40],
                "messages": [m for m in st.session_state.msgs if "image_url" not in m or True]
            }
            # Serialize safely (image_url is a string, fine to save)
            serializable = []
            for m in st.session_state.msgs:
                serializable.append({k: v for k, v in m.items()})
            chats[email][cid]["messages"] = serializable
            st.session_state.chats = chats
            save_chats(chats)
            st.rerun()
