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

# ── API Keys & OAuth Config ──────────────────────────────────────────────────
GROQ_API_KEY    = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY  = st.secrets["GEMINI_API_KEY"]
COHERE_API_KEY  = st.secrets["COHERE_API_KEY"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
CLIENT_ID       = st.secrets["CLIENT_ID"]
CLIENT_SECRET   = st.secrets["CLIENT_SECRET"]

AUTHORIZE_URL = "[accounts.google.com](https://accounts.google.com/o/oauth2/auth)"
TOKEN_URL     = "[oauth2.googleapis.com](https://oauth2.googleapis.com/token)"
REDIRECT_URI  = "[nz5ossng47a243vac5nrti.streamlit.app](https://nz5ossng47a243vac5nrti.streamlit.app)"
SCOPE         = "openid email profile"
CHATS_FILE    = "chats.json"

groq_client   = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
cohere_client = cohere.ClientV2(api_key=COHERE_API_KEY)

MODELS = ["Llama 3.3 (Groq)", "Gemini 1.5 Flash", "Command R+ (Cohere)", "Mistral Small"]

# ── Robot Logo SVG ───────────────────────────────────────────────────────────
ROBOT_LOGO_SVG = """
<svg xmlns="[w3.org](http://www.w3.org/2000/svg)" viewBox="0 0 64 64" width="{size}" height="{size}">
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

# ── Live Search Functions ────────────────────────────────────────────────────
def search_wikipedia(query):
    try:
        url = f"[en.wikipedia.org](https://en.wikipedia.org/api/rest_v1/page/summary/{quote(query)})"
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
        url = f"[api.duckduckgo.com](https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1)"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results = []
            if data.get("Answer"):
                results.append(f"Quick Answer: {data['Answer']}")
            if data.get("AbstractText"):
                results.append(f"Summary: {data['AbstractText'][:800]}")
            infobox = data.get("Infobox", {})
            if infobox and infobox.get("content"):
                for fact in infobox["content"][:3]:
                    label, value = fact.get("label", ""), fact.get("value", "")
                    if label and value:
                        results.append(f"{label}: {value}")
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
    results = []
    seen = set()
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

# ── AI Response (with toggle) ────────────────────────────────────────────────
def _build_messages(messages, use_search):
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    if not (use_search and needs_live_search(last_user)):
        return list(messages), False

    live = get_live_context(last_user)
    if not live:
        return list(messages), False

    ctx = "\n\n".join(f"[{r['source']}] {r['title']}:\n{r['content']}" for r in live)
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
    return [m for m in messages[:-1]] + [inject], True

def get_ai_response(messages, model_choice, use_search=True):
    try:
        msgs, _ = _build_messages(messages, use_search)

        if model_choice == "Llama 3.3 (Groq)":
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile", messages=msgs)
            return resp.choices[0].message.content

        elif model_choice == "Gemini 1.5 Flash":
            model = genai.GenerativeModel("gemini-1.5-flash")
            history = []
            for m in msgs[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})
            # Gemini history must start with a user turn
            while history and history[0]["role"] != "user":
                history.pop(0)
            chat = model.start_chat(history=history)
            return chat.send_message(msgs[-1]["content"]).text

        elif model_choice == "Command R+ (Cohere)":
            cohere_msgs = [
                {"role": "user" if m["role"] == "user" else "assistant",
                 "content": m["content"]}
                for m in msgs
            ]
            resp = cohere_client.chat(model="command-r-plus", messages=cohere_msgs)
            return resp.message.content[0].text

        elif model_choice == "Mistral Small":
            headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}",
                       "Content-Type": "application/json"}
            data = {"model": "mistral-small-latest", "messages": msgs}
            resp = requests.post("[api.mistral.ai](https://api.mistral.ai/v1/chat/completions)",
                                 headers=headers, json=data, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        return "⚠️ Unknown model selected."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

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
        pass  # ephemeral FS on Streamlit Cloud — best effort

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
    all_chats[user_email][chat_id] = {
        "title": title[:40],
        "messages": st.session_state.messages,
    }
    st.session_state.all_chats = all_chats
    save_chats(all_chats)

def do_signout():
    st.session_state.authenticated = False
    st.session_state.user_info = None
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    st.session_state.show_account = False

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Surya Dev AI", page_icon="🤖", layout="wide")

 
 
