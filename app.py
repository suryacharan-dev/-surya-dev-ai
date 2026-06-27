import streamlit as st
from groq import Groq
import google.generativeai as genai
import cohere
import requests
from streamlit_oauth import OAuth2Component
import jwt
import json
import os
from datetime import datetime

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REDIRECT_URI = "https://nz5ossng47a243vac5nrti.streamlit.app"
SCOPE = "openid email profile"
CHATS_FILE = "chats.json"

groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
cohere_client = cohere.Client(api_key=COHERE_API_KEY)

def load_chats():
    if os.path.exists(CHATS_FILE):
        with open(CHATS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_chats(chats):
    with open(CHATS_FILE, "w") as f:
        json.dump(chats, f)

def get_ai_response(messages, model_choice):
    if model_choice == "⚡ Groq (Llama 3.3)":
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return response.choices[0].message.content
    elif model_choice == "🔵 Google Gemini":
        model = genai.GenerativeModel("gemini-1.5-flash")
        history = []
        for m in messages[:-1]:
            role = "user" if m["role"] == "user" else "model"
            history.append({"role": role, "parts": [m["content"]]})
        chat = model.start_chat(history=history)
        response = chat.send_message(messages[-1]["content"])
        return response.text
    elif model_choice == "🟠 Cohere":
        chat_history = []
        for m in messages[:-1]:
            role = "USER" if m["role"] == "user" else "CHATBOT"
            chat_history.append({"role": role, "message": m["content"]})
        response = cohere_client.chat(
            message=messages[-1]["content"],
            chat_history=chat_history,
            model="command-r"
        )
        return response.text
    elif model_choice == "⚫ Mistral":
        headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
        data = {"model": "mistral-small-latest", "messages": messages}
        response = requests.post("https://api.mistral.ai/v1/chat/completions", headers=headers, json=data)
        return response.json()["choices"][0]["message"]["content"]

st.set_page_config(page_title="Surya Dev AI", page_icon="🚀", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Söhne:wght@300;400;500;600&family=Inter:wght@300;400;500;600&display=swap');
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    * { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #1a1a1a; color: #e8e8e6; }
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #2a2a2a; }
    .stChatInput textarea { background-color: #2a2a2a !important; color: #e8e8e6 !important; border-radius: 20px !important; border: 1px solid #3a3a3a !important; font-size: 15px !important; padding: 16px !important; }
    .stChatInput textarea:focus { border: 1px solid #c084fc !important; box-shadow: 0 0 0 2px rgba(192,132,252,0.1) !important; }
    .stButton button { background-color: transparent; color: #e8e8e6; border-radius: 10px; border: none; width: 100%; text-align: left; padding: 10px 14px; font-size: 13px; transition: all 0.2s; }
    .stButton button:hover { background-color: #2a2a2a !important; color: white !important; }
    .stSelectbox > div > div { background-color: #2a2a2a !important; border: 1px solid #3a3a3a !important; border-radius: 12px !important; color: white !important; }
    .profile-card { background: linear-gradient(135deg, #2a2a2a, #222); border-radius: 14px; padding: 14px; display: flex; align-items: center; gap: 12px; margin-bottom: 8px; border: 1px solid #333; }
    .profile-pic { width: 40px; height: 40px; border-radius: 50%; border: 2px solid #c084fc; }
    .profile-name { font-weight: 600; font-size: 14px; color: #e8e8e6; }
    .profile-email { font-size: 11px; color: #888; }
    .model-badge { display: inline-flex; align-items: center; gap: 4px; background: linear-gradient(135deg, #581c87, #3b0764); border-radius: 8px; padding: 3px 10px; font-size: 11px; color: #c084fc; margin-bottom: 8px; border: 1px solid #7c3aed33; }
    .suggestion-btn { background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 14px; padding: 16px; text-align: left; color: #e8e8e6; font-size: 14px; cursor: pointer; transition: all 0.2s; width: 100%; }
    .suggestion-btn:hover { border-color: #c084fc; background: #2f2f2f; }
    .chat-user { background: linear-gradient(135deg, #2a2a2a, #252525); border-radius: 18px 18px 4px 18px; padding: 14px 18px; margin: 4px 0; }
    .chat-ai { background: transparent; border-radius: 18px 18px 18px 4px; padding: 14px 0; margin: 4px 0; }
    .sidebar-title { font-size: 16px; font-weight: 700; color: #c084fc; letter-spacing: -0.3px; }
    .new-chat-btn button { background: linear-gradient(135deg, #7c3aed, #6d28d9) !important; color: white !important; border-radius: 12px !important; font-weight: 500 !important; padding: 10px !important; }
    .new-chat-btn button:hover { background: linear-gradient(135deg, #8b5cf6, #7c3aed) !important; }
    h1, h2, h3 { color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_info" not in st.session_state:
    st.session_state.user_info = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None
if "all_chats" not in st.session_state:
    st.session_state.all_chats = load_chats()
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "⚡ Groq (Llama 3.3)"

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align:center; margin-bottom:20px;'>
                <div style='width:64px; height:64px; background:linear-gradient(135deg,#7c3aed,#c084fc); border-radius:16px; display:inline-flex; align-items:center; justify-content:center; font-size:32px; margin-bottom:16px;'>🚀</div>
                <h1 style='font-size:2.2rem; margin:0; background:linear-gradient(135deg,#c084fc,#e879f9); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>Surya Dev AI</h1>
                <p style='color:#888; margin-top:8px;'>Your intelligent AI assistant</p>
            </div>
        """, unsafe_allow_html=True)
        st.divider()
        oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, TOKEN_URL, TOKEN_URL)
        result = oauth2.authorize_button("🔐 Continue with Google", redirect_uri=REDIRECT_URI, scope=SCOPE, key="google", extras_params={"prompt": "consent", "access_type": "offline"}, use_container_width=True)
        if result and "token" in result:
            payload = jwt.decode(result["token"]["id_token"], options={"verify_signature": False})
            st.session_state.authenticated = True
            st.session_state.user_info = payload
            st.rerun()

else:
    user = st.session_state.user_info
    user_email = user.get("email", "default")
    user_name = user.get("name", "User")
    user_pic = user.get("picture", "")

    all_chats = st.session_state.all_chats
    if user_email not in all_chats:
        all_chats[user_email] = {}
    user_chats = all_chats[user_email]

    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-title'>🚀 Surya Dev AI</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='new-chat-btn'>", unsafe_allow_html=True)
        if st.button("✏️ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.session_state.model_choice = st.selectbox("🤖 AI Model", ["⚡ Groq (Llama 3.3)", "🔵 Google Gemini", "🟠 Cohere", "⚫ Mistral"])
        st.divider()
        st.markdown("#### 💬 Chats")
        for chat_id, chat_data in sorted(user_chats.items(), reverse=True):
            title = chat_data.get("title", "New Chat")
            if st.button(f"🗨️ {title[:25]}", key=chat_id):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = chat_data["messages"]
                st.rerun()
        st.divider()
        st.markdown(f"""
            <div class="profile-card">
                <img src="{user_pic}" class="profile-pic"/>
                <div>
                    <div class="profile-name">{user_name}</div>
                    <div class="profile-email">{user_email}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Sign out", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.session_state.messages = []
            st.session_state.current_chat_id = None
            st.rerun()

    if len(st.session_state.messages) == 0:
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
                <div style='text-align:center; margin-bottom:32px;'>
                    <h1 style='font-size:2rem; background:linear-gradient(135deg,#c084fc,#e879f9); -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                        Good morning, {user_name.split()[0]}! ☀️
                    </h1>
                    <p style='color:#666; font-size:15px;'>What can I help you with today?</p>
                </div>
            """, unsafe_allow_html=True)
            suggestions = [
                ("✍️", "Write me a poem about the stars"),
                ("💡", "Give me a startup business idea"),
                ("🧠", "Explain machine learning simply"),
                ("🎯", "Help me plan my day effectively")
            ]
            c1, c2 = st.columns(2)
            for i, (icon, suggestion) in enumerate(suggestions):
                col = c1 if i % 2 == 0 else c2
                if col.button(f"{icon} {suggestion}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": suggestion})
                    reply = get_ai_response(st.session_state.messages, st.session_state.model_choice)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
                    st.session_state.current_chat_id = chat_id
                    all_chats[user_email][chat_id] = {"title": suggestion[:40], "messages": st.session_state.messages}
                    st.session_state.all_chats = all_chats
                    save_chats(all_chats)
                    st.rerun()
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant":
                    st.markdown(f"<div class='model-badge'>{st.session_state.model_choice}</div>", unsafe_allow_html=True)
                st.markdown(message["content"])

    if prompt := st.chat_input("Message Surya Dev AI..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner(""):
            reply = get_ai_response(st.session_state.messages, st.session_state.model_choice)
        with st.chat_message("assistant"):
            st.markdown(f"<div class='model-badge'>{st.session_state.model_choice}</div>", unsafe_allow_html=True)
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        chat_id = st.session_state.current_chat_id or datetime.now().strftime("%Y%m%d%H%M%S")
        st.session_state.current_chat_id = chat_id
        all_chats[user_email][chat_id] = {"title": prompt[:40], "messages": st.session_state.messages}
        st.session_state.all_chats = all_chats
        save_chats(all_chats)
        st.rerun()
