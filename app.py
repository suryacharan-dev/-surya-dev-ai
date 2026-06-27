import streamlit as st
import google.generativeai as genai
from streamlit_oauth import OAuth2Component
import jwt
import json
import os
import time
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION & INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Gemini Clone", page_icon="✨", layout="wide")

# Fetch keys securely from Streamlit Secrets
GEMINI_API_KEY  = st.secrets.get("GEMINI_API_KEY", "AQ.Ab8RN6KH1N0kCQfJBeUndy8RprjXQllBlAXalOfZGqnAucX6fg")
CLIENT_ID       = st.secrets.get("CLIENT_ID", "464095238623-4sevhtos09v5vehbj5jmdm4f5tkfg5un.apps.googleusercontent.com")
CLIENT_SECRET   = st.secrets.get("CLIENT_SECRET", "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")

AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URL     = "https://oauth2.googleapis.com/token"
REDIRECT_URI  = st.secrets.get("REDIRECT_URI", "http://localhost:8501") # Update for deployment
SCOPE         = "openid email profile"
CHATS_FILE    = "chats_db.json"

# Configure Gemini globally
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize OAuth Component
oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_URL, TOKEN_URL, TOKEN_URL, REDIRECT_URI)

# ════════════════════════════════════════════════════════════════════════════
# 2. SESSION STATE MANAGEMENT
# ════════════════════════════════════════════════════════════════════════════
def load_all_chats():
    if os.path.exists(CHATS_FILE):
        try:
            with open(CHATS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_all_chats(chats):
    try:
        with open(CHATS_FILE, "w") as f:
            json.dump(chats, f, indent=4)
    except Exception as e:
        st.error(f"Failed to save chat database: {e}")

_defaults = {
    "authenticated": False,
    "user_info": None,
    "messages": [],
    "current_chat_id": None,
    "db_chats": load_all_chats()
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════════════════════════════
# 3. ADVANCED GEMINI GRAPHICS & SKINNING (CSS)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #131314 !important;
        color: #e3e3e3;
        font-family: 'Outfit', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1e1f20 !important;
        border-right: none;
    }
    
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .gemini-greeting {
        background: -webkit-linear-gradient(74deg, #4285f4 0, #9b72cb 9%, #d96570 20%, #d96570 24%, #9b72cb 35%, #4285f4 44%, #9b72cb 50%, #d96570 56%, #131314 75%, #131314 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 56px;
        font-weight: 500;
        letter-spacing: -1px;
        line-height: 1.2;
        margin-top: 8vh;
    }
    
    .gemini-subtitle {
        color: #444746;
        font-size: 56px;
        font-weight: 500;
        letter-spacing: -1px;
        line-height: 1.2;
        margin-bottom: 40px;
    }

    .suggestion-card {
        background-color: #1e1f20;
        border-radius: 12px;
        padding: 16px;
        height: 160px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        border: 1px solid transparent;
    }
    
    .suggestion-text {
        font-size: 14.5px;
        color: #e3e3e3;
    }
    
    .suggestion-icon {
        align-self: flex-end;
        background-color: #131314;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
    }

    [data-testid="stChatInput"] {
        background-color: #1e1f20 !important;
        border-radius: 24px !important;
        border: none !important;
    }
    
    [data-testid="stChatInput"] textarea {
        color: #e3e3e3 !important;
    }
    
    .sidebar-btn button {
        background-color: #1a1a1a !important;
        border: none !important;
        border-radius: 24px !important;
        color: #c4c7c5 !important;
        font-weight: 500 !important;
        display: flex !important;
        justify-content: flex-start !important;
        padding: 10px 16px !important;
        text-align: left !important;
    }
    .sidebar-btn button:hover {
        background-color: #333537 !important;
        color: #fff !important;
    }
    
    .active-chat button {
        background-color: #2b3952 !important;
        color: #a8c7fa !important;
    }
    
    .profile-card {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px;
        background: #1a1a1a;
        border-radius: 12px;
        margin-bottom: 10px;
    }
    .profile-img {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        object-fit: cover;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 4. AUTHENTICATION GATEWAY
# ════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    _, center_col, _ = st.columns([1, 1.5, 1])
    with center_col:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown('<div style="text-align:center; font-size: 60px;">✨</div>', unsafe_allow_html=True)
        st.markdown('<h1 style="text-align:center; font-size:32px; font-weight:600; margin-bottom:10px;">Gemini</h1>', unsafe_allow_html=True)
        st.markdown('<p style="text-align:center; color:#888; margin-bottom:30px;">Sign in with your Google Account to continue to the clone application.</p>', unsafe_allow_html=True)
        
        auth_result = oauth2.authorize_button(
            name="Sign in with Google",
            icon="https://www.google.com/favicon.ico",
            redirect_uri=REDIRECT_URI,
            scope=SCOPE,
            key="google_auth_btn",
            use_container_width=True,
        )
        if auth_result and "token" in auth_result:
            try:
                token_payload = jwt.decode(auth_result["token"].get("id_token", ""), options={"verify_signature": False})
                st.session_state.user_info = token_payload
                st.session_state.authenticated = True
                st.rerun()
            except Exception as e:
                st.error(f"Authentication handling error: {e}")
    st.stop()

# Context Variables for Active User Profile
user_data  = st.session_state.user_info or {}
user_email = user_data.get("email", "unknown_user")
user_name  = user_data.get("name", "User")
user_pic   = user_data.get("picture", "")

# Ensure user bucket exists inside the chat database
if user_email not in st.session_state.db_chats:
    st.session_state.db_chats[user_email] = {}

user_conversations = st.session_state.db_chats[user_email]

# Helper to save context seamlessly
def save_current_session_to_db(first_prompt_text):
    cid = st.session_state.current_chat_id or datetime.now().strftime("%Y%m%d%H%M%S%f")
    st.session_state.current_chat_id = cid
    
    # Generate static short title if it's a new log
    existing_title = user_conversations.get(cid, {}).get("title", "")
    title = existing_title if existing_title else (first_prompt_text[:30] + "..." if len(first_prompt_text) > 30 else first_prompt_text)
    
    user_conversations[cid] = {
        "title": title,
        "messages": st.session_state.messages,
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.db_chats[user_email] = user_conversations
    save_all_chats(st.session_state.db_chats)

# ════════════════════════════════════════════════════════════════════════════
# 5. SIDEBAR NAVIGATION & PERSISTENCE
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-btn">', unsafe_allow_html=True)
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br><span style='color:#a8c7fa; font-weight:500; font-size: 13px; padding-left:5px;'>Recent</span>", unsafe_allow_html=True)
    
    # Loop over and display true saved historical conversations
    if user_conversations:
        ordered_chats = sorted(user_conversations.items(), key=lambda x: x[1].get("timestamp", ""), reverse=True)
        for cid, info in ordered_chats[:20]:
            is_active = (cid == st.session_state.current_chat_id)
            btn_class = "sidebar-btn active-chat" if is_active else "sidebar-btn"
            
            st.markdown(f'<div class="{btn_class}">', unsafe_allow_html=True)
            if st.button(f"💬 {info['title']}", key=f"hist_{cid}", use_container_width=True):
                st.session_state.messages = info["messages"]
                st.session_state.current_chat_id = cid
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 13px; color: #888; padding-left:12px; margin-top:4px;'>No history found</div>", unsafe_allow_html=True)

    # Push contents to the lower end of the pane
    st.markdown("<br>" * 8, unsafe_allow_html=True)
    st.markdown("---")
    
    # Profile Card & Interactive Logout Button
    st.markdown(f"""
    <div class="profile-card">
        <img class="profile-img" src="{user_pic if user_pic else 'https://cdn-icons-png.flaticon.com/512/149/149071.png'}">
        <div style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            <div style="font-size:14px; font-weight:500; color:#fff;">{user_name}</div>
            <div style="font-size:11px; color:#888;">{user_email}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚪 Sign Out", use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.user_info = None
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════
# 6. MAIN WORKSPACE CHAT PANEL
# ════════════════════════════════════════════════════════════════════════════

# Configuration Verification Barrier
if not GEMINI_API_KEY:
    st.error("Missing Global configuration: GEMINI_API_KEY is not configured inside Streamlit secrets.")
    st.stop()

# Empty State Layout Design
if not st.session_state.messages:
    _, core_canvas, _ = st.columns([1, 5, 1])
    with core_canvas:
        first_name = user_name.split()[0] if user_name else "there"
        st.markdown(f'<div class="gemini-greeting">Hello, {first_name}</div>', unsafe_allow_html=True)
        st.markdown('<div class="gemini-subtitle">How can I help you today?</div>', unsafe_allow_html=True)
        
        # Interactive Suggestion Cards Map
        sc1, sc2, sc3, sc4 = st.columns(4)
        suggestions = [
            ("Brainstorm concepts", "for a highly modular machine learning model pipeline", "💡", "Brainstorm concepts for a highly modular machine learning model pipeline"),
            ("Refactor code modules", "converting regular loops to clean Python understandings", "💻", "Show me clean examples of refactoring nested loops to comprehension models in Python"),
            ("Compose technical draft", "explaining core API routing systems to new teammates", "✉️", "Draft a short introductory email guide explaining API architectural layout routing to junior engineers"),
            ("Analyze data models", "explaining the main architectural layers of microservices", "⚛️", "Explain microservices architecture data separation principles plain and simple")
        ]
        
        for idx, col in enumerate([sc1, sc2, sc3, sc4]):
            with col:
                st.markdown(f"""
                <div class="suggestion-card">
                    <div class="suggestion-text">
                        <b>{suggestions[idx][0]}</b><br>
                        <span style="color: #c4c7c5; font-size:13px;">{suggestions[idx][1]}</span>
                    </div>
                    <div class="suggestion-icon">{suggestions[idx][2]}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Make the static cards active trigger anchors
                if st.button("Activate prompt", key=f"sug_click_{idx}", use_container_width=True, help="Click to inject prompt"):
                    st.session_state.messages.append({"role": "user", "content": suggestions[idx][3]})
                    st.rerun()

# Rendering Conversations History Loop
else:
    for text_packet in st.session_state.messages:
        avatar_icon = "👤" if text_packet["role"] == "user" else "✨"
        with st.chat_message(text_packet["role"], avatar=avatar_icon):
            st.markdown(text_packet["content"])

# Formulated Prompt Submission Interface Handler
if prompt := st.chat_input("Message Gemini..."):
    # Append user execution state instantly
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Trigger generation pipeline stream response
    with st.chat_message("assistant", avatar="✨"):
        text_element_anchor = st.empty()
        cumulative_response = ""
        
        try:
            # Reconstruct model history explicitly dynamically mapping to standard API parameters
            ai_model = genai.GenerativeModel('gemini-1.5-flash')
            history_translation = []
            
            for historical_node in st.session_state.messages[:-1]:
                history_translation.append({
                    "role": "user" if historical_node["role"] == "user" else "model",
                    "parts": [historical_node["content"]]
                })
            
            # Align architectural parity bounds for Google conversations requirements
            while history_translation and history_translation[0]["role"] != "user":
                history_translation.pop(0)
                
            active_chat_engine = ai_model.start_chat(history=history_translation)
            stream_result = active_chat_engine.send_message(prompt, stream=True)
            
            for visual_chunk in stream_result:
                cumulative_response += visual_chunk.text
                text_element_anchor.markdown(cumulative_response + "▌")
                time.sleep(0.005) # Smooth pipeline throttling
                
            text_element_anchor.markdown(cumulative_response)
            st.session_state.messages.append({"role": "assistant", "content": cumulative_response})
            
            # Save logs locally and permanently inside chat ledger tracking
            save_current_session_to_db(st.session_state.messages[0]["content"])
            st.rerun()
            
        except Exception as api_err:
            st.error(f"Execution handling failure: {str(api_err)}")
