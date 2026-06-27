import streamlit as st
import google.generativeai as genai
import os
from datetime import datetime

# ════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Gemini", page_icon="✨", layout="wide")

# ════════════════════════════════════════════════════════════════════════════
# 2. CUSTOM CSS (Gemini Dark Theme)
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Global Background and Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background-color: #131314 !important;
        color: #e3e3e3;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e1f20 !important;
        border-right: none;
    }
    
    /* Hide Streamlit elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Greeting Text Gradients */
    .gemini-greeting {
        background: -webkit-linear-gradient(74deg, #4285f4 0, #9b72cb 9%, #d96570 20%, #d96570 24%, #9b72cb 35%, #4285f4 44%, #9b72cb 50%, #d96570 56%, #131314 75%, #131314 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 56px;
        font-weight: 500;
        letter-spacing: -1px;
        line-height: 1.2;
        margin-top: 10vh;
    }
    .gemini-subtitle {
        color: #444746;
        font-size: 56px;
        font-weight: 500;
        letter-spacing: -1px;
        line-height: 1.2;
        margin-bottom: 40px;
    }

    /* Suggestion Cards */
    .suggestion-card {
        background-color: #1e1f20;
        border-radius: 12px;
        padding: 16px;
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }
    .suggestion-card:hover {
        background-color: #333537;
    }
    .suggestion-text {
        font-size: 15px;
        color: #e3e3e3;
    }
    .suggestion-icon {
        align-self: flex-end;
        background-color: #131314;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
    }

    /* Chat Input Styling */
    [data-testid="stChatInput"] {
        background-color: #1e1f20 !important;
        border-radius: 24px !important;
        border: none !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #e3e3e3 !important;
    }
    
    /* New Chat Button Styling */
    .new-chat-btn button {
        background-color: #1a1a1a !important;
        border: none !important;
        border-radius: 20px !important;
        color: #a8c7fa !important;
        font-weight: 500 !important;
        display: flex !important;
        justify-content: flex-start !important;
        padding: 10px 16px !important;
        transition: background-color 0.2s !important;
    }
    .new-chat-btn button:hover {
        background-color: #333537 !important;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# 3. SESSION STATE INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = None
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None

# ════════════════════════════════════════════════════════════════════════════
# 4. SIDEBAR (Navigation & Setup)
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_session = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<span style='color:#a8c7fa; font-weight:500; font-size: 14px;'>Recent</span>", unsafe_allow_html=True)
    
    # Display recent chat history titles (mocked based on current session for simplicity)
    if st.session_state.messages:
        first_prompt = st.session_state.messages[0]["content"]
        title = first_prompt[:25] + "..." if len(first_prompt) > 25 else first_prompt
        st.markdown(f"<div style='padding: 8px 12px; border-radius: 8px; font-size: 14px; cursor: pointer; color: #e3e3e3; margin-top: 8px; background-color: #333537;'>💬 {title}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size: 13px; color: #888; margin-top: 8px;'>No recent chats</div>", unsafe_allow_html=True)

    st.markdown("<br>" * 10, unsafe_allow_html=True)
    st.markdown("---")
    
    # API Key Configuration
    api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter your key here...")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            st.session_state.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            if st.session_state.chat_session is None:
                st.session_state.chat_session = st.session_state.gemini_model.start_chat(history=[])
            st.success("API Key configured!", icon="✅")
        except Exception as e:
            st.error(f"Error configuring API: {e}")

# ════════════════════════════════════════════════════════════════════════════
# 5. MAIN CHAT INTERFACE
# ════════════════════════════════════════════════════════════════════════════

# A. Empty State (Greeting & Suggestions)
if not st.session_state.messages:
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.markdown('<div class="gemini-greeting">Hello,</div>', unsafe_allow_html=True)
        st.markdown('<div class="gemini-subtitle">How can I help you today?</div>', unsafe_allow_html=True)
        
        # Suggestion Cards
        sc1, sc2, sc3, sc4 = st.columns(4)
        
        cards = [
            ("Brainstorm ideas", "for a tech startup combining AI and agriculture", "💡"),
            ("Draft an email", "to my boss requesting time off for next week", "✉️"),
            ("Explain a concept", "how quantum computing works to a 5-year-old", "⚛️"),
            ("Write code", "to build a simple web scraper in Python", "💻")
        ]
        
        for i, col in enumerate([sc1, sc2, sc3, sc4]):
            with col:
                st.markdown(f"""
                <div class="suggestion-card">
                    <div class="suggestion-text">
                        <b>{cards[i][0]}</b><br>
                        <span style="color: #c4c7c5;">{cards[i][1]}</span>
                    </div>
                    <div class="suggestion-icon">{cards[i][2]}</div>
                </div>
                """, unsafe_allow_html=True)

# B. Filled State (Chat History)
else:
    for message in st.session_state.messages:
        # User message
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(message["content"])
        # Assistant message
        else:
            with st.chat_message("assistant", avatar="✨"):
                st.markdown(message["content"])

# C. Input Handling
if prompt := st.chat_input("Enter a prompt here"):
    if not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar to start chatting.", icon="⚠️")
        st.stop()
        
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant", avatar="✨"):
        response_placeholder = st.empty()
        
        try:
            # Stream the response from Gemini
            response = st.session_state.chat_session.send_message(prompt, stream=True)
            full_response = ""
            
            for chunk in response:
                full_response += chunk.text
                response_placeholder.markdown(full_response + "▌")
                
            # Finalize response without the cursor
            response_placeholder.markdown(full_response)
            
            # Save to history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
