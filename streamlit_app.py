import streamlit as st
import json
from predictionguard import PredictionGuard

# Load API key from Streamlit secrets
api_key = st.secrets["api"]["PREDICTIONGUARD_API_KEY"]

# Initialize PredictionGuard client
client = PredictionGuard(api_key, "https://globalpath.predictionguard.com")

# Chatbot configuration
messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant that provides clever and sometimes funny responses."
    }
]

# Streamlit app interface
st.title("Chatbot Interface")
st.write("Welcome to the Chatbot! Let me know how I can assist you.")

# Display the chat messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for chat in st.session_state.chat_history:
    st.write(f"**{chat['role'].capitalize()}:** {chat['content']}")

# User input
user_input = st.text_input("Your message:", key="user_input")

if user_input:
    # Append user's message
    messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate chatbot response
    with st.spinner("Generating response..."):
        response = client.chat.completions.create(
            model="Hermes-3-Llama-3.1-70B",
            messages=messages
        )['choices'][0]['message']['content'].strip()

    # Append assistant's response
    messages.append({
        "role": "assistant",
        "content": response
    })

    # Update chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Clear input box
    st.experimental_rerun()
