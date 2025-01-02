import streamlit as st
import json
from predictionguard import PredictionGuard

# Load API key from Streamlit secrets
api_key = st.secrets["api"]["PREDICTIONGUARD_API_KEY"]

# Initialize PredictionGuard client
client = PredictionGuard(str(api_key), "https://globalpath.predictionguard.com")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

# Debugging log
st.write("Current chat history:", st.session_state.chat_history)

# Display the chat messages
if st.session_state.chat_history:
    for chat in st.session_state.chat_history:
        st.write(f"**{chat['role'].capitalize()}:** {chat['content']}")

# Input field and button inside a form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Your message:", placeholder="Type your message here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    # Append user's input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.write(f"**User:** {user_input}")  # Display user input immediately

    # Generate chatbot response
    with st.spinner("Generating response..."):
        try:
            response = client.chat.completions.create(
                model="Hermes-3-Llama-3.1-70B",
                messages=[
                    *messages,
                    *[
                        {"role": chat["role"], "content": chat["content"]}
                        for chat in st.session_state.chat_history
                    ],
                    {"role": "user", "content": user_input},
                ]
            )['choices'][0]['message']['content'].strip()

            # Append chatbot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.write(f"**Assistant:** {response}")  # Display assistant response immediately
        except Exception as e:
            st.error(f"An error occurred: {e}")
