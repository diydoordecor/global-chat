import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import lancedb
from lancedb.embeddings import with_embeddings
from predictionguard import PredictionGuard

# Initialize PredictionGuard and LanceDB
api_key = st.secrets["api"]["PREDICTIONGUARD_API_KEY"]
client = PredictionGuard(api_key, "https://globalpath.predictionguard.com")
db = lancedb.connect(".lancedb")  # Local database folder

# Clear existing tables in LanceDB to avoid contamination
for table_name in db.table_names():
    db.drop_table(table_name)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L12-v2")

# Streamlit interface
st.title("RAG-Enhanced Chatbot")
st.write("Upload a CSV file with visa profile data to get started.")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload Visa Profile CSV", type=["csv"])

if uploaded_file:
    # Load the CSV data
    st.write("Processing CSV file...")
    visa_data = pd.read_csv(uploaded_file)

    # Combine all columns into a single text column for embedding
    visa_data['text'] = visa_data.apply(lambda row: ' '.join(row.astype(str)), axis=1)

    # Generate embeddings
    st.write("Generating embeddings...")
    visa_data['embedding'] = visa_data['text'].apply(lambda x: model.encode(str(x)))

    # Add data to LanceDB
    st.write("Storing data in LanceDB...")
    data_with_embeddings = pd.DataFrame({
        "text": visa_data['text'],
        "embedding": list(visa_data['embedding'])
    })
    table = db.create_table("visa_profiles", data=with_embeddings(model.encode, data_with_embeddings))
    st.success("Visa profile data has been loaded successfully!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Define function for RAG-based retrieval
def get_relevant_context(query):
    # Ensure the query embedding is a 1-dimensional array
    query_embedding = model.encode(query)

    # Search the LanceDB table for relevant context
    results = table.search(query_embedding).limit(1).to_df()  # Use only the top result
    if not results.empty:
        return results.iloc[0]['text']
    else:
        return None

# Define function for generating response
def generate_response(user_query, chat_history):
    # Retrieve context
    context = get_relevant_context(user_query)

    # If no relevant context found, inform the user
    if not context:
        return "Sorry, I couldn't find an answer in the provided data."

    # Create a concise prompt
    prompt = f"""
Context:
{context}

Question:
{user_query}

Answer concisely using only the provided context. If the answer is not explicitly in the context, respond with:
"Sorry, I couldn't find an answer in the provided data."
"""

    # Get response from PredictionGuard
    result = client.completions.create(
        model="Hermes-3-Llama-3.1-70B",
        prompt=prompt,
        max_tokens=100,  # Allow longer responses
        temperature=0.1  # Ensure deterministic and focused output
    )
    return result['choices'][0]['text'].strip()

# Chat interface
user_query = st.text_input("Your question:", placeholder="Ask me a question about visa profiles...")
if st.button("Ask"):
    if uploaded_file and user_query.strip():
        # Append user query to chat history
        chat_history = "\n".join([f"User: {chat['content']}" if chat['role'] == "user" else f"Assistant: {chat['content']}" for chat in st.session_state.chat_history])
        chat_history += f"\nUser: {user_query}"

        # Generate response
        with st.spinner("Generating response..."):
            response = generate_response(user_query, chat_history)
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Display chat history (only user questions and assistant answers)
        for chat in st.session_state.chat_history:
            st.write(f"**{chat['role'].capitalize()}:** {chat['content']}")
    else:
        st.warning("Please upload a CSV file and enter a question.")
