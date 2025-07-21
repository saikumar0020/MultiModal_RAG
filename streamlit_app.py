import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# Backend FastAPI endpoint
BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Multimodal RAG Chatbot",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("🧠 Multimodal RAG Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["type"] == "text":
            st.markdown(msg["content"])
        elif msg["type"] == "image":
            st.image(msg["content"], caption=msg.get("caption", None), use_column_width=True)

# Input from user
if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "type": "text", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing with Multimodal RAG..."):
            try:
                response = requests.post(f"{BASE_URL}/query", json={"question": user_input})

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "🤔 No response from agent.")
                    print("Answer ---> ", answer)
                    image_summaries = data.get("image_summaries", [])
                    image_base64 = data.get("image_base64", [])
                else:
                    answer = f"❌ Error from agent: {response.status_code} - {response.text}"
                    image_summaries = []
                    image_base64 = []

            except Exception as e:
                answer = f"🚨 Request failed: {e}"
                image_summaries = []
                image_base64 = []

        # Display text answer
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "type": "text", "content": answer})

        # Display images with summaries
        for idx, (img_str, summary) in enumerate(zip(image_base64, image_summaries)):
            try:
                img_data = base64.b64decode(img_str)
                img = Image.open(BytesIO(img_data))
                st.image(img, caption=summary, use_column_width=True)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "type": "image", 
                    "content": img, 
                    "caption": summary
                })
            except Exception as e:
                st.warning(f"Could not render image {idx}: {e}")
