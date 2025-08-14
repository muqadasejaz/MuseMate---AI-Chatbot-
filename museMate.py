import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#Page Config 
st.set_page_config(page_title="MuseMate 🎨🤖", page_icon="🤖", layout="centered")

# Initialize Model 
@st.cache_resource
def init_model():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        # mistralai/Mixtral-8x22B-Instruct-v0.1
        task="text-generation"
    )
    return ChatHuggingFace(llm=llm)

model = init_model()

# Initialize Chat History 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are MuseMate 🎨🤖 — a friendly, playful, and creative AI assistant.")]

#App Title 
st.title("MuseMate 🎨🤖")
st.caption("Your friendly & creative AI chat companion")

#Display Chat Messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

#User Input
if prompt := st.chat_input("Type your message... 💬"):
    # Add user message to chat history
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    # Display user's message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("MuseMate is thinking... 🤔"):
            result = model.invoke(st.session_state.chat_history)
            st.markdown(result.content)

    # Add AI message to history
    st.session_state.chat_history.append(AIMessage(content=result.content))
