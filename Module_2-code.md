import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 AI Chatbot")
st.subheader("Chat with AI - Remembers Your Conversation!")
st.divider()

# Session State Initialize
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_response(question, chat_history):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant.
        You remember previous conversations and give
        relevant answers based on chat history."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({
        "question": question,
        "chat_history": chat_history
    })

# Chat History Display
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat Input
question = st.chat_input("Type Your Message Here...")

if question:
    # User Message
    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = get_response(
                question,
                st.session_state.chat_history
            )
        st.write(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    # Update Chat History
    st.session_state.chat_history.append(
        HumanMessage(content=question)
    )
    st.session_state.chat_history.append(
        AIMessage(content=response)
    )

st.divider()

# Clear Button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()
