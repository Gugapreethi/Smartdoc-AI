import streamlit as st
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="centered"
)

# CSS
st.markdown("""
<style>
    .stApp {
        background-color: #f0f4f8;
    }
    h1 {
        color: #1a73e8;
        text-align: center;
    }
    h2, h3 {
        color: #1557b0;
    }
    .stSelectbox label {
        color: #1a73e8;
        font-weight: bold;
    }
    .stTextInput input {
        background-color: #ffffff;
        border: 2px solid #1a73e8;
        border-radius: 10px;
        color: #333333;
    }
    .stTextArea textarea {
        background-color: #ffffff;
        border: 2px solid #1a73e8;
        border-radius: 10px;
        color: #333333;
    }
    .stButton button {
        background-color: #1a73e8;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #1557b0;
        color: white;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        border: 1px solid #e0e0e0;
    }
    .stFileUploader {
        background-color: #ffffff;
        border: 2px dashed #1a73e8;
        border-radius: 10px;
        padding: 10px;
    }
    .module-card {
        background-color: #ffffff;
        border: 2px solid #1a73e8;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin: 10px;
        cursor: pointer;
    }
    .module-card:hover {
        background-color: #e8f0fe;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "page" not in st.session_state:
    st.session_state.page = "home"

# ============ HOME PAGE ============
if st.session_state.page == "home":

    # Logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists("logo.png"):
            st.image("logo.png", width=250)

    st.title("🤖 SmartDoc AI")
    st.markdown(
        "<p style='text-align:center; color:#666;'>"
        "All in One AI Platform</p>",
        unsafe_allow_html=True
    )
    st.divider()

    # Module Cards
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄\n\nPDF Chat"):
            st.session_state.page = "pdf"
            st.session_state.messages = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🌐\n\nWeb Summarizer"):
            st.session_state.page = "web"
            st.session_state.messages = []
            st.rerun()

    with col2:
        if st.button("💬\n\nAI Chatbot"):
            st.session_state.page = "chat"
            st.session_state.messages = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🧑‍💻\n\nCode Review"):
            st.session_state.page = "code"
            st.session_state.messages = []
            st.rerun()

    with col3:
        if st.button("📊\n\nCSV Analyst"):
            st.session_state.page = "csv"
            st.session_state.messages = []
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🌍\n\nTranslator"):
            st.session_state.page = "translate"
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.markdown(
        "<p style='text-align:center; color:#999;'>"
        "Built with LangChain + Groq AI ❤️</p>",
        unsafe_allow_html=True
    )

# ============ BACK BUTTON ============
else:
    if st.button("⬅️ Back to Home"):
        st.session_state.page = "home"
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.session_state.vectorstore = None
        st.rerun()

# ============ MODULE 1 - PDF CHAT ============
if st.session_state.page == "pdf":
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_groq import ChatGroq
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    st.title("📄 PDF Chat")
    st.divider()

    uploaded_file = st.file_uploader(
        "Upload PDF", type="pdf"
    )

    if uploaded_file:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Processing PDF..."):
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            chunks = splitter.split_documents(pages)
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            st.session_state.vectorstore = (
                FAISS.from_documents(chunks, embeddings)
            )
        st.success("PDF Ready! Ask Questions")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.vectorstore:
        question = st.chat_input("Ask about your PDF...")
        if question:
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({
                "role": "user", "content": question
            })
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0
            )
            prompt = PromptTemplate.from_template("""
            Use context to answer question.
            Context: {context}
            Question: {question}
            Answer:
            """)
            retriever = (
                st.session_state.vectorstore.as_retriever()
            )
            chain = (
                {"context": retriever,
                 "question": RunnablePassthrough()}
                | prompt | llm | StrOutputParser()
            )
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = chain.invoke(question)
                st.write(answer)
            st.session_state.messages.append({
                "role": "assistant", "content": answer
            })

# ============ MODULE 2 - WEB SUMMARIZER ============
elif st.session_state.page == "web":
    import requests
    from bs4 import BeautifulSoup
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    st.title("🌐 Web Summarizer")
    st.divider()

    url = st.text_input("Enter URL:")
    style = st.selectbox("Summary Style:", [
        "Bullet Points", "Short Paragraph",
        "Detailed", "Key Points"
    ])

    if st.button("Summarize"):
        if url:
            with st.spinner("Reading webpage..."):
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(
                    url, headers=headers
                )
                soup = BeautifulSoup(
                    response.content, "html.parser"
                )
                for tag in soup([
                    "script", "style", "nav", "footer"
                ]):
                    tag.decompose()
                content = soup.get_text(
                    separator=" ", strip=True
                )[:4000]

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0
                )
                prompt = PromptTemplate.from_template("""
                Summarize this content.
                Style: {style}
                Content: {content}
                Summary:
                """)
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({
                    "content": content,
                    "style": style
                })

            with st.chat_message("assistant"):
                st.write(result)
            st.session_state.messages.append({
                "role": "assistant", "content": result
            })

# ============ MODULE 3 - AI CHAT ============
elif st.session_state.page == "chat":
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import HumanMessage, AIMessage

    st.title("💬 AI Chatbot")
    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input("Type your message...")
    if question:
        with st.chat_message("user"):
            st.write(question)
        st.session_state.messages.append({
            "role": "user", "content": question
        })
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.7
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "question": question,
            "chat_history": st.session_state.chat_history
        })
        with st.chat_message("assistant"):
            st.write(response)
        st.session_state.messages.append({
            "role": "assistant", "content": response
        })
        st.session_state.chat_history.append(
            HumanMessage(content=question)
        )
        st.session_state.chat_history.append(
            AIMessage(content=response)
        )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# ============ MODULE 4 - CODE REVIEW ============
elif st.session_state.page == "code":
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    st.title("🧑‍💻 Code Review")
    st.divider()

    language = st.selectbox("Language:", [
        "Python", "JavaScript", "Java", "C++", "SQL"
    ])
    review_type = st.selectbox("Review Type:", [
        "Full Review", "Bug Detection",
        "Security Check", "Performance Check"
    ])
    code = st.text_area("Paste Code:", height=200)

    if st.button("Review Code"):
        if code:
            with st.spinner("Reviewing..."):
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0
                )
                prompt = PromptTemplate.from_template("""
                Expert code reviewer.
                Language: {language}
                Review: {review_type}
                Code: {code}
                Provide bugs and suggestions.
                Review:
                """)
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({
                    "code": code,
                    "language": language,
                    "review_type": review_type
                })
            with st.chat_message("assistant"):
                st.write(result)

# ============ MODULE 5 - CSV ANALYST ============
elif st.session_state.page == "csv":
    import pandas as pd
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    st.title("📊 CSV Analyst")
    st.divider()

    uploaded_csv = st.file_uploader(
        "Upload CSV", type="csv"
    )

    if uploaded_csv:
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df.head(5))

        df_info = f"""
        Columns: {list(df.columns)}
        Shape: {df.shape}
        Stats: {df.describe().to_string()}
        """

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        question = st.chat_input(
            "Ask about your data..."
        )
        if question:
            with st.chat_message("user"):
                st.write(question)
            st.session_state.messages.append({
                "role": "user", "content": question
            })
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0
            )
            prompt = PromptTemplate.from_template("""
            Data analyst expert.
            Data: {df_info}
            Question: {question}
            Answer:
            """)
            chain = prompt | llm | StrOutputParser()
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    answer = chain.invoke({
                        "df_info": df_info,
                        "question": question
                    })
                st.write(answer)
            st.session_state.messages.append({
                "role": "assistant", "content": answer
            })

# ============ MODULE 6 - TRANSLATOR ============
elif st.session_state.page == "translate":
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    st.title("🌍 Translator")
    st.divider()

    text = st.text_area("Enter Text:", height=150)
    language = st.selectbox("Target Language:", [
        "Tamil", "Hindi", "French",
        "Spanish", "German", "Japanese"
    ])
    tone = st.selectbox("Tone:", [
        "Formal", "Casual",
        "Professional", "Friendly"
    ])

    if st.button("Translate"):
        if text:
            with st.spinner("Translating..."):
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.3
                )
                prompt = PromptTemplate.from_template("""
                Translate to {language} with {tone} tone.
                Text: {text}
                Result:
                """)
                chain = prompt | llm | StrOutputParser()
                result = chain.invoke({
                    "text": text,
                    "language": language,
                    "tone": tone
                })
            with st.chat_message("assistant"):
                st.write(result)