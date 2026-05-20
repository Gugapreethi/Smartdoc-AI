import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import tempfile

load_dotenv()

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 SmartDoc AI")
st.subheader("PDF-ல என்ன வேணும்னாலும் கேளு!")
st.divider()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)

    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vectorstore

def ask_question(vectorstore, question):

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
    Use the following context to answer the question.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """)

    retriever = vectorstore.as_retriever()

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)

uploaded_file = st.file_uploader(
    "📄 PDF Upload பண்ணு",
    type="pdf"
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("📄 PDF Processing..."):
        vectorstore = process_pdf(tmp_path)

    st.success("✅ PDF Ready! Questions கேளு")
    st.divider()

    question = st.text_input("❓ உன் Question:")

    if st.button("🔍 Answer கேளு"):

        if question:

            with st.spinner("🤖 Thinking..."):
                answer = ask_question(
                    vectorstore,
                    question
                )

            st.divider()
            st.markdown("### 🤖 Answer:")
            st.write(answer)

        else:
            st.warning("⚠️ Question type பண்ணு!")