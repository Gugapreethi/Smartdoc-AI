import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load .env
load_dotenv()

# API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# LLM
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)


def load_pdf(file_path):
    print("馃搫 PDF Loading...")

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    print(f"鉁� {len(pages)} pages loaded!")

    return pages


def create_vectorstore(pages):
    print("馃攧 Processing...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(pages)

    print(f"鉁� {len(chunks)} chunks created!")

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector Store
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("鉁� Vector store ready!")

    return vectorstore


def ask_question(vectorstore, question):

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


def main():

    pdf_path = input("馃搧 PDF file path 喈曕瘖喈熰瘉: ")

    if not os.path.exists(pdf_path):
        print("鉂� File 喈囙喁嵿! 喈氞喈苦喈距 path 喈曕瘖喈熰瘉")
        return

    pages = load_pdf(pdf_path)

    vectorstore = create_vectorstore(pages)

    print("\n鉁� Ready! Questions 喈曕瘒喈赤瘉")
    print("-" * 40)

    while True:

        question = input("\n鉂� 喈夃喁� Question: ")

        if question.lower() == "quit":
            print("馃憢 Bye!")
            break

        answer = ask_question(vectorstore, question)

        print(f"\n馃 Answer:\n{answer}")


if __name__ == "__main__":
    main()
