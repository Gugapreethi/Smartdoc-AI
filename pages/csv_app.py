import streamlit as st
from dotenv import load_dotenv
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="CSV Analyst AI",
    page_icon="📊",
    layout="centered"
)

st.title("📊 CSV Analyst AI")
st.subheader("Upload CSV - Ask Questions in Plain English!")
st.divider()

def analyze_csv(df_info, question):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
    You are a data analyst expert.
    
    CSV Data Information:
    {df_info}
    
    User Question:
    {question}
    
    Please analyze the data and answer the question
    clearly with insights and observations.
    
    Answer:
    """)

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "df_info": df_info,
        "question": question
    })

# CSV Upload
uploaded_file = st.file_uploader(
    "Upload Your CSV File",
    type="csv"
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("CSV Loaded Successfully!")
    st.divider()

    st.markdown("### Data Preview:")
    st.dataframe(df.head(10))

    st.markdown("### Data Info:")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    st.divider()

    df_info = f"""
    Columns: {list(df.columns)}
    Shape: {df.shape}
    Data Types: {df.dtypes.to_string()}
    Sample Data: {df.head(5).to_string()}
    Basic Stats: {df.describe().to_string()}
    """

    question = st.text_input(
        "Ask Question About Your Data:",
        placeholder="Which month had highest sales?"
    )

    if st.button("Analyze"):
        if question:
            with st.spinner("Analyzing Data..."):
                answer = analyze_csv(df_info, question)
            st.divider()
            st.markdown("### Analysis Result:")
            st.write(answer)
        else:
            st.warning("Please Enter a Question!")