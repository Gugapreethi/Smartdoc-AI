import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Code Review AI",
    page_icon="🧑‍💻",
    layout="centered"
)

st.title("🧑‍💻 Code Review Assistant")
st.subheader("Paste Your Code - AI Will Review It!")
st.divider()

def review_code(code, language, review_type):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = PromptTemplate.from_template("""
    You are an expert code reviewer.
    
    Programming Language: {language}
    Review Type: {review_type}
    
    Code to Review:
    {code}
    
    Please provide:
    1. Code Quality Analysis
    2. Bugs Found (if any)
    3. Security Issues (if any)
    4. Performance Improvements
    5. Best Practice Suggestions
    6. Corrected Code (if needed)
    
    Review:
    """)

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "code": code,
        "language": language,
        "review_type": review_type
    })

# Language Select
language = st.selectbox(
    "Select Programming Language:",
    [
        "Python",
        "JavaScript",
        "Java",
        "C++",
        "HTML/CSS",
        "SQL",
        "Other"
    ]
)

# Review Type Select
review_type = st.selectbox(
    "Select Review Type:",
    [
        "Full Review",
        "Bug Detection Only",
        "Security Check Only",
        "Performance Check Only",
        "Best Practices Only"
    ]
)

# Code Input
code = st.text_area(
    "Paste Your Code Here:",
    height=300,
    placeholder="Enter your code here..."
)

if st.button("Review Code"):
    if code:
        with st.spinner("Reviewing Your Code..."):
            review = review_code(
                code,
                language,
                review_type
            )
        st.divider()
        st.markdown("### Code Review Result:")
        st.write(review)
    else:
        st.warning("Please Paste Your Code!")