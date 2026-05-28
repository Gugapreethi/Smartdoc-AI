import streamlit as st
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Web Summarizer AI",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Web Summarizer AI")
st.subheader("எந்த Website-யும் Summary பண்ணலாம்!")
st.divider()

def get_webpage_content(url):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    
    text = soup.get_text(separator=" ", strip=True)
    return text[:4000]

def summarize_content(content, style):
    llm = ChatGroq(model="llama3-8b-8192", temperature=0)
    
    prompt = PromptTemplate.from_template("""
    Summarize the following webpage content.
    Summary Style: {style}
    
    Content: {content}
    
    Give a clear and concise summary in English.
    Summary:
    """)
    
    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "content": content,
        "style": style
    })

# URL Input
url = st.text_input(
    "🔗 Website URL கொடு:",
    placeholder="https://example.com"
)

# Summary Style
style = st.selectbox(
    "📝 Summary Style:",
    [
        "Bullet Points",
        "Short Paragraph",
        "Detailed Summary",
        "Key Points Only"
    ]
)

if st.button("🔍 Summarize பண்ணு"):
    if url:
        with st.spinner("🌐 Website Reading..."):
            try:
                content = get_webpage_content(url)
                st.info(f"✅ Content loaded!")
                
                with st.spinner("🤖 Summarizing..."):
                    summary = summarize_content(content, style)
                
                st.divider()
                st.markdown("### 📋 Summary:")
                st.write(summary)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ URL கொடு!")
