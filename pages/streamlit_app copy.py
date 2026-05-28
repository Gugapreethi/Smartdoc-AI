import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Streamlit Page Config
st.set_page_config(
    page_title="Web Summarizer AI",
    page_icon="🌐",
    layout="centered"
)

# Title
st.title("🌐 Web Summarizer AI")
st.subheader("எந்த Website-யும் Summary பண்ணலாம்!")
st.divider()

# Function to fetch webpage content
def get_webpage_content(url):
    
    # URL validation
    if not url.startswith("http"):
        raise ValueError("⚠️ சரியான URL கொடு! (https://...)")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # Request webpage
    response = requests.get(
        url,
        headers=headers,
        timeout=10
    )

    # Check request success
    response.raise_for_status()

    # Parse HTML
    soup = BeautifulSoup(response.content, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract clean text
    text = soup.get_text(separator=" ", strip=True)

    # Limit text size
    return text[:6000]


# Function to summarize content
def summarize_content(content, style):

    # Updated Groq model
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # Prompt Template
    prompt = PromptTemplate.from_template("""
    Summarize the following webpage content.

    Summary Style: {style}

    Webpage Content:
    {content}

    Give a clear and concise summary in English.
    """)

    # LangChain pipeline
    chain = prompt | llm | StrOutputParser()

    # Generate summary
    return chain.invoke({
        "content": content,
        "style": style
    })


# URL Input
url = st.text_input(
    "🔗 Website URL கொடு:",
    placeholder="https://example.com"
)

# Summary Style Options
style = st.selectbox(
    "📝 Summary Style:",
    [
        "Bullet Points",
        "Short Paragraph",
        "Detailed Summary",
        "Key Points Only"
    ]
)

# Summarize Button
if st.button("🔍 Summarize பண்ணு"):

    if url:

        try:
            # Fetch content
            with st.spinner("🌐 Website Reading..."):
                content = get_webpage_content(url)

            st.success("✅ Website Content Loaded!")

            # Generate summary
            with st.spinner("🤖 AI Summarizing..."):
                summary = summarize_content(content, style)

            # Display summary
            st.divider()
            st.markdown("## 📋 Summary")
            st.write(summary)

        except requests.exceptions.RequestException:
            st.error("❌ Website open பண்ண முடியல!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    else:
        st.warning("⚠️ URL கொடு!")


# Footer
st.divider()
st.caption("Made with ❤️ using Streamlit + Groq + LangChain")