import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Translator AI",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Translator + Tone Changer AI")
st.subheader("Translate Any Text & Change Tone!")
st.divider()

def translate_text(text, target_language, tone):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

    prompt = PromptTemplate.from_template("""
    You are an expert translator and writing assistant.

    Task 1 - Translate the following text to: {target_language}
    Task 2 - Apply this tone to the translation: {tone}

    Original Text:
    {text}

    Please provide:
    1. Translated Text
    2. Tone Applied Version

    Result:
    """)

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "text": text,
        "target_language": target_language,
        "tone": tone
    })

def change_tone_only(text, tone):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3
    )

    prompt = PromptTemplate.from_template("""
    You are an expert writing assistant.

    Change the tone of the following text to: {tone}

    Original Text:
    {text}

    Tone Changed Version:
    """)

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "text": text,
        "tone": tone
    })

# Mode Select
mode = st.radio(
    "Select Mode:",
    [
        "Translate + Tone Change",
        "Tone Change Only"
    ]
)

st.divider()

# Text Input
text = st.text_area(
    "Enter Your Text:",
    height=150,
    placeholder="Type or paste your text here..."
)

if mode == "Translate + Tone Change":
    # Language Select
    target_language = st.selectbox(
        "Select Target Language:",
        [
            "Tamil",
            "Hindi",
            "French",
            "Spanish",
            "German",
            "Japanese",
            "Arabic",
            "Chinese",
            "Korean",
            "Italian"
        ]
    )

# Tone Select
tone = st.selectbox(
    "Select Tone:",
    [
        "Formal",
        "Casual",
        "Professional",
        "Friendly",
        "Persuasive",
        "Simple",
        "Academic"
    ]
)

if st.button("Convert"):
    if text:
        with st.spinner("Processing..."):
            if mode == "Translate + Tone Change":
                result = translate_text(
                    text,
                    target_language,
                    tone
                )
            else:
                result = change_tone_only(
                    text,
                    tone
                )
        st.divider()
        st.markdown("### Result:")
        st.write(result)
    else:
        st.warning("Please Enter Some Text!")