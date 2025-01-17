# Import libraries
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from dotenv import load_dotenv
import os

# Load the environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define prompts
system_template = """
You are a helpful assistant. The user will provide a sentence in English as input.
You need to rewrite the sentence according to the {tone} provided by the user.
"""
human_template= "{text}"

system_prompt_template = SystemMessagePromptTemplate.from_template(system_template)
human_prompt_template = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate([system_prompt_template, human_prompt_template])

# Define LLM model
model = ChatOpenAI(model="gpt-3.5-turbo")

# Define the chain
chain = chat_prompt | model

# Define the Streamlit app
st.set_page_config(
    page_title="Text Tone Converter", 
    page_icon="📝",
    layout= "wide"
    )

st.title("Text Tone Converter")
st.header("Powered by OpenAI's GPT-3.5")

input_text = st.text_area("Enter a sentence")
tone = st.selectbox("Select your tone", options=["Angry", "Sweet", "Request"])

if st.button("Rewrite"):
    with st.spinner("I am thinking. Pleas wait for a while..."):
        output = chain.invoke({"text": input_text, "tone": tone})
        st.write(output.content)