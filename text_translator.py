# Import libraries
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import BaseOutputParser
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define comma separated output parser
class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of a LLM calll to a comma separated list.
    """
    def parse(self, text:str):
        return text.strip().split(", ")

# Define Prompt
system_template = """
You are a helpful assistant. The user will provide a sentence in any language.
You translate this input text into {output_languages}.
Separate the output for each lanuage by comma.
"""
human_template= "{text}"

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate([system_message_prompt, human_message_prompt])

# Define LLM model
model = ChatOpenAI(model= "gpt-4o")

# Define output parser
output_parser = CommaSeparatedListOutputParser()

# Define a chain
chain = chat_prompt | model | output_parser

# Define the Streamlit app
st.set_page_config(
    page_title="Text Translator", 
    page_icon="📝",
    layout= "wide"
    )

st.title("Text Translator")
st.header("Powered by OpenAI's GPT-4o")

input_text = st.text_area("Enter a sentence")
output_languages= st.text_input("Enter output languages", "Bengali, Hindi")

if st.button("Translate"):
    with st.spinner("I am thinking. Pleas wait for a while..."):
        output = chain.invoke({"text": input_text, "output_languages": {output_languages}})
        print(output)
        st.write(output)