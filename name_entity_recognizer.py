# Import libraries
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Define NER schema

ner_schema = (
    "Extract the following entities based on the following schema:\n"
    "- name: string (required)\n"
    "- height: float (required)\n",
    "- hair_color: string (optional)"
)

# Define model
model = ChatOpenAI(model= "gpt-3.5-turbo")
# Define a prompt with this schema
ner_prompt_template ="""
You are a helpful assisteant trained to extract the entities from a text strictly following the schema:
{ner_schema}

Text:
{{text}}

Extracted entities in JSON format.
"""

ner_prompt = PromptTemplate(template=ner_prompt_template, input_variables=["text"])
# Define chain
chain = ner_prompt | model

# input_text = """
# Aghra is 5.5 feet and he is 1 feet taller than Swagata". He has short black hair but Swagata is brunette.
# """
# print(chain.invoke(input_text).content)

# Define page config
st.set_page_config(
    page_title="NER Extractor",
    layout="centered"
)

# Set title
st.title("NER Extractor")
st.header("Powered by GPT-3.5-Turbo")

input_text = st.text_area("Enter a text")
ner_btn = st.button("Extract NER")

try:
    if ner_btn:
        if input_text:
            with st.spinner("Extracting entities. Please wait for a while..."):
                result = chain.invoke(input_text).content
                st.json(result)
        else:
            st.error("Please enter a text.")
except Exception as e:
    st.error(f"An error occured: {e}")