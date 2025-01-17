import streamlit as st
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
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

system_template = """
You are a english teather. The user will provide a english sentence as input.
Output Correct if it is grammatically correct otherwise output Incorrect and
rewrite the sentence in a grammatically correct way in the next line.
"""

human_template= "{text}"

system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate([system_message_prompt, human_message_prompt])

# Define LLM chain
# chain = LLMChain(
#     llm= ChatOpenAI(model="gpt-3.5-turbo"),
#     prompt= chat_prompt
# )

# print("Prompt 1:", chain.run("Get out house my you."))
# print("Prompt 2:", chain.run("I am a teacher."))

model = ChatOpenAI(model="gpt-3.5-turbo")
chain = chat_prompt | model

# print("Prompt 1:", chain.invoke("Get out house my you.").content)
# print("Prompt 2:", chain.invoke("I am a teacher.").content)

st.set_page_config(
    page_title="Grammar Checker", 
    page_icon="📝",
    layout= "wide"
    )

st.title("Grammar Checker")
st.header("Powered by OpenAI's GPT-3.5")

input_text = st.text_area("Enter a sentence in English", "Get out house my you.")

if st.button("Check Grammar"):
    output = chain.invoke(input_text).content
    st.write(output)