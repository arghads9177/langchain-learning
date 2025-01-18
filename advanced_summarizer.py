# Import libraries
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variable
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o")
# Define custom prompts
map_prompt_template = """
            You are an expert summarizer. Summarize the following text clearly and concisely:

            {text}

            Summary:
            """
combine_prompt_template = """
            Combine the following summaries into a single, concise, and coherent summary:

            {text}
            The final summary should:
            - Be easy to understand.
            - Focus on key points and main ideas.
            - Avoid unnecessary details and repetitions.
            - It should be in a format of a linkedIn post.

            Final Summary:
            """
map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
# Define Page Confifuration
st.set_page_config(
    page_title="Content Summarizer",
    page_icon= "",
    layout= "wide"
)

# Define page header
st.header("Web Content Summarizer")
st.subheader("Powerd By GPT 4o")

url_input = st.text_area("Enter URLs (one per line):")

if st.button("Summarize"):
    if url_input.strip():
        with st.spinner("Please wait. Reading the contents from URL..."):
            # url = url_input
            # loader = WebBaseLoader(url)
            urls = [url.strip() for url in url_input.splitlines() if url.strip()]
            loader = WebBaseLoader(urls)
            documents= loader.load()
            st.write("Document loaded successfully.")
        with st.spinner("Please wait. Summarizing textL..."):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size= 5000,
                chunk_overlap= 200
            )
            chunks = text_splitter.split_documents(documents)
            print(len(chunks))
            chain = load_summarize_chain(
                llm= model, 
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt= combine_prompt
            )
            # summary = chain.invoke(documents)
            summary = chain.invoke(chunks)
            st.subheader("Summary:")
            st.write(summary["output_text"])
    else:
        st.error("Please enter a URL for summarization")

