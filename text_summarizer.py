# Import libraries
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import streamlit as st

# Load environment variable
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model="gpt-4o")
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
                chain_type="map_reduce"
            )
            # summary = chain.invoke(documents)
            summary = chain.invoke(chunks)
            st.subheader("Summary:")
            st.write(summary["output_text"])
    else:
        st.error("Please enter a URL for summarization")

