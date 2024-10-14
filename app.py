from dotenv import load_dotenv
load_dotenv() ##load all the environment varaible from .env
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    try:
        # Convert bytes object to a file-like object
        pdf_file = io.BytesIO(pdf_bytes)
        
        # Read the PDF
        pdf_reader = PdfReader(pdf_file)
        text = "" #reading full pdf
        for page in pdf_reader.pages:
            text+=page.extract_text()#extracting all pages
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

#spliting text into small chunks of text
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#chunks into vectors
def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template=""" Answer the question as detailed as possible from the provided context, make sure to provided all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question:\n{question}\n

    Answer:"""

    model= ChatGoogleGenerativeAI(mode="gemini-pro",temperature=0.3)
    prompt =PromptTemplate(template=prompt_template,input_variables=["context","question"])
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="model/embedding-001")

    new_db= FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs= new_db.similarity_search(user_question)

    chain = get_conversational_chain()


    response =chain(
        {"input_documents":docs,"question":user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config(page_title="Chat with Multiple PDF")
    st.header("Chat with PDF using Gemini Pro")

    user_question = st.text_input("Ask a question from the PDF files")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing complete. You can now ask questions.")
            else:
                st.warning("Please upload at least one PDF file.")

    if user_question:
        user_input(user_question)




if __name__ == "__main__":
    main()
