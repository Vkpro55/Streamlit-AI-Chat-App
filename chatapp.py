import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.

    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    index_path = "faiss_index/index.faiss"

    # Check if the FAISS index file exists
    if not os.path.exists(index_path):
        st.error("The FAISS index file was not found. Please process the PDF files first.")
        return

    # Load the vector store if the index exists
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response["output_text"])
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")

def main():
    st.set_page_config("Steamlit AI Chatbot", page_icon="🤖")
    st.header("Multi-PDF's - Streamlit Gemini Chat Application 🤖 ")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️")

    if user_question:
        user_input(user_question)

    with st.sidebar:
    
        st.title("PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit Pdf"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing completed and FAISS index created.")

    
        
    st.markdown(
    """
    <style>
        .custom-footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #2C3E50;
            color: #ECF0F1;
            padding: 20px;
            text-align: center;
            font-size: 0.85rem;
            font-family: 'Helvetica Neue', sans-serif;
            border-top: 1px solid #34495E;
        }
        .custom-footer a {
            color: #E74C3C;
            text-decoration: none;
            font-weight: 600;
        }
        .custom-footer a:hover {
            color: #C0392B;
        }
        .heart {
            color: #E74C3C;
            animation: beat 1s infinite;
        }
        @keyframes beat {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.2); }
        }
    </style>
    <div class="custom-footer">
        Created with <span class="heart">❤️</span> by <a href="https://github.com/Vkpro55/Streamlit-AI-Chat-App" target="_blank">Vinod Kumar</a> | Powered by AI & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)


if __name__ == "__main__":
    main()
