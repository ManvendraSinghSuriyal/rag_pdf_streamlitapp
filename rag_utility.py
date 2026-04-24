import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
import dotenv

# if get errro for spacy then run this command in terminal "python -m spacy download en"
# import unstructured_pytesseract
# unstructured_pytesseract.pytesseract.tesseract_cmd = 'C:/Users/msinghsuriyal/AppData/Local/Programs/Tesseract-OCR/tesseract.exe'


dotenv.load_dotenv()

working_dir = os.path.dirname(os.path.abspath((__file__)))

# Load the embedding model
embedding = HuggingFaceEmbeddings()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)



def process_document_to_chroma_db(filename):
    loader = UnstructuredPDFLoader(f"{working_dir}/{filename}")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    chuncks_document = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(
        documents=chuncks_document,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore",

    )

    return 0

def answer_question(query):


    vector_store = Chroma(
        embedding_function=HuggingFaceEmbeddings(),
        persist_directory=f"{working_dir}/doc_vectorstore",
    )

    retriever = vector_store.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    response = qa_chain.invoke({"query": query})

    ans=response["result"]
    return ans
