import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🤖",
)

st.title("🤖 Zyro Dynamics HR Help Desk")
st.write("Ask questions about Zyro Dynamics HR policies.")

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


@st.cache_resource
def build_rag():

    pdf_folder = "data"

    documents = []

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_folder, file))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20
        }
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are the Zyro Dynamics HR Help Desk Assistant.

Answer ONLY using the provided context.

If the answer is not available in the context, reply exactly:

Information not found.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, rag_chain


retriever, rag_chain = build_rag()


def is_hr_question(question):

    docs = retriever.invoke(question)

    if len(docs) == 0:
        return False

    return True


question = st.text_input(
    "Enter your HR question"
)

if st.button("Ask"):

    if question.strip():

        if not is_hr_question(question):

            st.warning(
                "I can only answer questions related to Zyro Dynamics HR policies."
            )

        else:

            answer = rag_chain.invoke(question)

            st.success(answer)

            st.subheader("Sources")

            docs = retriever.invoke(question)

            for i, doc in enumerate(docs[:3], start=1):

                source = doc.metadata.get(
                    "source",
                    "Unknown"
                )

                st.write(f"Source {i}: {source}")
