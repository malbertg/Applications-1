import os
import tempfile

# langchain-community
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

# Define the prompts
prompt_template_questions = """
You are an expert in creating practice questions based on a given text.
Your goal is help a student to learn English. You do this by asking questions about the text below:

------------
{text}
------------

Create questions about the content of the text. Make sure that answering those questions implies a full comprehension of the text and its vocabulary and grammar. 

QUESTIONS:
"""

PROMPT_QUESTIONS = PromptTemplate(
    template=prompt_template_questions, input_variables=["text"]
)

refine_template_questions = """ 
You are an expert in creating practice questions based on a given text.
Your goal is help a student to learn English. 
We have received some practice questions to a certain extent: {existing_answer}.
We have the option to refine the existing questions or add new ones.
(only if necessary) with some more context below.
------------
{text}
------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.

QUESTIONS:
"""
REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template_questions,
)


def initialize_app():
    st.title("ReadWise: Your Q&A Generator")
    st.markdown(
        "<style>h1{color: #66ccff; text-align: center;}</style>", unsafe_allow_html=True
    )
    st.subheader("Empowering Language Acquisition")
    st.markdown(
        "<style>h3{color: aquamarine;  text-align: center;}</style>",
        unsafe_allow_html=True,
    )


def upload_pdf_file():
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            return temp_file.name
    return None


def load_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    text_pages = loader.load()

    # Combine text from pages into a single string
    text = ""
    for page in text_pages:
        text += page.page_content

    # Convert text to string
    text = str(text)

    return text


def generate_questions_and_answers(text):
    print("Generating questions and answers...")
    text_splitter_question_gen = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=50
    )
    text_chunks_question_gen = text_splitter_question_gen.split_text(text)
    docs_question_gen = [Document(page_content=t) for t in text_chunks_question_gen]

    llm_question_gen = Replicate(
        model="mistralai/mistral-7b-instruct-v0.2:f5701ad84de5715051cb99d550539719f8a7fbcf65e0e62a3d1eb3f94720764e",
        input={"temperature": 0.01, "max_length": 500, "top_p": 1},
    )

    question_gen_chain = load_summarize_chain(
        llm=llm_question_gen,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    questions = question_gen_chain.run(docs_question_gen)

    llm_answer_gen = Replicate(
        model="mistralai/mistral-7b-instruct-v0.2:f5701ad84de5715051cb99d550539719f8a7fbcf65e0e62a3d1eb3f94720764e",
        input={"temperature": 0.01, "max_length": 500, "top_p": 1},
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = Chroma.from_documents(docs_question_gen, embeddings)

    answer_gen_chain = RetrievalQA.from_chain_type(
        llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever(k=2)
    )

    question_list = questions.split("\n")
    qa_pairs = []

    for question in question_list:
        answer = answer_gen_chain.run(question)
        qa_pairs.append((question, answer))

    print("Questions and answers generated.")
    return qa_pairs


def save_questions_and_answers(qa_pairs):
    answers_dir = os.path.join(tempfile.gettempdir(), "answers")
    os.makedirs(answers_dir, exist_ok=True)
    qa_file_path = os.path.join(answers_dir, "questions_and_answers.txt")

    with open(qa_file_path, "w") as qa_file:
        for idx, (question, answer) in enumerate(qa_pairs):
            qa_file.write(f"Question {idx + 1}: {question}\n")
            qa_file.write(f"Answer {idx + 1}: {answer}\n")
            qa_file.write("--------------------------------------------------\n\n")

    return qa_file_path


def display_questions_and_answers(qa_pairs):
    for question, answer in qa_pairs:
        st.write("Question: ", question)
        st.write("Answer: ", answer)
        st.write("--------------------------------------------------\n\n")


def download_questions_and_answers(file_path):
    with open(file_path, "rb") as file:
        file_contents = file.read()
    st.download_button(
        label="Download Questions and Answers",
        data=file_contents,
        file_name="questions_and_answers.txt",
        mime="text/plain",
    )


def cleanup_temporary_files(file_path):
    if file_path:
        os.remove(file_path)


def main():
    initialize_app()
    file_path = upload_pdf_file()
    if file_path:
        text = load_text_from_pdf(file_path)
        qa_pairs = generate_questions_and_answers(text)
        display_questions_and_answers(qa_pairs)
        qa_file_path = save_questions_and_answers(qa_pairs)
        download_questions_and_answers(qa_file_path)
        cleanup_temporary_files(file_path)


if __name__ == "__main__":
    main()
