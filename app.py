# ✅ Fixed version of app.py (Hugging Face compatible)
# Uses langchain-community and fixes import errors

import os
import gradio as gr
import csv
import tempfile


# ✅ Use this version of langchain-community loader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

qa_chain = None
source_docs = []

# Step 1: Load PDF and build retriever
def process_pdf(pdf_file, openai_api_key):
    global qa_chain, source_docs
    if not pdf_file:
        return "Please upload a PDF file first."

    os.environ["OPENAI_API_KEY"] = openai_api_key

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file)
        tmp_path = tmp.name
    
    loader = PyMuPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embedding_model)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )

    return f"✅ Document processed. {len(chunks)} chunks indexed. You can now ask questions."

# Step 2: Ask question
def ask_question(user_input):
    global qa_chain, source_docs
    if not qa_chain:
        return "Please upload and load a PDF document first."

    result = qa_chain.invoke({"query": user_input})
    answer = result["result"]
    source_docs = result["source_documents"]

    summary_prompt = f"Summarize the main idea of the following content:\n\n{source_docs[0].page_content}"
    summary = qa_chain.combine_documents_chain.llm_chain.llm.predict(summary_prompt)

    highlighted = "\n---\n".join(doc.page_content[:500] + "..." for doc in source_docs[:2])
    return answer, summary, highlighted

# Step 3: Save feedback
def save_feedback(user_input, answer, feedback):
    with open("feedback.csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_input, answer, feedback])
    return "✅ Thanks for your feedback!"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📄 PDF Q&A Assistant (RAG + GPT)")

    with gr.Row():
        pdf_file = gr.File(label="Upload PDF", type="binary")
        openai_key_box = gr.Textbox(label="🔐 OpenAI API Key", type="password")
        load_button = gr.Button("📚 Load Document")
    load_status = gr.Textbox(label="Status")

    load_button.click(fn=process_pdf, inputs=[pdf_file, openai_key_box], outputs=[load_status])

    with gr.Row():
        question_input = gr.Textbox(label="Ask a Question")
        ask_button = gr.Button("🔍 Ask")

    answer_output = gr.Textbox(label="Answer")
    summary_output = gr.Textbox(label="Summary")
    source_output = gr.Textbox(label="Relevant Chunks", lines=8)

    ask_button.click(fn=ask_question, inputs=[question_input], outputs=[answer_output, summary_output, source_output])

    gr.Markdown("## 🙋‍♂️ Was the answer helpful?")

    with gr.Row():
        good_btn = gr.Button("👍 Yes")
        bad_btn = gr.Button("👎 No")

    feedback_status = gr.Textbox(label="Feedback Status")

    good_btn.click(fn=save_feedback, inputs=[question_input, answer_output, gr.Textbox(value="👍")], outputs=[feedback_status])
    bad_btn.click(fn=save_feedback, inputs=[question_input, answer_output, gr.Textbox(value="👎")], outputs=[feedback_status])

# Launch
demo.launch()
