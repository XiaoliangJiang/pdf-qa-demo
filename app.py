import os
import tempfile
import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from transformers import pipeline

qa_chain = None
source_docs = None


def process_pdf(pdf_file, openai_api_key, model_choice):
    global qa_chain, source_docs
    if not pdf_file:
        return "Please upload a PDF file first.", "", ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())  # Fixed: read bytes from uploaded file
        tmp_path = tmp.name

    loader = PyMuPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    if model_choice == "OpenAI GPT":
        os.environ["OPENAI_API_KEY"] = openai_api_key
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
        llm = ChatOpenAI(temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    else:
        qa_chain = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        source_docs = docs

    return "Document processed. You can now ask questions!", "", ""


def ask_question(question):
    if qa_chain is None:
        return "Please upload and load a PDF document first.", "", ""

    if isinstance(qa_chain, pipeline):
        context = "\n\n".join(doc.page_content for doc in source_docs[:5])
        result = qa_chain(question=question, context=context)
        return result['answer'], "", ""
    else:
        result = qa_chain.run(question)
        return result, "", ""


def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("""
        # 📄 PDF Q&A Assistant (RAG + GPT)

        Upload your PDF document, ask questions in natural language, and get summarized answers with source highlighting.
        You can choose between **OpenAI GPT** (requires API key) or a **free local model**. No data is stored.
        """)

        with gr.Row():
            pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
            model_choice = gr.Dropdown(label="Choose Model", choices=["OpenAI GPT", "Local Model"], value="OpenAI GPT")
            api_key = gr.Textbox(label="🔑 OpenAI API Key (only needed for GPT)", type="password", visible=True)
            load_button = gr.Button("📚 Load Document")

        status = gr.Textbox(label="Status")

        with gr.Row():
            question = gr.Textbox(label="Ask a Question")
            ask_button = gr.Button("🔍 Ask")

        answer = gr.Textbox(label="Answer")
        sources = gr.Textbox(label="Sources")

        def toggle_api_visibility(model):
            return gr.update(visible=(model == "OpenAI GPT"))

        model_choice.change(fn=toggle_api_visibility, inputs=model_choice, outputs=api_key)

        load_button.click(fn=process_pdf, inputs=[pdf_file, api_key, model_choice], outputs=[status, answer, sources])
        ask_button.click(fn=ask_question, inputs=[question], outputs=[answer, sources])

    return demo


app = build_ui()

if __name__ == "__main__":
    app.launch()

# # ✅ Fixed version of app.py (Hugging Face compatible)
# # Uses langchain-community and fixes import errors
#
# import os
# import gradio as gr
# import csv
# import tempfile
#
#
# # ✅ Use this version of langchain-community loader
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores.faiss import FAISS
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
#
# qa_chain = None
# source_docs = []
#
# # Step 1: Load PDF and build retriever
# def process_pdf(pdf_file, openai_api_key):
#     global qa_chain, source_docs
#     if not pdf_file:
#         return "Please upload a PDF file first."
#
#     os.environ["OPENAI_API_KEY"] = openai_api_key
#
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(pdf_file)
#         tmp_path = tmp.name
#
#     loader = PyMuPDFLoader(tmp_path)
#     documents = loader.load()
#
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(documents)
#
#     embedding_model = OpenAIEmbeddings()
#     vector_store = FAISS.from_documents(chunks, embedding_model)
#
#     llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=vector_store.as_retriever(),
#         return_source_documents=True
#     )
#
#     return f"✅ Document processed. {len(chunks)} chunks indexed. You can now ask questions."
#
# # Step 2: Ask question
# def ask_question(user_input):
#     global qa_chain, source_docs
#     if not qa_chain:
#         return "Please upload and load a PDF document first."
#
#     result = qa_chain.invoke({"query": user_input})
#     answer = result["result"]
#     source_docs = result["source_documents"]
#
#     summary_prompt = f"Summarize the main idea of the following content:\n\n{source_docs[0].page_content}"
#     summary = qa_chain.combine_documents_chain.llm_chain.llm.predict(summary_prompt)
#
#     highlighted = "\n---\n".join(doc.page_content[:500] + "..." for doc in source_docs[:2])
#     return answer, summary, highlighted
#
# # Step 3: Save feedback
# def save_feedback(user_input, answer, feedback):
#     with open("feedback.csv", "a", newline='', encoding="utf-8") as f:
#         writer = csv.writer(f)
#         writer.writerow([user_input, answer, feedback])
#     return "✅ Thanks for your feedback!"
#
# # Gradio UI
# with gr.Blocks() as demo:
#     gr.Markdown("# 📄 PDF Q&A Assistant (RAG + GPT)")
#
#     with gr.Row():
#         pdf_file = gr.File(label="Upload PDF", type="binary")
#         openai_key_box = gr.Textbox(label="🔐 OpenAI API Key", type="password")
#         load_button = gr.Button("📚 Load Document")
#     load_status = gr.Textbox(label="Status")
#
#     load_button.click(fn=process_pdf, inputs=[pdf_file, openai_key_box], outputs=[load_status])
#
#     with gr.Row():
#         question_input = gr.Textbox(label="Ask a Question")
#         ask_button = gr.Button("🔍 Ask")
#
#     answer_output = gr.Textbox(label="Answer")
#     summary_output = gr.Textbox(label="Summary")
#     source_output = gr.Textbox(label="Relevant Chunks", lines=8)
#
#     ask_button.click(fn=ask_question, inputs=[question_input], outputs=[answer_output, summary_output, source_output])
#
#     gr.Markdown("## 🙋‍♂️ Was the answer helpful?")
#
#     with gr.Row():
#         good_btn = gr.Button("👍 Yes")
#         bad_btn = gr.Button("👎 No")
#
#     feedback_status = gr.Textbox(label="Feedback Status")
#
#     good_btn.click(fn=save_feedback, inputs=[question_input, answer_output, gr.Textbox(value="👍")], outputs=[feedback_status])
#     bad_btn.click(fn=save_feedback, inputs=[question_input, answer_output, gr.Textbox(value="👎")], outputs=[feedback_status])
#
# # Launch
# demo.launch()
