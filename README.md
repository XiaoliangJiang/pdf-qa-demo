---
title: PDF QA Assistant
emoji: 📄
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: "3.50.2"
app_file: app.py
pinned: false
---
## 🧑‍💼 Author

**Xiaoliang Jiang**  
PhD candidate @ UIUC · NLP, IR, Digital Libraries  
Feel free to ⭐ this repo or contact me via GitHub.

# 📄 PDF QA Assistant

A lightweight demo of a RAG (Retrieval-Augmented Generation) system that enables question answering over PDF documents.

- 📤 Upload any PDF file (e.g., textbook, paper, report)
- 💬 Ask natural language questions
- 🔍 Get answers based on document content
- ✨ Highlight the source paragraph
- 📊 Feedback mechanism built-in (optional)

---

## 🔧 Tech Stack

| Component        | Role                                             |
|------------------|--------------------------------------------------|
| `LangChain`      | Orchestrates document loading & retrieval        |
| `FAISS`          | Embedding-based vector search engine             |
| `OpenAI API`     | Powers answer generation (can switch to others)  |
| `Gradio`         | Builds the web interface                         |
| `PyMuPDF`        | Extracts text from uploaded PDFs                 |

---

## 🚀 Live Demo

👉 [Try it on Hugging Face](https://huggingface.co/spaces/xjiang36/pdf-qa-demo)

---

## 🧠 Example Use Case

You can upload any textbook or research article and ask:

> *“What is the main idea of Chapter 3?”*  
> *“Define the difference between supervised and unsupervised learning.”*

The model will return an answer + highlight the most relevant context paragraph.

---

## 🛡️ Security Notes

This Space currently uses Gradio `3.50.2`. You may upgrade if needed by editing:

