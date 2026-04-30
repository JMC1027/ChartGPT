# ClinicalQuery

A medical document Q&A app built on a RAG (Retrieval-Augmented Generation) pipeline. Upload a patient note and ask natural language questions about it, answers are grounded strictly in the document.

## How it works

1. A patient note is split into small overlapping chunks
2. Each chunk is converted into an embedding (a list of numbers representing its meaning) via OpenAI
3. Those embeddings are stored in a FAISS vector index
4. When you ask a question, it gets embedded the same way and the most semantically similar chunks are retrieved
5. The retrieved chunks and your question are sent to GPT-4o-mini, which answers using only that context

## Stack

- [LangChain](https://www.langchain.com/) — document splitting and chain orchestration
- [OpenAI API](https://platform.openai.com/) — embeddings and chat completions
- [FAISS](https://faiss.ai/) — in-memory vector similarity search
- [Gradio](https://www.gradio.app/) — browser UI

## Setup

1. Clone the repo
   ```bash
   git clone https://github.com/JMC1027/Clinical-Query.git
   cd Clinical-Query
Install dependencies


pip install -r requirements.txt
Run the app


python ClinicalQuery.py
Open http://127.0.0.1:7860 in your browser

Usage
Paste your OpenAI API key into the key field
Use the built-in sample patient note or upload your own .txt file
Click Load Note, then ask questions in plain English
Example questions
What medications is the patient on?
Does the patient have any allergies?
What were the lab results?
What is the follow-up plan?
