# ============================================================
# ClinicalQuery.py — Medical Document Q&A (RAG Pipeline)
# ============================================================
# What this does in plain English:
#   1. You give it a patient note (a text document)
#   2. It chops the document into small chunks
#   3. It converts each chunk into a list of numbers (an "embedding")
#      that captures the *meaning* of that chunk
#   4. Those numbers get stored in a searchable index (FAISS)
#   5. When you ask a question, your question also gets converted
#      to numbers — then we find the chunks whose numbers are
#      closest (most similar in meaning) to your question
#   6. We hand those relevant chunks + your question to GPT,
#      which reads them and writes you an answer
#
# That whole flow = Retrieval-Augmented Generation (RAG)
# ============================================================

import os
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# ------------------------------------------------------------------
# FAKE PATIENT NOTE
# A realistic but entirely synthetic note — no real patient data.
# ------------------------------------------------------------------
SAMPLE_NOTE = """
PATIENT: John Doe  |  DOB: 03/14/1968  |  MRN: 00482917
DATE OF VISIT: 04/28/2026
ATTENDING: Dr. Sarah Patel, MD — Internal Medicine

CHIEF COMPLAINT:
Patient presents with persistent fatigue, increased thirst, and frequent urination
over the past 6 weeks. Reports blurred vision intermittently.

HISTORY OF PRESENT ILLNESS:
Mr. Doe is a 58-year-old male with a past medical history of hypertension and
hyperlipidemia. He was referred by his primary care physician after routine labs
showed a fasting blood glucose of 312 mg/dL and an HbA1c of 9.4%. He denies
chest pain, shortness of breath, or lower extremity edema. Family history is
significant for Type 2 diabetes in his mother and paternal uncle.

MEDICATIONS:
- Lisinopril 10mg daily (for hypertension)
- Atorvastatin 40mg nightly (for high cholesterol)
- Aspirin 81mg daily

ALLERGIES: Penicillin (rash)

PHYSICAL EXAM:
Vitals: BP 148/92, HR 78, Temp 98.6°F, BMI 31.2
General: Alert and oriented, no acute distress
HEENT: Mild bilateral retinal changes noted on fundoscopic exam
Cardiovascular: Regular rate and rhythm, no murmurs
Extremities: No peripheral edema, intact distal pulses

LABORATORY RESULTS:
- Fasting glucose: 312 mg/dL (normal: 70–99)
- HbA1c: 9.4% (normal: below 5.7%)
- LDL cholesterol: 118 mg/dL
- Creatinine: 1.1 mg/dL (normal)
- eGFR: 74 mL/min (mildly reduced)
- Urine microalbumin: 45 mg/g (elevated — early kidney involvement)

ASSESSMENT AND PLAN:
1. New diagnosis of Type 2 Diabetes Mellitus
   - Start Metformin 500mg twice daily with meals; titrate to 1000mg twice daily
     over 4 weeks as tolerated
   - Refer to diabetes education program
   - Continuous glucose monitoring (CGM) recommended
   - Ophthalmology referral for diabetic retinopathy screening
   - Nephrology referral given elevated microalbumin and reduced eGFR

2. Hypertension — suboptimal control
   - Increase Lisinopril to 20mg daily
   - Target BP below 130/80 per ADA guidelines for diabetic patients

3. Hyperlipidemia
   - Continue Atorvastatin; recheck lipid panel in 3 months

FOLLOW-UP: Return in 4 weeks for repeat labs and medication tolerance check.
Patient instructed on low-carbohydrate diet and 150 min/week aerobic activity.
""".strip()


# ------------------------------------------------------------------
# STEP 1 — BUILD THE KNOWLEDGE BASE (the "index")
# ------------------------------------------------------------------
# This function takes a document (the patient note) and:
#   a) Splits it into overlapping chunks so no sentence gets cut off
#      right at a chunk boundary and lost
#   b) Converts each chunk to an embedding (list of numbers) via OpenAI
#   c) Stores all those embeddings in FAISS, an in-memory search index
#
# Think of FAISS like a library card catalog — except instead of
# searching by title, you search by *meaning*.
# ------------------------------------------------------------------
def build_index(text: str, api_key: str) -> FAISS:
    os.environ["OPENAI_API_KEY"] = api_key

    # Split into chunks of ~500 characters with 100-char overlap
    # Overlap prevents important info from being stranded at a boundary
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])

    # OpenAI converts each chunk into a 1536-dimension vector
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # FAISS stores the vectors and lets us do fast nearest-neighbor search
    index = FAISS.from_documents(docs, embeddings)
    return index


# ------------------------------------------------------------------
# STEP 2 — ANSWER A QUESTION
# ------------------------------------------------------------------
# Given a question and the index we built:
#   a) Embed the question the same way we embedded the chunks
#   b) Find the top 3 chunks whose embeddings are closest to the question
#   c) Build a prompt: "Here is the context: [chunks]. Answer: [question]"
#   d) Send that to GPT-4o-mini and return the answer
# ------------------------------------------------------------------
def answer_question(question: str, index: FAISS, api_key: str) -> str:
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # RetrievalQA wires the retriever + LLM together into one chain
    # retriever fetches the top 3 most relevant chunks (k=3)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=index.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
    )

    result = qa_chain.invoke({"query": question})
    return result["result"]


# ------------------------------------------------------------------
# STEP 3 — THE GRADIO UI
# ------------------------------------------------------------------
# Gradio gives us a browser interface with zero HTML/CSS.
# State (gr.State) holds the FAISS index between interactions —
# once you load a note, it stays in memory for follow-up questions.
# ------------------------------------------------------------------
def load_note(file, use_sample: bool, api_key: str, state):
    if not api_key.strip():
        return "Please enter your OpenAI API key.", state

    if use_sample:
        text = SAMPLE_NOTE
    elif file is not None:
        with open(file.name, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        return "Upload a file or check 'Use sample note'.", state

    try:
        index = build_index(text, api_key)
        return "Note loaded. Ask a question below.", index
    except Exception as e:
        return f"Error building index: {e}", state


def ask(question: str, api_key: str, state):
    if state is None:
        return "Load a note first."
    if not question.strip():
        return "Please type a question."
    try:
        return answer_question(question, state, api_key)
    except Exception as e:
        return f"Error: {e}"


with gr.Blocks(title="ClinicalQuery") as app:
    gr.Markdown("## ClinicalQuery — Medical Document Q&A")
    gr.Markdown(
        "Load a patient note, then ask questions about it. "
        "Answers are grounded only in the document — the model can't hallucinate facts that aren't there."
    )

    index_state = gr.State(None)

    with gr.Row():
        api_key_box = gr.Textbox(label="OpenAI API Key", type="password", placeholder="sk-...")

    with gr.Row():
        file_upload = gr.File(label="Upload patient note (.txt)", file_types=[".txt"])
        use_sample_box = gr.Checkbox(label="Use built-in sample note", value=True)

    load_btn = gr.Button("Load Note")
    load_status = gr.Textbox(label="Status", interactive=False)

    load_btn.click(
        fn=load_note,
        inputs=[file_upload, use_sample_box, api_key_box, index_state],
        outputs=[load_status, index_state],
    )

    gr.Markdown("---")
    question_box = gr.Textbox(label="Your question", placeholder="What medications is the patient on?")
    ask_btn = gr.Button("Ask")
    answer_box = gr.Textbox(label="Answer", lines=6, interactive=False)

    ask_btn.click(
        fn=ask,
        inputs=[question_box, api_key_box, index_state],
        outputs=[answer_box],
    )

if __name__ == "__main__":
    app.launch()
