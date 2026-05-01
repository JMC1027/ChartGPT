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
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

BASE_PROMPT = PromptTemplate.from_template(
    "You are a clinical assistant reviewing a patient note. "
    "Answer using ONLY information from the context. "
    "Format your response using markdown: "
    "use bullet points (–) for lists, **bold** key clinical terms and drug names, "
    "and keep answers concise and precise. Never invent information.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

REDACT_PROMPT = PromptTemplate.from_template(
    "You are a clinical assistant reviewing a de-identified patient note. "
    "Answer using ONLY information from the context. "
    "Format your response using markdown: bullet points (–) for lists, **bold** key terms. "
    "Only state what is explicitly available. "
    "If information is not in the context, silently omit it — never say it is missing or redacted.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)

# ------------------------------------------------------------------
# DE-IDENTIFICATION FUNCTION
# ------------------------------------------------------------------
# Automatically removes or replaces PHI/PII elements to comply with
# OpenAI's terms of service and privacy regulations.
# ------------------------------------------------------------------
def deidentify_text(text: str) -> str:
    # Replace patient names (assuming format: PATIENT: Name)
    text = re.sub(r'PATIENT:\s*[A-Za-z\s]+', 'PATIENT: [REDACTED NAME]', text)
    # Replace DOB
    text = re.sub(r'DOB:\s*\d{2}/\d{2}/\d{4}', 'DOB: [REDACTED DATE]', text)
    # Replace MRN
    text = re.sub(r'MRN:\s*\d+', 'MRN: [REDACTED ID]', text)
    # Replace dates in various formats
    text = re.sub(r'\d{2}/\d{2}/\d{4}', '[REDACTED DATE]', text)
    # Replace phone numbers (basic)
    text = re.sub(r'\(\d{3}\)\s*\d{3}-\d{4}', '[REDACTED PHONE]', text)
    # Replace email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED EMAIL]', text)
    # Replace SSNs (basic)
    text = re.sub(r'\d{3}-\d{2}-\d{4}', '[REDACTED SSN]', text)
    return text


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
def build_index(text: str) -> FAISS:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Split into chunks of 500 characters with 100-char overlap
    # Overlap prevents important info from being stranded at a boundary
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.create_documents([text])

    # OpenAI converts each chunk into a 1536-dimension vector
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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
def answer_question(question: str, index: FAISS, redact_phi: bool = False) -> str:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain_kwargs = {"prompt": REDACT_PROMPT} if redact_phi else {"prompt": BASE_PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=index.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False,
        chain_type_kwargs=chain_kwargs,
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
def _process_text(raw: str, redact_phi: bool):
    text = deidentify_text(raw) if redact_phi else raw
    return build_index(text)


def _status(html): return f'<div class="note-status">{html}</div>'

def load_note(file, redact_phi: bool, state, raw_text):
    filepath = file if isinstance(file, str) else getattr(file, "name", None)
    if filepath is None:
        return _status('<span class="ns-empty">No file received.</span>'), state, raw_text
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        new_state = _process_text(raw, redact_phi)
        fname = os.path.basename(filepath)
        return _status(f'<span class="ns-dot"></span><span class="ns-name">{fname}</span><span class="ns-ready">Ready</span>'), new_state, raw
    except Exception as e:
        return _status(f'<span class="ns-error">⚠ {e}</span>'), state, raw_text


PHI_TRIGGERS = {
    "name", "dob", "date of birth", "birth date", "birthday",
    "mrn", "medical record", "record number", "patient id",
    "ssn", "social security", "address", "phone", "email",
    "zip", "zipcode", "contact info",
}


def toggle_redact(is_on: bool, raw_text, idx_state):
    new_val = not is_on
    btn_update = gr.update()  # visual handled by JS
    if raw_text is None:
        return new_val, btn_update, "", idx_state
    try:
        new_idx = _process_text(raw_text, new_val)
        msg = "**PHI/PII mode on** — patient identifiers excluded from answers." if new_val else "*PHI/PII mode off.*"
        return new_val, btn_update, msg, new_idx
    except Exception as e:
        return is_on, gr.update(), f"Error rebuilding index: {e}", idx_state


def ask(question: str, state, redact_phi: bool):
    yield '<span style="color:#ffffff;font-style:italic;">Generating answer…</span>'
    if state is None:
        yield "Load a note first."; return
    if not question.strip():
        yield "Please type a question."; return
    try:
        answer = answer_question(question, state, redact_phi=redact_phi)
        if redact_phi and any(t in question.lower() for t in PHI_TRIGGERS):
            yield (
                "> **PHI/PII withheld at your request.** "
                "Non-identifying information is shown below.\n\n"
                + answer
            )
        else:
            yield answer
    except Exception as e:
        yield f"Error: {e}"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

@keyframes shimmer {
    0%   { background-position: -400% center; }
    100% { background-position:  400% center; }
}
@keyframes pulse-dot {
    0%, 100% { box-shadow: 0 0 0 0 rgba(61,255,160,0.6); }
    50%       { box-shadow: 0 0 0 5px rgba(61,255,160,0); }
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Base reset ── */
body,
.gradio-container,
.gradio-container *:not(button):not(textarea):not(input):not(span):not(p):not(h1):not(h2):not(h3):not(a):not(label):not(svg):not(path):not(li):not(ul):not(ol):not(blockquote):not(strong):not(em):not(code) {
    background-color: #000000 !important;
}
body, .gradio-container {
    color: #ffffff !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer, .footer { display: none !important; }

/* ── Outer wrapper ── */
#app-root {
    min-height: 100vh;
    display: flex;
    align-items: stretch;
    padding: 36px 40px;
    gap: 40px;
    box-sizing: border-box;
}

/* ── Left panel ── */
#left-panel {
    flex: 0 0 36%;
    display: flex;
    flex-direction: column;
    padding: 8px 0;
    gap: 0;
}
#brand-block { margin-bottom: 32px; }
#brand-block h1 {
    font-size: 2.8rem; font-weight: 900; line-height: 1.05;
    color: #ffffff; margin: 0 0 10px;
    letter-spacing: -0.5px;
}
#brand-block h1 .accent {
    background: linear-gradient(90deg, #3dffa0, #a0ffdb, #3dffa0);
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite;
}
#brand-block .tagline {
    font-size: 1rem; color: #666666; line-height: 1.5; margin: 0 0 4px;
}
#brand-block .sub-tagline {
    font-size: 0.78rem; color: #3dffa0; letter-spacing: 0.04em;
    text-transform: uppercase; margin: 0;
}

.feature-row { display: flex; align-items: flex-start; gap: 14px; margin-bottom: 18px; }
.feature-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #3dffa0; margin-top: 6px; flex-shrink: 0;
}
.feature-row h3 { font-size: 0.92rem; font-weight: 700; color: #ffffff; margin: 0 0 3px; }
.feature-row p  { font-size: 0.8rem; color: #555555; margin: 0; }

.panel-divider { border: none; border-top: 1px solid #141414; margin: 24px 0; }

#tech-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
    background: #0a0a0a !important;
    border: 1px solid #1e1e1e;
    border-radius: 20px;
    padding: 5px 12px;
    font-size: 0.75rem; color: #555555; font-weight: 600;
}
.disclaimer {
    font-size: 0.72rem; color: #333333; line-height: 1.5; margin-top: 20px;
}

/* ── Right panel ── */
#right-panel { flex: 1; display: flex; flex-direction: column; gap: 12px; }
.card {
    background: #080808 !important;
    border: 1px solid #181818 !important;
    border-radius: 16px !important;
    padding: 18px 22px !important;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    border-color: #1e3326 !important;
    box-shadow: 0 0 28px rgba(61,255,160,0.04) !important;
}

/* ── Note status ── */
.note-status { display: flex; align-items: center; gap: 10px; padding: 2px 0; min-height: 24px; }
.ns-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #3dffa0; flex-shrink: 0;
    animation: pulse-dot 2s ease-in-out infinite;
}
.ns-name { font-size: 0.85rem; font-weight: 600; color: #ffffff; }
.ns-ready {
    font-size: 0.72rem; color: #3dffa0; background: rgba(61,255,160,0.08);
    border: 1px solid rgba(61,255,160,0.2);
    border-radius: 10px; padding: 2px 8px; font-weight: 600;
}
.ns-empty { font-size: 0.82rem; color: #333333; }
.ns-error { font-size: 0.82rem; color: #ff6b6b; }

/* ── Labels ── */
label, .label-wrap span, .block label span, span.svelte-1gfkn6j,
.block > label > span, [class*="label"] { color: #555555 !important; font-size: 0.78rem !important; }

/* ── Inputs ── */
textarea, input[type="text"], input[type="password"] {
    background: #040404 !important;
    border: 1px solid #1e1e1e !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus, input:focus {
    border-color: #3dffa0 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(61,255,160,0.08) !important;
}

/* ── Example chips ── */
.chip button {
    background: transparent !important;
    border: 1px solid #1e1e1e !important;
    color: #555555 !important;
    font-size: 0.78rem !important;
    border-radius: 20px !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}
.chip button:hover {
    border-color: #3dffa0 !important;
    color: #3dffa0 !important;
    background: rgba(61,255,160,0.05) !important;
}

/* ── Buttons (base) ── */
button {
    background: #0d0d0d !important;
    border: 1px solid #1e1e1e !important;
    color: #888888 !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
#ask-btn button {
    border-color: #3dffa0 !important;
    color: #3dffa0 !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
}
#ask-btn button:hover {
    background: rgba(61,255,160,0.08) !important;
    box-shadow: 0 0 16px rgba(61,255,160,0.15) !important;
}
#load-btn button { border-color: #1e1e1e !important; color: #666666 !important; }
#load-btn button:hover { border-color: #3dffa0 !important; color: #3dffa0 !important; }
#redact-btn button { border-color: #1e1e1e !important; color: #666666 !important; font-weight: 600 !important; }
#redact-btn button:hover { border-color: #888888 !important; color: #ffffff !important; }
#redact-btn button.phi-active {
    border: 2px solid #ffffff !important;
    color: #ffffff !important;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.4) !important;
    background: #161616 !important;
}

/* ── Markdown answer ── */
#answer-box { animation: fadeIn 0.3s ease; }
#answer-box .prose, #answer-box p, #answer-box li, #answer-box ul {
    color: #e0e0e0 !important;
    font-size: 0.93rem !important;
    line-height: 1.75 !important;
}
#answer-box strong { color: #3dffa0 !important; }
#answer-box ul { padding-left: 1.2em !important; margin: 6px 0 !important; }
#answer-box li { margin-bottom: 5px !important; }
#answer-box blockquote {
    border-left: 3px solid #3dffa0 !important;
    padding-left: 12px !important;
    color: #888888 !important;
    font-size: 0.85rem !important;
    margin: 0 0 12px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #000; }
::-webkit-scrollbar-thumb { background: #1e1e1e; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #3dffa0; }
"""

with gr.Blocks(title="ClinicalQuery — Medical Document Q&A", theme=gr.themes.Base(), css=CSS) as app:

    index_state    = gr.State(None)
    raw_text_state = gr.State(None)
    redact_state   = gr.State(False)

    with gr.Row(elem_id="app-root"):

        # ── LEFT PANEL ──────────────────────────────────────────────
        with gr.Column(elem_id="left-panel", scale=4):
            gr.HTML("""
            <div id="brand-block">
              <h1>Clinical<span class="accent">Query</span></h1>
              <p class="tagline">AI-powered patient note analysis.<br>Accurate. Private. Instant.</p>
              <p class="sub-tagline">Grounded in your documents. Never fabricated.</p>
            </div>

            <div class="feature-row">
              <div class="feature-dot"></div>
              <div><h3>Load Patient Note</h3><p>Plain-text clinical documents (.txt)</p></div>
            </div>
            <div class="feature-row">
              <div class="feature-dot"></div>
              <div><h3>Ask Clinical Questions</h3><p>Medications, Dx, labs, allergies, history</p></div>
            </div>
            <div class="feature-row">
              <div class="feature-dot"></div>
              <div><h3>Receive Structured Answers</h3><p>Formatted, precise, source-grounded</p></div>
            </div>

            <hr class="panel-divider"/>

            <div id="tech-badges">
              <span class="badge">Python</span>
              <span class="badge">GPT-4o mini</span>
              <span class="badge">LangChain</span>
              <span class="badge">Gradio</span>
            </div>
            <p class="disclaimer">
              Answers are based solely on the uploaded document.<br>
              Always verify clinical decisions with source material.
            </p>
            """)

        # ── RIGHT PANEL ─────────────────────────────────────────────
        with gr.Column(elem_id="right-panel", scale=6):

            # Card 1: Load note + status
            with gr.Row(elem_classes=["card"]):
                note_status = gr.HTML(
                    '<div class="note-status"><span class="ns-empty">No note loaded</span></div>',
                    scale=4,
                )
                load_btn = gr.UploadButton(
                    "Load Note",
                    file_types=[".txt"],
                    scale=1,
                    elem_id="load-btn",
                )

            # Card 2: Question + chips + action buttons
            with gr.Row(elem_classes=["card"]):
                with gr.Column(scale=1):
                    question_box = gr.Textbox(
                        label="What do you want to know?",
                        placeholder="e.g., What medications is the patient on?",
                        lines=3,
                    )
                    with gr.Row():
                        chip1 = gr.Button("What medications is the patient on?", elem_classes=["chip"])
                        chip2 = gr.Button("What is the primary diagnosis?",      elem_classes=["chip"])
                        chip3 = gr.Button("Summarize the patient history",        elem_classes=["chip"])
                    with gr.Row():
                        redact_btn = gr.Button("No PHI/PII", scale=1, elem_id="redact-btn")
                        ask_btn    = gr.Button("Ask",         scale=3, elem_id="ask-btn")

            # Card 3: Formatted answer
            with gr.Row(elem_classes=["card"]):
                answer_box = gr.Markdown(
                    value="",
                    elem_id="answer-box",
                )

    # ── Event wiring ────────────────────────────────────────────────
    load_btn.upload(
        fn=load_note,
        inputs=[load_btn, redact_state, index_state, raw_text_state],
        outputs=[note_status, index_state, raw_text_state],
    )
    redact_btn.click(
        fn=toggle_redact,
        inputs=[redact_state, raw_text_state, index_state],
        outputs=[redact_state, redact_btn, answer_box, index_state],
    )
    ask_btn.click(
        fn=ask,
        inputs=[question_box, index_state, redact_state],
        outputs=[answer_box],
    )
    chip1.click(lambda: "What medications is the patient on?",  outputs=[question_box])
    chip2.click(lambda: "What is the primary diagnosis?",       outputs=[question_box])
    chip3.click(lambda: "Summarize the patient history",        outputs=[question_box])

    app.load(fn=None, js="""
    () => {
        function attachToggle() {
            const btn = document.querySelector('#redact-btn button');
            if (!btn) { setTimeout(attachToggle, 300); return; }
            if (btn._phiListenerAttached) return;
            btn._phiListenerAttached = true;
            btn.addEventListener('click', () => btn.classList.toggle('phi-active'));
        }
        attachToggle();
        const observer = new MutationObserver(() => attachToggle());
        observer.observe(document.body, { childList: true, subtree: true });
    }
    """)

if __name__ == "__main__":
    app.launch()
