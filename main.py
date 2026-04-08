from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import openai
import PyPDF2
import docx
import io
import json

app = FastAPI(title="Legal Judgment Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = ""

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_text_from_docx(file_bytes: bytes) -> str:
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

EXTRACTIVE_SYSTEM_PROMPT = """You are an expert legal analyst summarizing Indian court judgments. You work in two steps:

STEP 1 — EXTRACTION: Internally extract the most important sentences from each rhetorical section of the document. Do not output this step.

STEP 2 — PARAPHRASE & REWRITE: Rewrite every extracted sentence in clear third-person summarizer voice. Do not use first-person ("the court said that I...") or copy sentences verbatim. Write as an objective legal analyst describing what each party said, what the court found, and what was decided. Do NOT include any page numbers, paragraph numbers, or references like "p.1", "para 2", etc.

Extract and rewrite a good number of sentences per section — enough to give a clear and complete picture — but avoid redundancy. Keep the total response well-structured and readable.

Structure your summary using EXACTLY these rhetorical role sections:

## Preamble
Describe the court, case number, names of parties, judges, and lawyers. Include headnotes or act references if present.

## Facts
Describe the chronology of events that led to the case, how it evolved through the legal system, depositions, proceedings, and lower court history — all in third person.

## Ruling by Lower Court
Describe the verdict, analysis, and reasoning given by the lower court that prompted the present appeal, written from the summarizer's perspective.

## Issues
State the key legal questions framed by the court that needed to be decided, written as the summarizer reporting what issues were framed.

## Arguments by Petitioner
Describe in third person what arguments, contentions, and precedents the petitioner's counsel put forward.

## Arguments by Respondent
Describe in third person what arguments, contentions, and precedents the respondent's counsel put forward.

## Analysis
Describe the court's discussion on evidence, facts, applicable statutes, and prior cases. Convey the court's observations and reasoning from the summarizer's perspective.

## Statutes Referenced
Describe which Acts, Sections, Articles, Rules, or legal provisions the court discussed or quoted, and what they provide.

## Precedents Relied Upon
Describe which prior case decisions the court relied upon and briefly what principle from those cases was applied.

## Precedents Not Relied Upon
Describe which prior case decisions the court did not rely upon and why they were found inapplicable.

## Ratio of the Decision
Describe the main legal reasoning and rationale the court applied just before delivering the final decision.

## Ruling by Present Court
Describe the final decision, conclusion, and orders passed by the present court in clear third-person language.

STRICT RULES:
- Write entirely in third person summarizer voice — never copy sentences verbatim from the document
- Do NOT use quotation marks since sentences are paraphrased, not quoted
- Do NOT include any page numbers, paragraph numbers, or location references
- Keep each section substantive — enough sentences to be clear and complete
- If a section is not present in the document, write: _Not mentioned in the document._
- Maintain a neutral, professional, and objective legal tone throughout"""

async def stream_summary(text: str):
    client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    max_chars = 80000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Document truncated at 80,000 characters...]"

    try:
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EXTRACTIVE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Provide an extractive summary of this legal judgment:\n\n{text}"},
            ],
            stream=True,
            max_tokens=4000,
            temperature=0.1,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield f"data: {json.dumps({'content': delta.content})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    except openai.AuthenticationError:
        yield f"data: {json.dumps({'error': 'Authentication failed. Check the API key in the backend.'})}\n\n"
    except openai.RateLimitError as e:
        print(f"Rate limit detail: {e}")  # add this
        yield f"data: {json.dumps({'error': 'Rate limit exceeded. Please try again later.'})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

@app.post("/summarize")
async def summarize_judgment(file: UploadFile = File(...)):
    file_bytes = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif filename.endswith(".docx"):
            text = extract_text_from_docx(file_bytes)
        elif filename.endswith(".txt"):
            text = extract_text_from_txt(file_bytes)
        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use PDF, DOCX, or TXT.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Text extraction failed: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=422, detail="No text could be extracted from the document.")

    return StreamingResponse(
        stream_summary(text),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

