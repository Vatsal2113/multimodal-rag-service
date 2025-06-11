#!/usr/bin/env python3
# coding: utf-8
"""
Multimodal RAG pipeline module.

Exports:
  - run_pipeline_on_pdf(pdf_path, workdir=None) -> dict
  - answer_question(question, workdir) -> str
  - load_query_history(workdir) -> list[dict]
"""

import os
import re
import difflib
import sqlite3
import shutil
import tempfile
import json
import datetime
from pathlib import Path
from collections import defaultdict

from PIL import Image
import fitz
import pytesseract
import tiktoken
import google.generativeai as genai

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from transformers import AutoTokenizer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


# ─── Helpers ─────────────────────────────────────────────────────────

# Token counter
_enc = tiktoken.get_encoding("cl100k_base")
def n_tok(txt: str) -> int:
    return len(_enc.encode(txt))

# Roman numerals → int
_ROMAN = {'i':1,'v':5,'x':10,'l':50,'c':100,'d':500,'m':1000}
def roman_to_int(s: str) -> int:
    total = prev = 0
    for ch in reversed(s.lower()):
        val = _ROMAN.get(ch, 0)
        total = total - val if val < prev else total + val
        prev = max(prev, val)
    return total

# Normalize labels (“Figure IIa” → “fig2a”)
LABEL_RE = re.compile(r'^(fig(?:ure)?|table)\s*([IVXLCDM]+|\d+)([a-z])?[\.\s]*[:\-]?', re.I)
def label_key(txt: str) -> str:
    if not txt:
        return None
    s = txt.lower().replace(' ', '')
    m = LABEL_RE.match(s)
    if not m:
        return None
    head, raw, suffix = m.groups()
    num = raw if raw.isdigit() else str(roman_to_int(raw))
    key = ('fig' if head.startswith('fig') else 'table') + num
    return key + suffix if suffix else key

# Detect existing labels in markdown
FIG_TBL = re.compile(r'(fig(?:ure)?\.?\s*\d+[a-z]?|table\s+[IVXLCDM\d]+[a-z]?)', re.I)
def canon(lbl: str) -> str:
    return FIG_TBL.sub(lambda m: re.sub(r'\s+','',m[0].lower()) + ":", lbl)

# Clean OCR/text artifacts
def clean(text: str) -> str:
    t = text.replace('\u00ad','')
    t = re.sub(r'(?<=\w)-\n(?=\w)','', t)
    return re.sub(r'\s*\n\s*',' ', t).strip()

# Pixmap → PIL
def pixmap_to_pil(pix: fitz.Pixmap) -> Image.Image:
    mode = "RGBA" if pix.alpha else "RGB"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)

# OCR an image
def ocr_image(im: Image.Image) -> str:
    return clean(pytesseract.image_to_string(im, lang="eng"))


# ─── 1. PDF → chunks & index ─────────────────────────────────────────

def run_pipeline_on_pdf(pdf_path: str, workdir: str = None) -> dict:
    """
    Run the full multimodal-RAG pipeline on a single PDF.
    Args:
      pdf_path: path to a .pdf file
      workdir:  where to store intermediate DB, images, vector store.
                if None, a temp dir is auto-created.
    Returns:
      dict {
        "texts": [ {chunk_id, source, page, type, content}, … ],
        "tables":[ {chunk_id, source, page, markdown, caption}, … ],
        "images":[ {chunk_id, source, page, ocr, caption, img_path}, … ],
        "store_dir": <path to chroma_store>
      }
    """
    # — Gemini setup
    gem_key = os.getenv("GEMINI_API_KEY", "")
    if not gem_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    genai.configure(api_key=gem_key)
    GEM_MM = genai.GenerativeModel("gemini-1.5-flash-latest")

    # — Prepare workspace
    if workdir:
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
    else:
        wd = Path(tempfile.mkdtemp())
    pdfs_dir    = wd / "pdfs"
    img_out_dir = wd / "pics"
    db_path     = wd / "chunks.db"
    store_dir   = wd / "chroma_store"

    for d in (pdfs_dir, img_out_dir, store_dir):
        d.mkdir(exist_ok=True)

    # copy in the PDF
    pdf_file = Path(pdf_path)
    shutil.copy(pdf_file, pdfs_dir / pdf_file.name)

    # — SQLite registry
    conn = sqlite3.connect(db_path)
    conn.create_function("REGEXP", 2, lambda expr, itm: 1 if itm and re.search(expr, itm) else 0)
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE chunks(
        chunk_id        INTEGER PRIMARY KEY,
        source          TEXT,
        page            INTEGER,
        type            TEXT,
        content         TEXT,
        caption         TEXT,
        img_path        TEXT,
        parent_chunk_id INTEGER,
        label_key       TEXT
      )
    """)
    conn.commit()

    # — Embedding setup
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedder  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # — Docling conversion
    converter = DocumentConverter(format_options={
      InputFormat.PDF: PdfFormatOption(
        pipeline_options=PdfPipelineOptions(images_scale=2.0, generate_picture_images=True)
      )
    })
    doc = converter.convert(str(pdfs_dir / pdf_file.name)).document

    # — Chunk, OCR-fallback, tables, images
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=1024, stride=200)
    cid = 0

    # Text & equations
    for ch in chunker.chunk(doc):
        page = ch.meta.doc_items[0].prov[0].page_no
        txt  = clean(ch.text)
        typ  = ("equation" if len(txt)<300 and re.search(r"(\\frac|\\sum|\\int|=|[∑∫√±×÷])", txt)
                         else "text")
        if typ=="equation" and not re.search(r'\(\s*\d+\s*\)', txt):
            cid += 1
            txt  = f"( {cid} ) {txt}"
        cid += 1
        cur.execute(
          "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
          (cid, pdf_file.stem, page, typ, txt, None, None, None, None)
        )

    # OCR fallback on blank pages
    pdfdoc = fitz.open(str(pdfs_dir / pdf_file.name))
    for p in range(1, len(doc.pages)+1):
        if not any(cur.execute(
              "SELECT 1 FROM chunks WHERE source=? AND page=? AND type='text'",
              (pdf_file.stem, p)
            )):
            pix     = pdfdoc[p-1].get_pixmap(dpi=300)
            pil_img = pixmap_to_pil(pix)
            page_txt= clean(pytesseract.image_to_string(pil_img, lang="eng"))
            cid += 1
            cur.execute(
              "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
              (cid, pdf_file.stem, p, "page", page_txt, None, None, None, None)
            )
    pdfdoc.close()
    conn.commit()

    # Tables + captions
    tbl_no = 0
    for tbl in doc.tables:
        if not tbl.prov:
            continue
        tbl_no += 1
        page = tbl.prov[0].page_no
        md   = canon(tbl.export_to_markdown(doc))
        if not FIG_TBL.search(md):
            md = f"table{tbl_no}:\n{md}"
        prompt  = ["Write a one-sentence caption for this table:", md]
        summary = GEM_MM.generate_content(prompt).text.strip()
        cap     = f"table{tbl_no}: {summary}"
        lk      = label_key(cap)
        cid += 1
        cur.execute(
          "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
          (cid, pdf_file.stem, page, "table", md, cap, None, None, lk)
        )

    # Images + OCR
    pdfdoc = fitz.open(str(pdfs_dir / pdf_file.name))
    fig_no = 0
    for i, pic in enumerate(doc.pictures, start=1):
        if not pic.prov:
            continue
        fig_no += 1
        page = pic.prov[0].page_no
        pil  = pic.get_image(doc)
        out  = img_out_dir / f"fig{fig_no}_p{page}_{i}.png"
        pil.save(out, "PNG")

        ocr_txt = ocr_image(pil)
        lk      = label_key(f"fig{fig_no}:")
        cid += 1
        cur.execute(
          "INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
          (cid, pdf_file.stem, page, "image", ocr_txt, None, str(out), None, lk)
        )
    pdfdoc.close()
    conn.commit()

    # Summarize all images
    cur.execute("SELECT chunk_id, img_path FROM chunks WHERE type='image'")
    for img_cid, img_path in cur.fetchall():
        pil = Image.open(img_path)
        prompt  = ["Write a concise one-sentence summary of this figure:", pil]
        summary = GEM_MM.generate_content(prompt).text.strip()
        cur.execute(
          "UPDATE chunks SET caption=? WHERE chunk_id=?",
          (summary, img_cid)
        )
    conn.commit()

    # Build Chroma vector store
    docs = []
    cur.execute("SELECT chunk_id, source, page, type, content, caption, img_path FROM chunks")
    for cid, src, page, typ, content, caption, img_path in cur.fetchall():
        full = (caption or "") + "\n" + content if typ in ("table","image") else content
        docs.append(Document(full, metadata={
            "chunk_id": cid,
            "source":   src,
            "page":     page,
            "type":     typ
        }))

    vectordb = Chroma(
      persist_directory=str(store_dir),
      collection_name="multimodal_rag",
      embedding_function=embedder
    )
    vectordb.add_documents(docs)
    vectordb.persist()

    # Collect outputs
    texts  = []
    tables = []
    images = []
    cur.execute("SELECT chunk_id, source, page, type, content, caption, img_path FROM chunks")
    for cid, src, page, typ, content, caption, img_path in cur.fetchall():
        if typ in ("text","equation","page"):
            texts.append({
                "chunk_id": cid, "source": src,
                "page": page,  "type": typ,
                "content": content
            })
        elif typ == "table":
            tables.append({
                "chunk_id": cid, "source": src,
                "page": page,
                "markdown": content,
                "caption":  caption
            })
        elif typ == "image":
            images.append({
                "chunk_id": cid, "source": src,
                "page": page,
                "ocr":      content,
                "caption":  caption,
                "img_path": img_path
            })

    conn.close()

    return {
      "texts":     texts,
      "tables":    tables,
      "images":    images,
      "store_dir": str(store_dir)
    }


# ─── 2. QA + History ──────────────────────────────────────────────────

def answer_question(question: str, workdir: str, top_k: int = 5) -> str:
    """
    Load the vector store from workdir/chroma_store, do similarity search,
    ask Gemini to answer, and log the Q&A pair to workdir/query_history.jsonl.
    """
    wd = Path(workdir)
    store_dir = wd / "chroma_store"
    if not store_dir.exists():
        raise FileNotFoundError(f"No vector store at {store_dir!r}")

    # 1) Load the embeddings & index
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory=str(store_dir),
        collection_name="multimodal_rag",
        embedding_function=embedder
    )

    # 2) Retrieve top-k text chunks
    hits = vectordb.similarity_search(question, k=top_k, filter={"type":"text"})
    context = "\n\n".join(h.page_content for h in hits)

    # 3) Call Gemini
    gem_key = os.getenv("GEMINI_API_KEY", "")
    if not gem_key:
        raise RuntimeError("GEMINI_API_KEY not set in environment")
    genai.configure(api_key=gem_key)
    GEM = genai.GenerativeModel("gemini-1.5-flash-latest")

    prompt = [
        "Use the following context to answer the question:\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ]
    answer = GEM.generate_content(prompt).text.strip()

    # 4) Log to history
    hist_file = wd / "query_history.jsonl"
    record = {
      "timestamp": datetime.datetime.now().isoformat(),
      "question":  question,
      "answer":    answer
    }
    with open(hist_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return answer

def load_query_history(workdir: str) -> list:
    """
    Read back the JSONL of past queries+answers from workdir/query_history.jsonl.
    Returns a list of dicts: {timestamp, question, answer}
    """
    hist_file = Path(workdir) / "query_history.jsonl"
    if not hist_file.exists():
        return []
    records = []
    with open(hist_file, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


__all__ = ["run_pipeline_on_pdf", "answer_question", "load_query_history"]
