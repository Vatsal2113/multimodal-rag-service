#!/usr/bin/env python3
# coding: utf-8
"""
Multimodal RAG pipeline module, updated to process ALL PDFs
in a given workdir/pdfs directory in one go.
Exports:
  - run_pipeline(workdir)       → processes every PDF under workdir/pdfs
  - answer_question(question, workdir)
  - load_query_history(workdir)
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
    return FIG_TBL.sub(lambda m: re.sub(r'\s+','', m[0].lower()) + ":", lbl)

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


# ─── 1. Process all PDFs in workdir/pdfs ─────────────────────────────

def run_pipeline(workdir: str) -> dict:
    """
    Process every PDF under {workdir}/pdfs and build:
      - SQLite chunks DB
      - Chroma vector store
    Returns a dict:
      {
        "texts": [...],
        "tables": [...],
        "images": [...],
        "store_dir": "<workdir>/chroma_store"
      }
    """
    wd         = Path(workdir)
    pdfs_dir   = wd / "pdfs"
    img_out_dir= wd / "pics"
    db_path    = wd / "chunks.db"
    store_dir  = wd / "chroma_store"

    # sanity
    if not pdfs_dir.exists():
        raise RuntimeError(f"Expected PDFs in {pdfs_dir!r}")

    # 1) Gemini setup
    gem_key = os.getenv("GEMINI_API_KEY", "")
    if not gem_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=gem_key)
    GEM_MM = genai.GenerativeModel("gemini-1.5-flash-latest")

    # 2) Clean / recreate dirs
    for d in (img_out_dir, store_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    # 3) SQLite registry
    if db_path.exists():
        db_path.unlink()
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

    # 4) Embedding setup
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    embedder  = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5) Convert all PDFs via Docling
    converter = DocumentConverter(format_options={
      InputFormat.PDF: PdfFormatOption(
        pipeline_options=PdfPipelineOptions(images_scale=2.0, generate_picture_images=True)
      )
    })
    docling_docs = {}
    for pdf in pdfs_dir.glob("*.pdf"):
        docling_docs[pdf.name] = converter.convert(str(pdf)).document

    # 6) Chunking & Extraction (across all docs)
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=1024, stride=200)
    cid = 0

    for src, doc in docling_docs.items():
        stem    = Path(src).stem.lower()
        img_dir = img_out_dir / stem
        img_dir.mkdir(exist_ok=True)

        # 6.1 Text & Equations
        for ch in chunker.chunk(doc):
            page = ch.meta.doc_items[0].prov[0].page_no
            txt  = clean(ch.text)
            typ  = "equation" if len(txt)<300 and re.search(r"(\\frac|\\sum|\\int|=|[∑∫√±×÷])", txt) else "text"
            if typ=="equation" and not re.search(r'\(\s*\d+\s*\)', txt):
                cid += 1
                txt  = f"( {cid} ) {txt}"
            cid += 1
            cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                        (cid, stem, page, typ, txt, None, None, None, None))

        # 6.2 OCR fallback
        pdfdoc = fitz.open(str(pdfs_dir / src))
        for p in range(1, len(doc.pages)+1):
            if not any(cur.execute(
                  "SELECT 1 FROM chunks WHERE source=? AND page=? AND type='text'",
                  (stem, p)
                )):
                pix     = pdfdoc[p-1].get_pixmap(dpi=300)
                pil     = pixmap_to_pil(pix)
                page_txt= clean(pytesseract.image_to_string(pil, lang="eng"))
                cid += 1
                cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                            (cid, stem, p, "page", page_txt, None, None, None, None))
        pdfdoc.close()
        conn.commit()

        # 6.3 Tables + captions
        tbl_no = 0
        for tbl in doc.tables:
            if not tbl.prov: continue
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
            cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                        (cid, stem, page, "table", md, cap, None, None, lk))

        # 6.4 Images + OCR only
        pdfdoc = fitz.open(str(pdfs_dir / src))
        fig_no = 0
        for i, pic in enumerate(doc.pictures, start=1):
            if not pic.prov: continue
            fig_no += 1
            page = pic.prov[0].page_no
            pil  = pic.get_image(doc)
            fp   = img_dir / f"fig{fig_no}_p{page}_{i}.png"
            pil.save(fp, "PNG")

            ocr_txt = ocr_image(pil)
            lk      = label_key(f"fig{fig_no}:")
            cid    += 1
            cur.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?)",
                        (cid, stem, page, "image", ocr_txt, None, str(fp), None, lk))
        pdfdoc.close()
        conn.commit()

    # 7) Summarize images
    cur.execute("SELECT chunk_id, img_path FROM chunks WHERE type='image'")
    for img_cid, img_path in cur.fetchall():
        pil     = Image.open(img_path)
        prompt  = ["Write a concise one-sentence summary of this figure:", pil]
        summ    = GEM_MM.generate_content(prompt).text.strip()
        cur.execute("UPDATE chunks SET caption=? WHERE chunk_id=?", (summ, img_cid))
    conn.commit()

    # 8) Build Chroma index
    docs = []
    cur.execute("SELECT chunk_id, source, page, type, content, caption, img_path FROM chunks")
    for cid_, src, page, typ, content, caption, ip in cur.fetchall():
        full = (caption or "") + "\n" + content if typ in ("table","image") else content
        docs.append(Document(full, metadata={
          "chunk_id": cid_, "source": src, "page": page, "type": typ
        }))

    vectordb = Chroma(
        persist_directory=str(store_dir),
        collection_name="multimodal_rag",
        embedding_function=embedder
    )
    vectordb.add_documents(docs)
    vectordb.persist()

    # 9) Collect outputs
    texts, tables, images = [], [], []
    cur.execute("SELECT chunk_id, source, page, type, content, caption, img_path FROM chunks")
    for cid_, src, page, typ, cont, cap, ip in cur.fetchall():
        if typ in ("text","equation","page"):
            texts.append({"chunk_id":cid_,"source":src,"page":page,"type":typ,"content":cont})
        elif typ=="table":
            tables.append({"chunk_id":cid_,"source":src,"page":page,"markdown":cont,"caption":cap})
        else:  # image
            images.append({"chunk_id":cid_,"source":src,"page":page,"ocr":cont,"caption":cap,"img_path":ip})

    conn.close()
    return {"texts": texts, "tables": tables, "images": images, "store_dir": str(store_dir)}


# ─── 2. QA + History (unchanged) ────────────────────────────────────

def answer_question(question: str, workdir: str, top_k: int = 5) -> str:
    wd = Path(workdir)
    store_dir = wd / "chroma_store"
    if not store_dir.exists():
        raise FileNotFoundError(f"No vector store at {store_dir!r}")

    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=str(store_dir),
                      collection_name="multimodal_rag",
                      embedding_function=embedder)

    hits = vectordb.similarity_search(question, k=top_k, filter={"type":"text"})
    context = "\n\n".join(h.page_content for h in hits)

    gem_key = os.getenv("GEMINI_API_KEY", "")
    if not gem_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=gem_key)
    GEM = genai.GenerativeModel("gemini-1.5-flash-latest")

    prompt = [
      f"Use the following context to answer the question:\n\n"
      f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    ]
    answer = GEM.generate_content(prompt).text.strip()

    hist_file = wd / "query_history.jsonl"
    record    = {
      "timestamp": datetime.datetime.now().isoformat(),
      "question":  question,
      "answer":    answer
    }
    with open(hist_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return answer


def load_query_history(workdir: str) -> list:
    hist_file = Path(workdir) / "query_history.jsonl"
    if not hist_file.exists():
        return []
    out = []
    for line in open(hist_file, encoding="utf-8"):
        try:
            out.append(json.loads(line))
        except:
            continue
    return out


__all__ = ["run_pipeline", "answer_question", "load_query_history"]
