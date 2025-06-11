#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# updated import:
from multimodal.pipeline import run_pipeline, answer_question, load_query_history

app = FastAPI(title="Multimodal RAG Service")

# enable CORS so browsers can call your API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── POST /extract ───────────────────────────────────────────────────
@app.post("/extract")
async def extract(files: List[UploadFile] = File(...)):
    # must be at least one PDF
    if not files:
        raise HTTPException(status_code=400, detail="Upload 1+ PDF files")

    tmpdir = tempfile.mkdtemp(prefix="mmrag_")
    pdfs_dir = Path(tmpdir) / "pdfs"
    pdfs_dir.mkdir()

    # save each PDF
    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDFs supported")
        dest = pdfs_dir / f.filename
        with open(dest, "wb") as out:
            out.write(await f.read())

    # run full pipeline on every PDF in tmpdir/pdfs
    try:
        result = run_pipeline(tmpdir)
    except Exception as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    # return everything + the workdir so client can call /qa & /history
    result["workdir"] = tmpdir
    return JSONResponse(content=result)


# ─── POST /qa ─────────────────────────────────────────────────────────
@app.post("/qa")
async def qa(
    workdir: str = Form(..., description="workdir returned by /extract"),
    question: str = Form(...)
):
    wd = Path(workdir)
    if not wd.exists():
        raise HTTPException(status_code=400, detail="Invalid workdir")

    try:
        answer = answer_question(question, workdir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"QA error: {e}")

    return {"answer": answer}


# ─── GET /history ─────────────────────────────────────────────────────
@app.get("/history")
async def history(
    workdir: str = Query(..., description="workdir returned by /extract")
):
    wd = Path(workdir)
    if not wd.exists():
        raise HTTPException(status_code=400, detail="Invalid workdir")

    return {"history": load_query_history(workdir)}
