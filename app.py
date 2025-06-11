#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse

from multimodal.pipeline import (
    run_pipeline_on_pdf,
    answer_question,
    load_query_history,
)

app = FastAPI(title="Multimodal RAG Service")

# ─── POST /extract ────────────────────────────────────────────────────
@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files supported")

    # new temp dir for this job
    tmpdir = tempfile.mkdtemp(prefix="mmrag_")
    in_path = Path(tmpdir) / file.filename
    with open(in_path, "wb") as f:
        f.write(await file.read())

    # run extraction
    try:
        out = run_pipeline_on_pdf(str(in_path), workdir=tmpdir)
    except Exception as e:
        # cleanup on error
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise HTTPException(500, f"Pipeline error: {e}")

    # include the workdir so clients can ask /qa and /history
    out["workdir"] = tmpdir
    return JSONResponse(out)


# ─── POST /qa ─────────────────────────────────────────────────────────
@app.post("/qa")
async def qa(
    workdir: str = Form(..., description="The workdir returned by /extract"),
    question: str = Form(...),
):
    wd = Path(workdir)
    if not wd.exists():
        raise HTTPException(400, f"Workdir not found: {workdir}")

    try:
        answer = answer_question(question, workdir)
    except Exception as e:
        raise HTTPException(500, f"QA error: {e}")

    return {"answer": answer}


# ─── GET /history ─────────────────────────────────────────────────────
@app.get("/history")
async def history(
    workdir: str = Query(..., description="The workdir returned by /extract")
):
    wd = Path(workdir)
    if not wd.exists():
        raise HTTPException(400, f"Workdir not found: {workdir}")
    history = load_query_history(workdir)
    return {"history": history}
