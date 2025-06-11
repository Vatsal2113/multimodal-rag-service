# multimodal/__init__.py

"""
Multimodal RAG package.

Exports:
  - run_pipeline_on_pdf: main entry-point for the pipeline
  - chat:    (optional) local CLI chat loop
"""
from .pipeline import run_pipeline_on_pdf, chat

__all__ = ["run_pipeline_on_pdf", "chat"]
