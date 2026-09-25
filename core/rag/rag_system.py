"""End-to-end RAG pipeline: retrieve -> build prompt -> generate."""
from __future__ import annotations

from typing import Optional

from core.database.knowledge_db import KnowledgeDB
from core.llm.llm_client import LLMClient
from core.rag.prompt_builder import PromptBuilder
from core.rag.retriever import RetrieverBase


class RAGSystem:
    def __init__(self, retriever: RetrieverBase, llm: LLMClient,
                 prompt_builder: Optional[PromptBuilder] = None, top_k: int = 5):
        self.retriever = retriever
        self.llm = llm
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.top_k = top_k

    def retrieve(self, question: str, db: KnowledgeDB) -> list[dict]:
        return db.search(question, self.top_k)

    def answer(self, question: str, db: KnowledgeDB) -> dict:
        """Run full RAG: retrieve Top-K, build prompt, query LLM."""
        topk = self.retrieve(question, db)
        contexts = [d["text"] for d in topk]
        prompt = self.prompt_builder.build(question, contexts)
        response = self.llm.query(prompt)
        return {
            "response": response,
            "topk": topk,
            "contexts": contexts,
            "prompt": prompt,
        }
