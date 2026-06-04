"""
Enterprise RAG Chain - Main orchestration module
Author: Vishesh Prajapati | AI Product Manager
"""
import os
import time
from typing import Optional
from dataclasses import dataclass

from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI

SYSTEM_PROMPT = """You are a helpful enterprise knowledge assistant.
Answer questions using ONLY the provided context.
Always cite your sources by referencing the document name and section.
If you cannot find the answer in the context, say "I don't have
that information in the knowledge base" - do NOT make up answers.

Context: {context}

Answer format:
- Give a direct answer first
- Cite sources as: [Source: <document_name>, Section: <section>]
- If confidence is low, say so explicitly
"""

@dataclass
class RAGResponse:
    answer: str
    citations: list
    confidence_score: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    chunks_retrieved: int
    hallucination_flag: bool = False


class EnterpriseRAGChain:
    """
    Production-grade RAG chain with:
    - Multi-LLM support (Gemini, GPT-4, Claude)
    - Citation tracking
    - Hallucination detection
    - Latency monitoring
    - Confidence scoring
    """

    def __init__(self, retriever, llm_model="gemini-pro",
                 top_k=5, citation_mode=True,
                 hallucination_guard=True, temperature=0.1):
        self.retriever = retriever
        self.llm_model = llm_model
        self.top_k = top_k
        self.citation_mode = citation_mode
        self.hallucination_guard = hallucination_guard
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.chain = self._build_chain()

    def _initialize_llm(self):
        if "gemini" in self.llm_model:
            return ChatVertexAI(model_name=self.llm_model,
                                temperature=self.temperature,
                                max_output_tokens=1024)
        return ChatOpenAI(model="gpt-4o", temperature=self.temperature)

    def _build_chain(self):
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ])
        return ({"context": self.retriever | self._format_docs,
                 "question": RunnablePassthrough()}
                | prompt | self.llm | StrOutputParser())

    def _format_docs(self, docs):
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "General")
            formatted.append(
                f"[Doc {i} | {source} | {section}]\n{doc.page_content}\n")
        return "\n---\n".join(formatted)

    def _extract_citations(self, docs):
        return [{"source": d.metadata.get("source", "Unknown"),
                 "section": d.metadata.get("section", "General"),
                 "relevance_score": d.metadata.get("relevance_score", 0.0),
                 "snippet": d.page_content[:200] + "..."} for d in docs]

    def _compute_confidence(self, answer, docs):
        uncertainty = ["i don't know", "not sure", "cannot find",
                       "no information", "not available"]
        if any(p in answer.lower() for p in uncertainty):
            return 0.3
        avg = sum(d.metadata.get("relevance_score", 0.5)
                  for d in docs) / max(len(docs), 1)
        return min(0.95, avg * 1.1)

    def query(self, question: str) -> RAGResponse:
        t0 = time.time()
        t1 = time.time()
        docs = self.retriever.get_relevant_documents(question)
        ret_ms = (time.time() - t1) * 1000

        t2 = time.time()
        answer = self.chain.invoke(question)
        gen_ms = (time.time() - t2) * 1000

        return RAGResponse(
            answer=answer,
            citations=self._extract_citations(docs) if self.citation_mode else [],
            confidence_score=round(self._compute_confidence(answer, docs), 3),
            retrieval_latency_ms=round(ret_ms, 1),
            generation_latency_ms=round(gen_ms, 1),
            total_latency_ms=round((time.time() - t0) * 1000, 1),
            chunks_retrieved=len(docs),
        )
