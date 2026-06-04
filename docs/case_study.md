# Case Study: Enterprise RAG Knowledge Platform

## Scaling AI Knowledge Access to 470,000+ Store Associates

**Author:** Vishesh Prajapati — AI Product Manager
**Platform:** Google Cloud Vertex AI + Gemini
**Scale:** 2,300+ stores | 470,000+ users

---

## Problem

39% failed query rate. 8.2s response time. 34% hallucination rate. 14,000+ docs scattered across PDF, Confluence, SharePoint. Associates asking natural language questions into keyword-only search.

---

## Solution: Hybrid RAG Platform

### Architecture Decision: Why RAG over Fine-tuning?

- Fine-tuning: expensive, slow to update, still hallucinates
- RAG: always current, citable, grounded in real data ✅

### Key Decision 1: Hybrid Retrieval

BM25(0.3) + Dense(0.7) = 91.3% precision vs 78.9% dense-only. Policy IDs and SKU numbers need exact matching. Hybrid wins.

### Key Decision 2: Semantic Chunking

Fixed-size splits mid-sentence → context dilution → hallucinations. Semantic chunking: faithfulness 0.76 → 0.91.

### Key Decision 3: RAGAS from Day 1

Weekly evaluation cycles. Caught 2 regressions before production. Quality gate: block deployment if faithfulness < 0.88.

---

## Results

| Metric | Before | After |
|--------|--------|-------|
| Faithfulness | 0.71 | 0.94 |
| Context Precision | 0.61 | 0.91 |
| Hallucination Rate | 34% | 4.2% |
| DAU Growth | — | +28% |
| Response Time | 8.2s | 1.4s |

---

## Top 5 Learnings

1. **Chunking > model selection** for RAG quality
2. **Measure hallucination from day 1**, not day 90
3. **Citation tracking = trust = adoption**
4. **Hybrid retrieval beats pure dense** every time
5. **Responsible AI compliance** is a product feature, not afterthought

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-4)
- Document ingestion pipeline (PDF, DOCX, HTML, JSON)
- Baseline dense-only retrieval with ChromaDB
- Initial RAGAS evaluation framework
- Faithfulness baseline: 0.71

### Phase 2: Hybrid Retrieval (Weeks 5-8)
- BM25 + Dense ensemble with EnsembleRetriever
- Cross-encoder reranking with ms-marco-MiniLM
- Context precision: 0.61 → 0.87
- Hallucination rate: 34% → 12%

### Phase 3: Semantic Chunking (Weeks 9-12)
- RecursiveCharacterTextSplitter with semantic separators
- Chunk size optimization: 512 tokens, 64 overlap
- Faithfulness: 0.87 → 0.91
- Hallucination rate: 12% → 6%

### Phase 4: Production Hardening (Weeks 13-16)
- Citation tracking + source attribution
- Hallucination guard with uncertainty detection
- Responsible AI compliance scoring (98%)
- Final hallucination rate: 4.2%

---

## Architecture Decision Records (ADRs)

### ADR-001: ChromaDB vs Pinecone
**Decision:** ChromaDB for development, Pinecone for production
**Rationale:** ChromaDB zero-infra for fast iteration; Pinecone for scale, filtering, and metadata

### ADR-002: Gemini Pro vs GPT-4o
**Decision:** Gemini Pro as primary, GPT-4o as fallback
**Rationale:** Cost optimization + Google Cloud integration; GPT-4o for complex reasoning

### ADR-003: RAGAS vs Custom Evaluation
**Decision:** RAGAS with custom quality gates
**Rationale:** Industry-standard metrics + faithfulness threshold enforcement

---

*Built with real enterprise experience managing AI products at scale.*
*Author: Vishesh Prajapati | AI Product Manager | Chicago, IL*
