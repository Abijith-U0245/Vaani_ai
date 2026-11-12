# Project: The Relay (Vaani ML Engineering take-home)

Production-minded LLM orchestration prototype: multi-stage RAG pipeline over 3 fixed arXiv papers,
rate-limit-aware scheduling, a local stand-in model service, and honest measurement.

## Hard rules
- Python 3.11+, FastAPI, asyncio, Pydantic, PyMuPDF, sentence-transformers, numpy + BM25, pandas, pytest, matplotlib.
- No Redis/Kafka/Kubernetes. No unnecessary dependencies.
- Secrets only via environment variables (GROQ_API_KEY). Never hardcode, log, or commit a key.
- Papers (exact versions only): 1706.03762v7, 2410.00037v1, 2507.07955v2. Verify version + SHA-256.
- Answers come ONLY from retrieved paper text. If the papers don't say, refuse. Never use outside knowledge.
- Never fabricate page numbers, citations, benchmark numbers, or results. Unknown = blank / "not measured".
- Never present an estimate as a measurement. Label every number: documented / observed / estimated / unknown.
- Every LLM call streams. Every call logs provenance: hosted_inference | synthetic_output | replayed_output.
- Real Groq calls go through llm/budget_guard.py with a hard max-calls and max-tokens cap that fails closed.
- Deterministic Python for anything that does not need an LLM.

## results.csv (exact schema, do not change)
request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade
- status: ok | rate_limited | timeout | error
- answer_grade: correct | wrong | refused_correctly | refused_wrongly (blank if no answer)
- source: real | stand-in
- Extra information goes in supplementary jsonl/csv files, never extra columns.

## Latency fields (never conflate)
queue_wait_ms, provider_ttft_ms, provider_gen_ms, verify_ms,
user_ttft_ms (= first_token_ms in the CSV), e2e_ms (= total_ms in the CSV).

## Admission control
- Daily budget (RPD/TPD): reserve worst case per request at admission, persist to disk.
- Per-minute limits (RPM/TPM): reserve per call, reconcile actual usage afterwards.
- Reserve/check must be synchronous (no await inside the critical section) so asyncio makes it atomic.
- Ambiguous network failures are marked "uncertain", not refunded.

## Working style
- Work in small steps. Explain design decisions briefly. Add type hints, structured logging, unit tests.
- When code fails, debug the actual error; do not rewrite blindly.
- If a requirement is ambiguous or my request seems technically wrong, say so before coding.
- Do not generate more than one module group per response.
- Never invent results. If something has not been measured, say "not measured".
