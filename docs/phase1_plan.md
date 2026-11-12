# The Relay: Phase 1 plan (revised architecture)

## Goal
Small, working, measurable LLM pipeline answering questions ONLY from three arXiv papers, with
rate-limit-aware admission control, a calibrated stand-in model service, and honest results.
Decisions must follow from measurements and the report must state what each choice costs.

## Papers (exact versions)
1. Attention Is All You Need, arXiv 1706.03762 v7
2. Moshi: a speech-text foundation model for real-time dialogue, arXiv 2410.00037 v1
3. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling, arXiv 2507.07955 v2

## Pipeline (6 stages, 1 branch, 1 bounded loop, 1 real tool)
1. Understand (LLM, small model, JSON out): question type, paper hints, sub-queries.
2. Retrieve (deterministic): hybrid dense + BM25, optional reranker, paper filter.
3. Sufficiency gate (deterministic, the BRANCH): clearly absent -> refuse early with no more LLM spend;
   borderline -> tool step; enough -> answer.
4. Tool step (the BOUNDED LOOP, max 2 rounds): whitelisted tools search_paper, expand_neighbors, calc.
   Tools genuinely execute in Python with asyncio.wait_for timeout and logged results. calc uses a safe
   AST evaluator, never eval.
5. Answer (LLM, larger model): claims cite chunk IDs only.
6. Verify (deterministic first: IDs exist, quotes are substrings, numbers appear in cited chunk; LLM
   entailment only if flagged). Failure -> regenerate once, else refuse. Never send unverified.

## Control plane (single process)
- BudgetLedger (daily RPD/TPD): reserve worst case at admission + deadline-feasibility check; persisted;
  settle with actual usage.
- WindowLimiter (per-minute RPM/TPM token buckets): reserve per call, reconcile after.
- Reservation states: reserved -> in_flight -> settled | refunded | uncertain.
- Bounded priority queue, per-request deadlines, concurrency semaphore.
- Retries: exponential backoff + jitter, honour retry-after, deadline-aware, global retry budget,
  no retry on 400/401/403/404/413, daily exhaustion is non-retryable.
- Never silently downgrade quality: shrink retrieval only where the gate still passes, otherwise return
  an explicit "temporarily unavailable" reply.

## Latency definitions (never conflate)
queue_wait_ms | provider_ttft_ms | provider_gen_ms | verify_ms |
user_ttft_ms (first token of the VERIFIED answer = first_token_ms) | e2e_ms (= total_ms).
unaccounted_ms is logged so wait + work + retry sums to total.

## Provenance per call (calls.jsonl)
hosted_inference | synthetic_output | replayed_output. Stand-in rows have blank answer_grade.

## Evaluation (rubric frozen in eval/RUBRIC.md BEFORE any run)
Separate: retrieval relevance (hit@k, precision@k, MRR on gold spans), evidence coverage (required facts
in retrieved text), factual correctness, citation correctness (overlap with gold evidence + page),
correct refusal (structured field). Gold evidence comes from extracted PDF text, not memory.
Offline screen of ~6 configs with no LLM; hosted pass on the best 3 only.

## Real Groq budget
Gate 0: read docs + 2 tiny requests to read headers. 4a: ~20 requests, then STOP and inspect.
Later: ~75 calls retrieval eval (Understand cached), ~48 real steady, ~24 real burst.
Never exceed 50% of a day's measured limit. All figures are estimates until measured.

## Stand-in scenarios (seeded)
A steady 300, B burst 300, C overload 600 (retries on and off), D per-minute halved 300,
E daily budget exhausted 300. Latency from measured empirical distributions, not just means.
Tail and load-dependent slowdown are labelled assumed; run at none/mild/strong slowdown.

## Folder structure
relay/
  README.md, pyproject.toml, Makefile, .env.example
  src/relay/{config.py, ingest/, retrieval/, pipeline/, tools/, llm/, control/, telemetry/, standin/, api/}
  eval/{RUBRIC.md, questions.yaml, offline_retrieval.py, grade.py, failure_demos.py}
  exp/{probe_groq.py, loadgen.py, scenarios/}
  analysis/{export_results.py, validate_csv.py, tables.py, plots.py}
  tests/
  data/{pdfs/, processed/, real/, sim/}
  results/  report/  conversation/  docs/

## Phase 2 scope
Download + verify PDFs, extract with PyMuPDF, chunk with offsets, hybrid index, citations,
questions + gold evidence, frozen rubric, offline retrieval screen, tests.
