# Prompts to paste into Copilot Chat (Agent mode), one at a time

Before starting: put copilot-instructions.md at .github/copilot-instructions.md, and put phase1_plan.md
and the assignment PDF in a docs/ folder. Run each prompt, test it, paste real errors back, then move on.

---

## PROMPT 2.1 (setup + download + extract + chunk)

Read docs/phase1_plan.md and docs/assignment.pdf. Do ONLY this step of Phase 2.

1. Create the folder structure from the plan (empty packages with __init__.py where needed),
   pyproject.toml with pinned dependencies, .env.example (no real key), .gitignore (ignore .env, data/processed).
2. Implement src/relay/ingest/download.py: fetch the exact arXiv versions (1706.03762v7, 2410.00037v1,
   2507.07955v2), verify the version via the arXiv API metadata, compute SHA-256, save the original PDFs to
   data/pdfs/ and write data/pdfs/manifest.json (URL, version, SHA-256, retrieval date).
3. Implement src/relay/ingest/extract.py with PyMuPDF: per-page text with the PDF page index, detected
   section headings, character offsets. Handle two-column ordering sensibly.
4. Implement src/relay/ingest/chunk.py: configurable chunk size and overlap; each chunk keeps paper id,
   page, section, and character offsets so citations can be resolved back to the source.
5. Write pytest tests: version assertion, chunk offset round-trip (chunk text equals source slice),
   deterministic output.

Before writing code, list your assumptions and anything in my request you think is technically wrong.
Stop when done and tell me exactly how to run and test it. Do not start retrieval yet.

---

## PROMPT 2.2 (retrieval + citations)

Continue Phase 2. Implement only retrieval and citations.

1. src/relay/retrieval/index.py: embeddings with a small pinned sentence-transformers model, numpy brute
   force (no FAISS), cached to data/processed.
2. src/relay/retrieval/hybrid.py: BM25 + dense with score fusion, optional paper filter, optional
   cross-encoder rerank switch.
3. src/relay/retrieval/citation.py: citation object (paper, page, section, chunk_id, char span) and a
   resolver that checks the chunk exists and a quoted span is a substring of the chunk.
4. Tests for citation resolution and deterministic indexing.

List assumptions first. Stop when done and tell me how to test.

---

## PROMPT 2.3 (questions + frozen rubric + offline screen)

Continue Phase 2. Do not write any LLM-calling code.

1. Write eval/RUBRIC.md defining, BEFORE any run: retrieval relevance (hit@k, precision@k, MRR on gold
   spans), evidence coverage, factual correctness, citation correctness, correct refusal, and the mapping
   to answer_grade (correct / wrong / refused_correctly / refused_wrongly).
2. Create eval/questions.yaml with 18 questions: direct factual, multi-passage, cross-paper, numerical,
   misleading terminology, and unanswerable. Each has: reference answer, required facts, forbidden facts,
   gold evidence as paper + page + character span taken from the extracted PDF text (never from memory).
   Leave 4 slots marked TODO_USER for questions I will write myself.
3. eval/offline_retrieval.py: evaluate ~6 configurations (chunk size, overlap, k, rerank on/off) on the
   metrics above with no LLM calls, and write results/offline_report.csv including tokens per answer.

Tell me which gold spans I must verify by hand.
