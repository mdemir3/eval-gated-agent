# Eval-Gated Agent

[![evals](https://github.com/mdemir3/eval-gated-agent/actions/workflows/evals.yml/badge.svg?branch=main)](https://github.com/mdemir3/eval-gated-agent/actions/workflows/evals.yml)

A small agent that only answers from its tools, plus the eval
harness that gates every change to it in CI.

The agent is a LangGraph state machine: **Planner → Researcher → Verifier**, with a
bounded retry loop and a **Fallback** that refuses rather than guesses. Its tools
(`rag_search` over a local ChromaDB corpus and `mock_api_get` over a deterministic
fake REST API) are served over **MCP** from a separate process. A 20-case golden
dataset is scored by a **deterministic suite** (no LLM, no key) that blocks merges,
and an optional **LLM-judge suite** (DeepEval) that runs only when a hosted judge
key is configured.

Everything runs locally with Ollama. No API key is needed for the agent, the tests
or the gate.

```
                 validated plan            draft + evidence
  question ─▶ Planner ───────────▶ Researcher ─────────────▶ Verifier ──PASS──▶ answer + sources
                                     ▲    │                     │
                                     │    │ MCP over stdio      ├─FAIL, retries < 2 ─▶ back to Researcher
                                     │    ▼                     │   (feedback + wider search)
                                     │  MCP tool server         └─FAIL, retries = 2 ─▶ Fallback:
                                     │   rag_search   (ChromaDB + all-MiniLM-L6-v2)    "I couldn't verify an answer
                                     │   mock_api_get (fake users/orders/weather/…)      from the available sources."
                                     └── each failed call retried once; a failed API call falls back to rag_search
```

## Quickstart

Requires Python 3.11 and, for the default config, [Ollama](https://ollama.com).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                    # pick LLM_PROVIDER / LLM_MODEL
ollama pull llama3.2:3b                 # if LLM_PROVIDER=ollama

python -m evals.build_index --check     # embed corpus/ into .chroma (downloads the 90 MB embedding model once)
python -m agent.main "Which city does user 3 live in, and which river is that city on?"
pytest                                  # 119 unit tests, no LLM needed
python -m evals.run                     # deterministic eval suite; exits 1 on any failure
python -m evals.run --suite all         # + LLM judge, if a hosted judge key is set
```

On Linux CI, install CPU-only torch first to avoid the multi-GB CUDA build (see the workflow).
On macOS, keep the virtualenv outside iCloud- or Dropbox-synced folders.

### Configuration

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `ollama`, `openai` or `anthropic` (required) |
| `LLM_MODEL` | model for that provider, e.g. `llama3.2:3b` (required; no default in code) |
| `LLM_TEMPERATURE`, `LLM_SEED` | default `0` and `42` |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OLLAMA_BASE_URL` | provider credentials / URL |
| `JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_THRESHOLD` | optional judge; provider/model fall back to `LLM_*`, threshold defaults to `0.7` |
| `CHROMA_DIR` | index location, default `.chroma` |

The tool server subprocess gets only the SDK's safe default environment plus a few
cache/index variables. It never sees the LLM API keys.

## Example

Real output (`llama3.2:3b`). The tracking endpoint fails on its first call by design:

```
$ python -m agent.main "According to its tracking, which city is order 1001 in right now, and which river flows through that city?" --trace
Answer: Paris; the river Seine.
Sources: api:/tracking/1001, capital-france

Plan:
  1. mock_api_get(endpoint='/tracking/1001')  - find the current city of order 1001
  2. rag_search(query='According to its tracking, … that city? {{1.*}}', top_k=3)  - find the river that flows through that city
Tools used:
  - mock_api_get(endpoint='/tracking/1001'): attempt 1 failed (… 503 tracking service timed out (transient, retry)); attempt 2 ok
  - rag_search(query='According to its tracking, … Demo Freight Paris in transit', top_k=3): attempt 1 ok
Verification: PASS (retries: 0)

Full decision trace:
   1. [planner] plan: {… "repairs": ["step 2: invalid endpoint '/inventory/{{1.current_city}} river' (id contains spaces); using rag_search"] …}
   2. [researcher] tool_call: {"tool": "mock_api_get", … "attempt": 1, "ok": false, …}
   …
```

The planner (a 3B model) asked for `/inventory/Paris river`. The plan validator turned
that step into a corpus search seeded with the tracking record, so the answer still
came out right. A question the sources can't answer gets the fallback:

```
$ python -m agent.main "What is the capital of Australia?"
Answer: I couldn't verify an answer from the available sources. Sources consulted: capital-canada, capital-brazil, capital-egypt, capital-norway, capital-japan.
…
Verification: FAIL (retries: 2)
```

`--json` prints only the structured `AgentAnswer` (`status`, `answer`, `sources`,
`verification_verdict`, `retries`, `tools_used`). That's what the evals parse.

## Evals

### Golden dataset: `evals/golden.jsonl`

| Category | Cases | What it tests |
|---|---|---|
| `rag` | 7 | corpus questions, including two-part ones ("who, and in what year?") |
| `api` | 2 | demo-API lookups |
| `multi_tool` | 6 | an API record feeds a corpus search (user → city → river); one hits the flaky endpoint |
| `unanswerable` | 5 | not in the corpus, a 404, an always-500 endpoint, a field the record lacks. The only correct output is the Fallback |

Each case lists its expected key facts, relevant doc ids and expected tools. A unit
test checks that every expected key fact actually appears in the corpus or the API
data, so the dataset can't demand an answer the sources don't support.

### Deterministic suite (the gate)

| Check | Passes when |
|---|---|
| Output format | the output parses as `AgentAnswer` and is internally consistent (an answer has PASS + sources; a fallback is the fixed message + sources consulted) |
| Tool-call correctness | the set of tools called equals `expected_tools`, and every call's arguments are valid for the tool schema / API routes |
| Key facts | the answer contains every expected key fact (accent-, case- and digit-grouping-insensitive) |
| Lexical groundedness | every expected key fact is contained, by token overlap, in a single retrieved evidence item |
| Retrieval | every relevant doc id was among the retrieved chunks |
| Refusal | Fallback if and only if the case is unanswerable |

"Lexical groundedness" is named that way on purpose: it's token overlap, not
semantic faithfulness. Semantic faithfulness is the judge suite's job. The eval
normalises text with its own code rather than importing the agent's guard, so a bug
in the agent can't make its own eval pass.

### LLM-judge suite (optional)

DeepEval `FaithfulnessMetric` and `AnswerRelevancyMetric` on answered cases,
threshold `0.7`. It runs only when the judge provider is `openai` or `anthropic` and
its key is set. Otherwise the report says **`LLM-judge: not run (no key)`** and the
suite scores nothing. A local model is deliberately not accepted as a judge, because
a 3B model grading itself isn't evidence.

### Results

Local run on the code in this repo: `ollama:llama3.2:3b`, temperature 0, seed 42,
Python 3.11 on macOS arm64.

| Check | Result |
|---|---|
| Output format | 20/20 |
| Tool-call correctness | 20/20 |
| Key facts | 15/15 |
| Lexical groundedness | 15/15 |
| Retrieval | 13/13 |
| Refusal | 20/20 |
| **Cases passing every check** | **20/20** |
| LLM-judge | not run (no key) |

About 3 minutes and ~73 LLM calls for the 20 cases. Two consecutive runs produced
identical answers, sources and retry counts. Ollama output can differ across
hardware (Metal locally vs CPU in CI), so the CI run is the result of record. See
[Limitations](#limitations) before reading much into the 100%.

### CI: `.github/workflows/evals.yml`

1. **Unit tests**: CPU-only torch, build the index, `pytest`.
2. **Eval suites**: install Ollama, pull `llama3.2:3b` (cached), `python -m evals.run --suite all`.
   A non-zero exit fails the job, which blocks the merge once the job is a required check.
   `eval-report.md` is appended to the job summary and uploaded with the raw
   `eval-results.json`.

To enable the judge, set repository variables `JUDGE_PROVIDER` / `JUDGE_MODEL` and
the matching secret (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).

## Design notes

These came out of measuring the agent rather than guessing:

- **All structured-output fields are required, and the answer comes before the
  verdict.** With Ollama's grammar-constrained decoding, optional fields let the 3B
  model emit `{"sufficient": false}` for everything. It did, even with the answer
  sitting in the evidence. Required fields, with the answer written first, fixed
  this.
- **Compound questions are drafted one part at a time.** Asked "which city …, and
  which river …?", the model reliably answered only the first part, even when the
  parts were listed explicitly (that made it worse). Drafting each part separately
  and joining the answers fixed all six multi-part cases.
- **Deterministic guards run before the LLM verifier.** A groundedness guard rejects
  answers containing names or numbers found in neither the evidence nor the
  question. A completeness guard rejects answers to an N-part question that state
  fewer than N specific facts. I tried a "checklist" LLM verifier instead; it scored
  7/10 against the plain verifier's 8/10 on a labelled probe set, and both of the
  plain verifier's misses were half-answers that the completeness guard catches.
  After the LLM verifier, a consistency check treats a PASS that still names
  missing information as a FAIL: the 3B model sometimes mixes up its output fields.
- **Sources are chosen lexically, not taken from the model's citations.** The model
  cited things like `/api/v1/evidence?ids=[capital-canada]`, or documents it hadn't
  used. Sources are the evidence items that contain the answer's specific terms.
- **The plan is validated before anything runs.** Unknown tools are dropped. Invented
  endpoints become corpus searches. Bare `{}` slots get wired to the previous API
  record. Placeholders are renumbered when a step is dropped. API calls are dropped
  for questions that mention no API record type.
- **Every eval case gets a fresh MCP server process**, so the flaky endpoint's state
  can't leak between cases.

## Limitations

- **The golden set was also the development set.** I tuned the prompts and guards
  while watching these 20 cases. The few-shot examples use no golden data, but 20/20
  is a regression baseline, not a measure of how the agent generalises.
- **The guards are lexical heuristics.** The groundedness guard can't catch a wrong
  pairing of words that both appear in the evidence. The completeness guard detects
  parts with a regex on ", and which/what/who/in what …".
- **In the reported run, every refusal came from the drafter declaring the evidence
  insufficient.** The guards and the LLM verifier are covered by unit tests
  (including mutation checks), but none of them was the deciding check for an
  unanswerable case in this run.
- `API_HINT_WORDS` (which words justify an API call) is specific to this mock API.
- Answers from a 3B model are terse ("Paris; the river Seine.").
- The judge suite hasn't produced scores for these results, because no hosted key was used.

## Layout

```
agent/        graph.py (nodes, routing, plan validation), state.py, prompts.py,
              lexical.py (guards), llm.py (provider factory), mcp_client.py, main.py (CLI)
tools/mcp_server/
              server.py (MCP server), rag.py (chunking, Chroma), mock_data.py (fake API)
corpus/       21 short Markdown documents
evals/        golden.jsonl, deterministic.py, llm_judge.py, run.py, build_index.py
tests/        tools over MCP, graph routing (scripted LLM), guards, eval harness
.github/workflows/evals.yml
```
