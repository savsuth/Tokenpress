<p align="center">
  <strong>ptk — Python Token Reducer</strong><br/>
  Zero dependencies · Auto type detection · 361 tests
</p>

<table align="center">
  <tr>
    <td align="left" valign="middle">
      <a href="https://github.com/amahi2001/python-token-killer/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/amahi2001/python-token-killer/ci.yml?branch=main&style=flat-square&label=CI" alt="CI"/></a><br/>
      <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/><br/>
      <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow?style=flat-square" alt="License"/></a>
    </td>
  </tr>
</table>

---

## LLM calls carry unnecessary weight

Consider a typical API response passed into an agent:

```json
{
  "user": {
    "id": 8821,
    "name": "Alice Chen",
    "email": "alice@example.com",
    "bio": null,
    "avatar_url": null,
    "phone": null,
    "address": null,
    "metadata": {},
    "preferences": {
      "theme": "dark",
      "notifications": null,
      "newsletter": null
    },
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-06-20T14:22:00Z",
    "last_login": null,
    "is_verified": true,
    "is_active": true
  },
  "errors": null,
  "warnings": []
}
```

Seven null fields and two empty containers add no value: the LLM reads them, you pay for them, and it learns nothing from them. `ptk` removes this noise:

```python
import ptk
ptk(response)
```

```json
{"user":{"id":8821,"name":"Alice Chen","email":"alice@example.com","preferences":{"theme":"dark"},"created_at":"2024-01-15T10:30:00Z","updated_at":"2024-06-20T14:22:00Z","is_verified":true,"is_active":true}}
```

A 52% reduction in tokens, with no loss of information and no configuration required.

```bash
pip install python-token-killer
# or
uv add python-token-killer
```

---

## Benchmarks

Token counts measured via tiktoken (`cl100k_base`, the tokenizer used by GPT-4 and Claude):

```
Input                          Tokens (before)   Tokens (after)   Saved
─────────────────────────────────────────────────────────────────────────
API response (JSON)                    1,450              792      45%
Python module (code → sigs)            2,734              309      89%
CI log (58 lines, errors only)         1,389              231      83%
50 user records (tabular)              2,774              922      67%
Verbose prose (text)                     101               74      27%
─────────────────────────────────────────────────────────────────────────
Total                                 11,182            2,627      76%
```

At Claude Sonnet 4.6 pricing ($3 per 1M input tokens), a 76% reduction on 100k tokens per day saves approximately $6 per month per user. This scales directly with your user base and the number of iterations in your agent loop.

Run the benchmark yourself: `python benchmarks/bench.py`

---

## How it works

`ptk` accepts any Python object, detects its content type, and applies the appropriate compression strategy:

| Input           | Strategy                                                                                                   | Savings |
| --------------- | ---------------------------------------------------------------------------------------------------------- | ------- |
| `dict` / `list` | Strips `null`, `""`, `[]`, `{}` recursively. Tabular encoding for uniform arrays.                          | 40–70%  |
| Code            | Strips comments (preserves `# noqa`, `# type: ignore`, `TODO`). Collapses docstrings. Extracts signatures. | 25–89%  |
| Logs            | Collapses duplicate lines with counts. Filters to errors and stack traces.                                 | 60–90%  |
| Diffs           | Folds unchanged context. Strips git noise (`index`, `old mode`).                                           | 50–75%  |
| Text            | Abbreviates verbose words (`implementation→impl`, `configuration→config`). Removes filler.                 | 10–30%  |

---

## Usage

```python
import ptk

# ── auto-detected, one call ──────────────────────────────────
ptk.minimize(api_response)        # dict/list → compact JSON, nulls stripped
ptk.minimize(source_code)         # strips comments, collapses docstrings
ptk.minimize(log_output)          # dedup repeated lines, keep errors
ptk.minimize(git_diff)            # fold context, keep changes
ptk.minimize(any_object)          # always returns a string, never raises

# ── aggressive mode: maximum compression ─────────────────────
ptk.minimize(response, aggressive=True)

# ── force content type ───────────────────────────────────────
ptk.minimize(text, content_type="code", mode="signatures")  # sigs only
ptk.minimize(logs, content_type="log", errors_only=True)    # errors only

# ── stats: token counts + savings ────────────────────────────
ptk.stats(response)
# {
#   "output": "...",
#   "original_tokens": 1450,
#   "minimized_tokens": 792,
#   "savings_pct": 45.4,
#   "content_type": "dict"
# }

# ── callable shorthand ───────────────────────────────────────
ptk(response)  # same as ptk.minimize(response)

# ── preserve nulls when they carry meaning ───────────────────
ptk.minimize({"status": "pending", "error": None}, strip_nulls=False)
# → {"status":"pending","error":null}
```

---

## Real-world examples

### RAG pipeline: compress retrieved documents before they reach the prompt

Retrievers often return full documents, but the LLM needs the content rather than the metadata surrounding it.

```python
import ptk

def build_context(docs: list[dict]) -> str:
    """Compress retrieved docs before injecting into an LLM prompt."""
    chunks = []
    for doc in docs:
        content = ptk.minimize(doc["content"])   # strip boilerplate
        chunks.append(f"[{doc['source']}]\n{content}")
    return "\n\n---\n\n".join(chunks)
```

A complete working demo with token counts is available at [`examples/rag_pipeline.py`](examples/rag_pipeline.py).

---

### LangGraph / LangChain: compress tool outputs between nodes

Place this node between a tool call and the next LLM call so that tool outputs are reduced before re-entering the context window.

```python
import ptk

def compress_tool_output(state: dict) -> dict:
    """Compress the last tool message before the next LLM call."""
    state["messages"][-1]["content"] = ptk.minimize(
        state["messages"][-1]["content"], aggressive=True
    )
    return state
```

A complete agent loop with per-step token savings is available at [`examples/langgraph_agent.py`](examples/langgraph_agent.py).

---

### Log triage: surface only failures to your LLM

A 10,000-line CI log can be reduced to just the failures and their stack traces.

```python
import ptk

errors = ptk.minimize(ci_log, content_type="log", aggressive=True)
# 80%+ fewer tokens, same diagnostic signal.
```

A before-and-after demo is available at [`examples/log_triage.py`](examples/log_triage.py).

---

## API reference

### `ptk.minimize(obj, *, aggressive=False, content_type=None, **kw) → str`

- **`aggressive=True`** maximizes compression: timestamps are stripped, code is reduced to signatures, and logs are reduced to errors only
- **`content_type`** overrides auto-detection: `"dict"`, `"list"`, `"code"`, `"log"`, `"diff"`, `"text"`
- **`format`** controls dict output: `"json"` (default), `"kv"`, `"tabular"`
- **`mode`** controls code output: `"clean"` (default) or `"signatures"`
- **`errors_only`** filters logs to errors and stack traces

### `ptk.stats(obj, **kw) → dict`

Shares the same interface as `minimize`. Returns `output`, `original_tokens`, `minimized_tokens`, `savings_pct`, and `content_type`.

### `ptk(obj)` callable shorthand

The module itself is callable: `ptk(x)` is equivalent to `ptk.minimize(x)`.

---

## Comparison

| Tool                                                              | Type           | Tradeoff                                          |
| ----------------------------------------------------------------- | -------------- | ------------------------------------------------- |
| **ptk**                                                           | Python library | One call, any Python object, zero dependencies    |
| [RTK](https://github.com/rtk-ai/rtk)                              | Rust CLI       | Compresses shell command output for coding agents |
| [claw-compactor](https://github.com/open-compress/claw-compactor) | Python library | 14-stage AST-aware pipeline, heavier setup        |
| [LLMLingua](https://github.com/microsoft/LLMLingua)               | Python library | Neural compression, requires GPU                  |

---

## Design principles

- **Zero required dependencies.** Standard library only; `tiktoken` is optional and used solely for exact token counts.
- **Never raises.** Any Python object produces a string. Circular references, `bytes`, `nan`, and generators are all handled gracefully.
- **Never mutates.** The input remains untouched.
- **Thread-safe.** Stateless singleton minimizers.
- **Fast.** Precompiled regular expressions, `frozenset` lookups, and single-pass algorithms result in microsecond-level execution per call.

---

## Development

```bash
git clone https://github.com/amahi2001/python-token-killer.git
cd python-token-killer
uv sync          # installs all dev dependencies, creates .venv automatically
make check       # lint + typecheck + 361 tests
```
