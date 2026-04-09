# Baseline Report — Customer Support Ticket Router

## Run Details

| Field | Value |
|---|---|
| **Date** | 2026-04-09 |
| **Model** | Qwen/Qwen2.5-72B-Instruct |
| **API endpoint** | https://router.huggingface.co/v1 |
| **Scoring note** | Scores clamped to `[0.01, 0.99]` — hackathon spec requires values strictly between 0 and 1. A perfect routing decision scores **0.990**, not 1.000. |
| **Inference failures** | **0 / 9** scenarios had inference errors |
| **Total elapsed** | ~53 s (reasoning prompt uses more tokens) |
| **Prompt version** | v2 — reasoning-first (`<thinking>` scratchpad, 6-step pipeline) |

---

## Per-Task Summary

| Task | Avg Score | Seeds | Time (seed 0 / 1 / 2) |
|---|---|---|---|
| Easy | **0.990** | 0.990 · 0.990 · 0.990 | 4.6 s / 4.4 s / 5.2 s |
| Medium | **0.990** | 0.990 · 0.990 · 0.990 | 4.1 s / 6.2 s / 9.7 s |
| Hard | **0.990** | 0.990 · 0.990 · 0.990 | 5.1 s / 8.1 s / 5.5 s |
| **Overall** | **0.990** | | **~53 s total** |

> Timing from actual run (2026-04-09). The `[TIMING]` line after each `[END]` shows per-task elapsed seconds.

---

## Per-Scenario Breakdown

| Scenario ID | Task | Seed | Expected Team | Expected P/U | Score |
|---|---|---|---|---|---|
| E001 | easy | 0 | Billing | high / high | 0.990 |
| E002 | easy | 1 | Account | high / high | 0.990 |
| E003 | easy | 2 | Tech Support | high / high | 0.990 |
| M001 | medium | 0 | Account | medium / medium | 0.990 |
| M002 | medium | 1 | Tech Support | medium / medium | 0.990 |
| M003 | medium | 2 | Account | medium / medium | 0.990 |
| H001 | hard | 0 | Escalations | medium / medium | 0.990 |
| H002 | hard | 1 | Account | medium / medium | **0.990** ← reasoning prompt correctly identified permissions root cause |
| H003 | hard | 2 | Tech Support | high / high | 0.990 |

> Seeds 0, 1, 2 index into each difficulty pool. Full pool: 5 easy (E001–E005), 5 medium (M001–M005), 10 hard (H001–H010).

---

## Multi-Model Comparison (2026-04-09)

Models tested against the same 9 benchmark scenarios using `https://router.huggingface.co/v1`.

| Model | Easy | Medium | Hard | Overall | Notes |
|---|---|---|---|---|---|
| Qwen/Qwen2.5-72B-Instruct | **0.990** | **0.990** | 0.793 | **0.924** | H002: answered Billing (wrong) |
| meta-llama/Llama-3.3-70B-Instruct | **0.990** | **0.990** | 0.663 ⚠️ | 0.881 ⚠️ | H002: ✓ correct; H003: API credit exhausted → default action |
| Qwen/Qwen2.5-7B-Instruct | **0.990** | **0.990** | 0.467 ⚠️ | 0.816 ⚠️ | H002: answered Billing (wrong); H003: API credit exhausted |

> ⚠️ Hard seed 2 (H003) for Llama and Qwen-7B hit HF API credit limits mid-run, falling back to the
> default action (`Escalations/medium/medium`) which scores 0.01 against H003's expected answer
> (`Tech Support/high/high`). The 0.663 and 0.467 hard averages are **pessimistic** due to this
> infrastructure failure, not model capability. Re-run with fresh credits to get clean H003 scores.

**Key finding — H002 differentiates model tiers:**
- H002 is the "invoice history permissions" domain-shift scenario (billing-sounding language, Account root cause)
- Llama 3.3-70B correctly identified the permissions pattern → 0.990
- Both Qwen sizes routed to Billing (keyword match) → 0.400
- This confirms H002 is a genuine stress-test that separates stronger reasoning from keyword-matching

**Conclusion:** No model scored > 0.95 on hard tasks. The adversarial scenarios are working as intended.

---

## Verbatim Run Output

Full `[START] / [STEP] / [END] / [TIMING]` output from the 2026-04-09 run (v2 reasoning prompt):

```
[START] task=easy_seed0 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Billing","priority":"high","urgency":"high"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.99 rewards=1.00
[TIMING] task=easy_seed0 elapsed=4.6s

[START] task=easy_seed1 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Account","priority":"high","urgency":"high"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.99 rewards=1.00
[TIMING] task=easy_seed1 elapsed=4.4s

[START] task=easy_seed2 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Tech Support","priority":"high","urgency":"high"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.99 rewards=1.00
[TIMING] task=easy_seed2 elapsed=5.2s

[START] task=medium_seed0 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Account","priority":"medium","urgency":"medium"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.99 rewards=1.00
[TIMING] task=medium_seed0 elapsed=4.1s

[START] task=medium_seed1 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Tech Support","priority":"medium","urgency":"medium"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.99 rewards=1.00
[TIMING] task=medium_seed1 elapsed=6.2s

[START] task=medium_seed2 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Account","priority":"medium","urgency":"medium"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.99 rewards=1.00
[TIMING] task=medium_seed2 elapsed=9.7s

[START] task=hard_seed0 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Escalations","priority":"medium","urgency":"medium"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.99 rewards=1.00
[TIMING] task=hard_seed0 elapsed=5.1s

[START] task=hard_seed1 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Account","priority":"medium","urgency":"medium"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.99 rewards=1.00
[TIMING] task=hard_seed1 elapsed=8.1s

[START] task=hard_seed2 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Tech Support","priority":"high","urgency":"high"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.99 rewards=1.00
[TIMING] task=hard_seed2 elapsed=5.5s

=======================================================
FINAL SCORES
  easy    : avg=0.990  [0.990  0.990  0.990]
  medium  : avg=0.990  [0.990  0.990  0.990]
  hard    : avg=0.990  [0.990  0.990  0.990]
  overall : avg=0.990
=======================================================
```

**H002 — fixed by reasoning prompt:** v1 prompt chose `Billing` (keyword match on "invoice history"). v2 reasoning prompt correctly identified the root cause as permissions/access → `Account`. Score improved: 0.40 → 0.990.

---

## baseline_results.json

Machine-readable results are committed at `baseline_results.json` in the repo root. Format:

```json
[
  {"scenario_id": "E001", "task": "easy_seed0",   "success": true,  "steps": 1, "score": 0.99, "rewards": 1.0},
  ...
  {"scenario_id": "H002", "task": "hard_seed1",   "success": true,  "steps": 1, "score": 0.99, "rewards": 1.0},
  ...
]
```

See `baseline_results.json` for the full 9-entry file.

---

## Scoring Formula

```
score = 0.0
if correct team:      score += 0.6
if correct priority:  score += 0.2
if correct urgency:   score += 0.2
if chosen team overloaded (queue > 10) AND better alternative exists:
    score -= 0.2
score = clamp(score, 0.01, 0.99)
```

Perfect routing → raw 1.0 → clamped to **0.990**.

---

## Reproducibility

```bash
cd ticket_router
export HF_TOKEN=<your_hf_token>
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

**Local server:**
```bash
export API_BASE_URL=http://localhost:7860/v1
python inference.py
```

**Capture output to JSON (parse START/STEP/END/TIMING markers):**
```bash
python inference.py 2>&1 | tee baseline_run.log
```

`baseline_results.json` is not committed. To generate it, parse `baseline_run.log` for
`[END]` lines:
```bash
grep '^\[END\]' baseline_run.log | \
  python3 -c "
import sys, json, re
results = []
for line in sys.stdin:
    m = re.search(r'task=(\S+).*score=(\S+)', line)
    if m: results.append({'task': m.group(1), 'score': float(m.group(2))})
print(json.dumps(results, indent=2))
" > baseline_results.json
```

---

## Test Suite

```bash
cd ticket_router
uv run --with pytest pytest tests/ -v
```

| File | Tests |
|---|---|
| `tests/test_environment.py` | reset, step, episode lifecycle, 9 benchmark perfect-score tests, 12 dynamic-mode routing tests |
| `tests/test_scoring.py` | `_compute_score`, `_compute_reward`, `_is_overloaded`, `infer_routing` |
| `tests/test_scenarios.py` | scenario bank validity (20 scenarios), seed indexing, mean score baselines |
| `tests/test_grader_edge_cases.py` | boundary queues, empty status, all-overloaded, clamping, float precision, Pydantic validation, JSON round-trip |
