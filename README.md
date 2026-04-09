---
title: Customer Support Ticket Router
emoji: 🎫
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
---

# Customer Support Ticket Router

An OpenEnv environment where an AI agent reads a customer support ticket and decides how to route it — which team handles it, at what priority, and with what urgency.

This is a real enterprise workflow. Organizations processing thousands of support tickets daily rely on accurate routing to reduce resolution time and improve customer satisfaction.

---

## What the agent does

The agent receives a support ticket (subject, body, customer tier) along with team availability data and must output:

| Decision | Options |
|---|---|
| **Primary Team** | Billing · Tech Support · Account · Product · Escalations |
| **Priority** | low · medium · high |
| **Urgency** | low · medium · high |

---

## Observation Space

```
ticket_subject         str    — One-line summary of the issue
ticket_body            str    — Full description written by the customer
customer_tier          str    — standard | premium | enterprise
team_status            list   — Per team: name, queue_length, avg_resolution_time_min, specialization
resolution_history     list   — Last 3 resolved tickets: team, issue_type, success, resolution_time_min
task_type              str    — easy | medium | hard | dynamic
scenario_id            str    — Unique scenario identifier
```

## Action Space

```
primary_team    Literal["Billing","Tech Support","Account","Product","Escalations"]
priority        Literal["low","medium","high"]
urgency         Literal["low","medium","high"]
```

---

## Reward Function

| Condition | Reward |
|---|---|
| Correct team | +0.6 |
| Correct priority | +0.2 |
| Correct urgency | +0.2 |
| Wrong team | −0.3 |
| Chose overloaded team (queue > 10) when better option exists | −0.2 |

Score is clamped to **[0.01, 0.99]** — strictly between 0 and 1 as required by the hackathon spec. Partial credit is given — an agent that gets the team right but misses priority still scores 0.6.

---

## Tasks

### Easy (`task_type="easy"`)
Clear, explicit keywords in the ticket body. All teams have balanced queue lengths. The correct team is unambiguous.

*Example*: "Invoice shows incorrect charge of $250 — please refund immediately" → **Billing / high / high**

### Medium (`task_type="medium"`)
Multi-intent tickets requiring primary intent identification. Moderate team loads. Agent must determine which issue takes precedence.

*Example*: "Can't log in AND have a billing question" → **Account / medium / medium** (login is primary)

### Hard (`task_type="hard"`)
Ambiguous signals + one or more teams overloaded (queue > 10). Agent must reason under load constraints and avoid the overloaded team.

*Example*: "Things seem off since the update..." with Tech Support queue=15 → **Escalations / medium / medium**

### Dynamic (free-form)
Pass any ticket body to `reset()` and the environment infers the expected routing via rule-based analysis. Supports unlimited unique tickets.

---

## Baseline Agent Performance (Qwen2.5-72B-Instruct)

Achieved with a few-shot prompted system prompt in `inference.py`:

| Task | Avg Score | Scenarios |
|---|---|---|
| Easy | **0.990** | 0.990 · 0.990 · 0.990 |
| Medium | **0.990** | 0.990 · 0.990 · 0.990 |
| Hard | **0.793** | 0.990 · 0.400 · 0.990 |
| **Overall** | **0.924** | |

> **Note:** Scores are clamped to `[0.01, 0.99]` by design — the hackathon spec requires scores strictly between 0 and 1. A perfect routing decision therefore returns 0.990, not 1.000.
>
> The hard tier average is lower because H002 is a genuine adversarial scenario (billing-sounding language but an account/permissions root cause). Qwen2.5-72B routed it to Billing instead of Account, scoring 0.400. See `BASELINE_REPORT.md` for the full breakdown.

---

## Setup

### Local (Python)
```bash
cd ticket_router
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t ticket_router-env:latest .
docker run -p 8000:8000 ticket_router-env:latest
```

### Demo UI
Open **http://localhost:8000/demo** — type any ticket, pick a routing, see your score instantly.

---

## Running Inference

```bash
cd ticket_router
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

Expected output:
```
[START] task=easy_seed0 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Billing","priority":"high","urgency":"high"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.990 rewards=1.00

[START] task=medium_seed0 env=ticket_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"primary_team":"Account","priority":"medium","urgency":"medium"} reward=1.00 done=true error=null
[END] success=true steps=1 score=0.990 rewards=1.00

=======================================================
FINAL SCORES
  easy    : avg=0.990  [0.990  0.990  0.990]
  medium  : avg=0.990  [0.990  0.990  0.990]
  hard    : avg=0.990  [0.990  0.990  0.990]
  overall : avg=0.990
=======================================================
```

---

## Running Tests

```bash
cd ticket_router
uv run --with pytest pytest tests/ -v
```

98 tests across 3 files:

| File | Coverage |
|---|---|
| `tests/test_environment.py` | reset, step, episode boundary, perfect score for all 9 benchmark scenarios |
| `tests/test_scoring.py` | `_compute_score`, `_compute_reward`, `_is_overloaded`, `infer_routing` |
| `tests/test_scenarios.py` | scenario bank validity, seed indexing, realistic mean score baselines |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/reset` | Load a ticket scenario |
| POST | `/step` | Submit routing decision |
| GET | `/state` | Current environment state |
| GET | `/schema` | Action/observation schemas |
| GET | `/health` | Health check |
| GET | `/demo` | Interactive web UI |
| WS | `/ws` | WebSocket persistent session |

### Reset parameters
```json
{
  "task_type": "easy",
  "seed": 0,
  "ticket_body": "optional free-form ticket text",
  "customer_tier": "standard"
}
```

---

## Project Structure

```
ticket_router/
├── Dockerfile                        ← Container definition
├── inference.py                      ← LLM inference agent (score: 0.990)
├── models.py                         ← TicketRouterAction, TicketRouterObservation
├── openenv.yaml                      ← OpenEnv spec
├── pyproject.toml                    ← Dependencies
├── validate-submission.sh            ← Pre-submit validation script
├── tests/
│   ├── test_environment.py           ← Environment reset/step/scoring tests
│   ├── test_scoring.py               ← Scoring function unit tests
│   └── test_scenarios.py             ← Scenario bank + mean score tests
└── server/
    ├── app.py                        ← FastAPI app + demo UI
    ├── ticket_router_environment.py  ← Core environment (15 scenarios + dynamic mode)
    └── requirements.txt
```
