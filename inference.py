"""
inference.py — Customer Support Ticket Router
=============================================
Runs easy, medium, and hard routing scenarios against the environment.

Mandatory environment variables:
    API_BASE_URL  — LLM API base URL         (default: HF router)
    MODEL_NAME    — Model identifier          (default: Qwen2.5-72B)
    HF_TOKEN      — Hugging Face / API key

Stdout format (required):
    [START] task=<task> env=ticket_router model=<model>
    [STEP]  step=<n> action=<json> reward=<r> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> score=<score> rewards=<r1,...>
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

# When running as `python inference.py` from inside ticket_router/,
# the server/ and models.py are importable directly.
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from dotenv import load_dotenv
from openai import OpenAI

from server.ticket_router_environment import TicketRouterEnvironment
from models import TicketRouterAction

load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env")   # also check workspace root .env

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
BENCHMARK    = "ticket_router"

TASK_TYPES   = ["easy", "medium", "hard"]
SEEDS        = [0, 1, 2]   # 3 distinct scenarios per difficulty level

DEFAULT_ACTION = {"primary_team": "Escalations", "priority": "medium", "urgency": "medium"}
VALID_TEAMS    = ["Billing", "Tech Support", "Account", "Product", "Escalations"]

SYSTEM_PROMPT = """\
You are an expert customer support routing specialist.

Think step-by-step inside <thinking> tags, then output ONLY the JSON decision.

Output format (exactly):
<thinking>
[your reasoning]
</thinking>
{"primary_team": "...", "priority": "...", "urgency": "..."}

Valid teams: Billing | Tech Support | Account | Product | Escalations
Valid priority/urgency: low | medium | high

---

## 6-STEP DECISION PIPELINE

STEP 1 — EXTRACT SIGNALS
  List every topic mentioned: charges, refunds, login, API errors, permissions, feature requests, etc.

STEP 2 — IDENTIFY ROOT CAUSE (not surface keywords)
  Ask: "What is the customer actually blocked on RIGHT NOW?"
  ⚠ CRITICAL: A keyword is NOT the root cause.
    • "invoice history" in "I can't VIEW my invoice history" → root cause = access/permissions, NOT a billing dispute
    • "payment" in "payment page is confusing" → root cause = UX feedback, NOT a financial dispute
    • "billing" in "billing section won't load" → root cause = tech bug, NOT a billing dispute
  Classify root cause as one of:
    financial_dispute | access_permissions | system_bug | feature_gap | multi_issue | unclear

STEP 3 — RESOLVE CONFLICTS
  If multiple root causes exist: which is the PRIMARY blocker right now?
  If customer explicitly states their primary concern → honour it unconditionally.
  If signals conflict: prefer the issue that blocks the customer from working.

STEP 4 — SELECT TEAM
  financial_dispute                        → Billing
  access_permissions (no payment context)  → Account
  system_bug / API failure                 → Tech Support
  feature_gap / suggestion                 → Product
  multi_issue OR enterprise + SLA mention  → Escalations
  unclear / vague symptoms                 → Escalations
  NOTE: Enterprise + clear bug, NO SLA     → Tech Support (not Escalations)

STEP 5 — QUEUE AWARENESS
  If chosen team has queue_length > 10 AND another team has queue ≤ 10 → switch to that alternative.

STEP 6 — PRIORITY & URGENCY
  high  → customer uses EXPLICIT urgency words: urgent, ASAP, immediately, production down,
           emergency, losing money, SLA breach, cannot work at all, deadline, "right now",
           "as soon as possible", "completely blocked", "entire team blocked"
         → also: service suspended due to non-payment
  ⚠ NOT high: slow performance, some features not loading, intermittent errors, degraded-but-working,
              "never loads" (degraded, not fully blocked), "painfully slow" — these are MEDIUM.
  low   → no rush, suggestion, feedback, "when possible", "nice to have"
  medium → everything else including: slow, intermittent, partial outage, some widgets broken
  Set priority and urgency to the SAME value in almost all cases.

---

## ANTI-KEYWORD RULES (override naive routing)

• Repeated keywords do NOT increase confidence — use semantic meaning, not frequency.
• "invoice" / "billing" words in a VIEWING or ACCESS context → Account, not Billing.
• "account" word alone does NOT mean Account team — check root cause.
• Enterprise tier alone does NOT mean Escalations — only if SLA/time-bound requirement is stated.

---

## EXAMPLES

Example 1 — Multi-issue, customer states primary concern:
  Subject: "Account locked AND overdue invoice question"
  Body: "I can't log in since this morning — my 2FA code is rejected. Also I received an invoice last month with a charge I don't recognise. My biggest concern right now is regaining access."
<thinking>
  Signals: login failure (2FA), invoice question.
  Root cause: customer explicitly says primary concern = regaining access → access_permissions.
  Conflict: invoice mention is secondary; customer resolved it themselves.
  Team: Account.
  Priority: no urgency words → medium.
</thinking>
  → {"primary_team": "Account", "priority": "medium", "urgency": "medium"}

Example 2 — Ambiguous subject, billing root cause:
  Subject: "Service not working — payment or account problem?"
  Body: "My dashboard shows 'Subscription expired — renew to continue'. I tried to renew but the checkout page gives a payment declined error even though my card is valid."
<thinking>
  Signals: subscription expired, payment declined on checkout.
  Root cause: payment failure (financial_dispute) — service is blocked because payment can't go through.
  "checkout page gives payment declined error" = financial transaction failure, not access issue.
  Team: Billing. Service blocked → high urgency.
</thinking>
  → {"primary_team": "Billing", "priority": "high", "urgency": "high"}

Example 3 — Enterprise + explicit SLA → Escalations:
  Subject: "File sharing broken for one user"
  Body: "A single user gets 'Sharing unavailable'. Other users on the same plan work fine. Our enterprise contract requires same-day resolution for service disruptions."
  Tier: enterprise
<thinking>
  Signals: one user can't share files (access_permissions or bug).
  Root cause: single-user permissions issue, BUT enterprise + "same-day resolution" = explicit SLA.
  Rule: enterprise + SLA → Escalations regardless of root cause.
  Service disruption + SLA → high urgency.
</thinking>
  → {"primary_team": "Escalations", "priority": "high", "urgency": "high"}

Example 4 — Enterprise + clear API bug, no SLA → Tech Support:
  Subject: "Webhooks silently failing on all POST /events calls"
  Body: "Webhook consumer stopped receiving events after Tuesday's release. HTTP 500 on every POST /events call. This is blocking our data pipeline. We need a fix immediately."
  Tier: enterprise
<thinking>
  Signals: HTTP 500 on all webhook calls, system-wide after a release.
  Root cause: system_bug (API failure affecting all users).
  Enterprise tier present, but NO SLA / time-bound contract requirement stated.
  "immediately" = urgency word → high.
  Rule: enterprise + clear bug + no SLA → Tech Support.
</thinking>
  → {"primary_team": "Tech Support", "priority": "high", "urgency": "high"}
"""

# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _build_user_prompt(obs) -> str:
    team_lines = "\n".join(
        f"  - {t['name']}: queue={t['queue_length']}"
        + (" *** OVERLOADED ***" if t['queue_length'] > 10 else "")
        + f", avg_resolution={t['avg_resolution_time_min']}min, {t['specialization']}"
        for t in obs.team_status
    )
    history_lines = "\n".join(
        f"  - {h['team']}: {h['issue_type']} | "
        f"success={h['success']} | {h['resolution_time_min']}min"
        for h in obs.resolution_history
    )
    tier_note = ""
    if obs.customer_tier == "enterprise":
        tier_note = " [ENTERPRISE — consider Escalations for ambiguous or multi-issue tickets]"
    elif obs.customer_tier == "premium":
        tier_note = " [PREMIUM]"
    return (
        f"TICKET\n"
        f"Subject : {obs.ticket_subject}\n"
        f"Body    : {obs.ticket_body}\n"
        f"Tier    : {obs.customer_tier}{tier_note}\n\n"
        f"TEAM STATUS (avoid *** OVERLOADED *** teams when possible)\n{team_lines}\n\n"
        f"RESOLUTION HISTORY (last 3)\n{history_lines}\n\n"
        f"Return ONLY the JSON routing decision."
    )


def _call_llm(client: OpenAI, obs) -> Optional[dict]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(obs)},
            ],
            temperature=0.2,
            max_tokens=600,
            timeout=30.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract the first JSON object from the response
            m = re.search(r'\{[^{}]+\}', raw, re.DOTALL)
            if m:
                return json.loads(m.group())
            return None
    except Exception as exc:
        print(f"[LLM_ERROR] {type(exc).__name__}: {exc}", flush=True, file=sys.stderr)
        return None


def _get_action(client: OpenAI, obs) -> tuple:
    result = _call_llm(client, obs)
    if result is not None:
        return result, None
    result = _call_llm(client, obs)   # one retry
    if result is not None:
        return result, "retried_once"
    return DEFAULT_ACTION.copy(), "parse_failed_used_default"


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(
    client: OpenAI,
    env: TicketRouterEnvironment,
    task_type: str,
    seed: int,
) -> float:
    task_label = f"{task_type}_seed{seed}"
    log_start(task=task_label, model=MODEL_NAME)
    _t0 = time.monotonic()

    obs = env.reset(task_type=task_type, seed=seed)
    action_dict, llm_error = _get_action(client, obs)

    # Coerce to valid values
    if action_dict.get("primary_team") not in VALID_TEAMS:
        action_dict["primary_team"] = DEFAULT_ACTION["primary_team"]
        llm_error = (llm_error or "") + " invalid_team"
    if action_dict.get("priority") not in ("low", "medium", "high"):
        action_dict["priority"] = "medium"
    if action_dict.get("urgency") not in ("low", "medium", "high"):
        action_dict["urgency"] = "medium"

    try:
        action = TicketRouterAction(**action_dict)
    except Exception:
        action = TicketRouterAction(**DEFAULT_ACTION)
        llm_error = (llm_error or "") + " action_validation_failed"

    action_str = json.dumps(action_dict, separators=(",", ":"))
    result_obs  = env.step(action)

    reward    = float(result_obs.reward) if result_obs.reward is not None else 0.0
    done      = result_obs.done
    score     = result_obs.metadata.get("score", 0.0)
    error_msg = llm_error
    elapsed   = time.monotonic() - _t0

    log_step(step=1, action=action_str, reward=reward, done=done, error=error_msg)
    success = score >= 0.6
    log_end(success=success, steps=1, score=score, rewards=[reward])
    print(f"[TIMING] task={task_label} elapsed={elapsed:.1f}s", flush=True)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not HF_TOKEN:
        print(
            "[ERROR] HF_TOKEN is not set. Export HF_TOKEN=<your_hf_token> before running.",
            file=sys.stderr,
        )
        sys.exit(1)
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env    = TicketRouterEnvironment()

    all_scores: dict = {t: [] for t in TASK_TYPES}

    for task_type in TASK_TYPES:
        for seed in SEEDS:
            score = run_episode(client, env, task_type, seed)
            all_scores[task_type].append(score)
            print(flush=True)

    print("=" * 55, flush=True)
    print("FINAL SCORES", flush=True)
    for task_type, scores in all_scores.items():
        avg = sum(scores) / len(scores)
        detail = "  ".join(f"{s:.3f}" for s in scores)
        print(f"  {task_type:8s}: avg={avg:.3f}  [{detail}]", flush=True)
    total_avg = sum(s for scores in all_scores.values() for s in scores) / (
        len(TASK_TYPES) * len(SEEDS)
    )
    print(f"  {'overall':8s}: avg={total_avg:.3f}", flush=True)
    print("=" * 55, flush=True)


if __name__ == "__main__":
    main()
