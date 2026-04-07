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
You are an expert customer support routing specialist. You will receive a support ticket with team status and must return a routing decision.

Return ONLY a JSON object — no markdown, no explanation, no extra text:
{"primary_team": "...", "priority": "...", "urgency": "..."}

TEAM SELECTION RULES (apply in order):

1. ESCALATIONS — use when ANY of these apply:
   - Multiple UNRELATED issues in one ticket (e.g. billing AND API bug AND login)
   - Vague/intermittent symptoms with no clear root cause ("something feels off", "some parts don't work")
   - Issue already sent to another team and remains unresolved
   - Customer uses words: "escalate", "manager", "legal", "unacceptable", "weeks without"
   - Enterprise customer + SLA or time-bound requirement mentioned → Escalations regardless of issue type
   - NOTE: Enterprise + clear API/bug issue with NO SLA mention → TECH SUPPORT, not Escalations
   - NOTE: If the customer states their PRIMARY concern explicitly, route to that team (unless SLA/enterprise applies)

2. BILLING — use when the ROOT CAUSE is financial, even if account UI is involved:
   - Wrong charge, double charge, refund request, invoice dispute
   - "Account suspended — payment required" or payment method page not saving → BILLING (payment root cause)
   - Subscription plan, renewal, coupon, overage charges

3. ACCOUNT — use when the issue is user-specific access or permissions:
   - Login, OTP, forgot password, locked out (NO payment context)
   - Customer explicitly states login/access is their PRIMARY concern → Account
   - One specific user cannot do something that others with the same role CAN → permissions

4. TECH SUPPORT — use when the issue is a clear system-wide bug or API failure:
   - API errors (500, 404, timeouts), webhooks, integrations — even for enterprise customers
   - Performance degradation affecting ALL users
   - Software crashes, data sync failures

5. PRODUCT — use when no immediate fix is possible:
   - Feature requests, suggestions, roadmap questions

AVOID teams with queue_length > 10 when a reasonable alternative exists.

PRIORITY & URGENCY RULES:
- "high" ONLY when ticket contains explicit urgency words: urgent, urgently, immediately, ASAP, deadline, production down, customers affected, SLA breach, losing money, cannot work, blocked, emergency, "as soon as possible", "right now"
- "high" also for: "account suspended — payment required" (service blocked due to billing)
- "low" only when: not urgent, when possible, no rush, nice to have, suggestion, feedback, future
- "medium" for everything else — slow performance, intermittent issues, locked out without urgency words, degraded-but-working

IMPORTANT: priority and urgency are almost always the same value. Set both identically.

EXAMPLES (follow these patterns exactly):

Example 1 — Multi-issue subject but customer states a clear primary concern:
  Subject: "Can't log in AND have a question about my invoice"
  Body: "...My primary concern right now is getting back into my account."
  → {"primary_team": "Account", "priority": "medium", "urgency": "medium"}
  Reason: Customer explicitly named Account as their primary concern. Route there.

Example 2 — Ambiguous subject but body reveals billing root cause:
  Subject: "Something wrong with my account or billing — not sure which"
  Body: "I tried to update my payment method and the page keeps refreshing. I see a banner: Account suspended — payment required."
  → {"primary_team": "Billing", "priority": "high", "urgency": "high"}
  Reason: Body shows payment method failure + suspended-due-to-payment = billing root. Service blocked = high urgency.

Example 3 — Enterprise + SLA = Escalations even if team seems clear:
  Subject: "User unable to collaborate — access or feature issue?"
  Body: "One user gets error. Others on same plan are fine. Our SLA requires this resolved within 4 hours."
  Customer tier: enterprise
  → {"primary_team": "Escalations", "priority": "high", "urgency": "high"}
  Reason: Enterprise + explicit SLA time requirement → Escalations.

Example 4 — Enterprise + clear API bug, no SLA = Tech Support:
  Subject: "API returning 500 errors on all POST requests"
  Body: "...breaking our production checkout flow and affecting thousands of customers. We need a hotfix immediately."
  Customer tier: enterprise
  → {"primary_team": "Tech Support", "priority": "high", "urgency": "high"}
  Reason: Clear API/system bug. No SLA mentioned. "Immediately" = high urgency.
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
            max_tokens=150,
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
    except Exception:
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

    log_step(step=1, action=action_str, reward=reward, done=done, error=error_msg)
    success = score >= 0.6
    log_end(success=success, steps=1, score=score, rewards=[reward])
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "hf_placeholder")
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
