"""
Customer Support Ticket Router — Environment Implementation.

Two modes:

  DYNAMIC (free-form) — User/agent provides any ticket body.
    reset(ticket_body="...", ticket_subject="...", customer_tier="standard")
    The environment infers the expected routing via rule-based analysis.
    Supports infinite unique tickets.

  PRESET (benchmark) — 15 curated scenarios across 3 difficulty levels.
    reset(task_type="easy|medium|hard", seed=0..4)
    Deterministic expected answers enable reproducible RL benchmarks.

Scoring (per episode, single step):
  +0.6  correct team
  +0.2  correct priority
  +0.2  correct urgency
  -0.3  wrong team
  -0.2  overloaded team chosen when a better option exists (queue > 10)

Score is clamped to [0.01, 0.99] — strictly between 0 and 1 as required by the hackathon spec.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TicketRouterAction, TicketRouterObservation
except (ImportError, ModuleNotFoundError):
    from models import TicketRouterAction, TicketRouterObservation


# ── Static team data ─────────────────────────────────────────────────────────

_BALANCED_TEAMS = [
    {"name": "Billing",      "queue_length": 3,  "avg_resolution_time_min": 14, "specialization": "Invoices, charges, refunds, payment methods"},
    {"name": "Tech Support", "queue_length": 4,  "avg_resolution_time_min": 28, "specialization": "Software bugs, connectivity, performance"},
    {"name": "Account",      "queue_length": 2,  "avg_resolution_time_min": 11, "specialization": "Login, passwords, profile, permissions"},
    {"name": "Product",      "queue_length": 5,  "avg_resolution_time_min": 38, "specialization": "Feature requests, product feedback, roadmap"},
    {"name": "Escalations",  "queue_length": 1,  "avg_resolution_time_min": 20, "specialization": "Unresolved issues, executive escalations, SLA breaches"},
]

_MODERATE_TEAMS = [
    {"name": "Billing",      "queue_length": 7,  "avg_resolution_time_min": 16, "specialization": "Invoices, charges, refunds, payment methods"},
    {"name": "Tech Support", "queue_length": 8,  "avg_resolution_time_min": 32, "specialization": "Software bugs, connectivity, performance"},
    {"name": "Account",      "queue_length": 6,  "avg_resolution_time_min": 13, "specialization": "Login, passwords, profile, permissions"},
    {"name": "Product",      "queue_length": 9,  "avg_resolution_time_min": 42, "specialization": "Feature requests, product feedback, roadmap"},
    {"name": "Escalations",  "queue_length": 5,  "avg_resolution_time_min": 22, "specialization": "Unresolved issues, executive escalations, SLA breaches"},
]

_STRAINED_TEAMS = [
    {"name": "Billing",      "queue_length": 6,  "avg_resolution_time_min": 18, "specialization": "Invoices, charges, refunds, payment methods"},
    {"name": "Tech Support", "queue_length": 15, "avg_resolution_time_min": 65, "specialization": "Software bugs, connectivity, performance"},
    {"name": "Account",      "queue_length": 7,  "avg_resolution_time_min": 14, "specialization": "Login, passwords, profile, permissions"},
    {"name": "Product",      "queue_length": 12, "avg_resolution_time_min": 55, "specialization": "Feature requests, product feedback, roadmap"},
    {"name": "Escalations",  "queue_length": 4,  "avg_resolution_time_min": 22, "specialization": "Unresolved issues, executive escalations, SLA breaches"},
]

_COMMON_HISTORY = [
    {"team": "Billing",      "issue_type": "invoice dispute",  "success": True,  "resolution_time_min": 17},
    {"team": "Tech Support", "issue_type": "login error",      "success": False, "resolution_time_min": 52},
    {"team": "Account",      "issue_type": "password reset",   "success": True,  "resolution_time_min": 9},
]


# ── Rule-based routing inference ─────────────────────────────────────────────

_ROUTING_RULES = {
    "Billing": [
        "invoice", "charge", "charged", "billing", "payment", "refund", "price",
        "subscription", "fee", "cost", "overcharged", "credit", "debit", "transaction",
        "receipt", "bill", "plan cost", "renewal", "coupon", "discount", "money",
        "paid", "unpaid", "bank", "card", "wallet",
    ],
    "Account": [
        "login", "log in", "sign in", "password", "forgot password", "reset password",
        "locked out", "account access", "cannot access", "not able to login",
        "not able to see", "permission", "permissions", "role", "roles", "profile",
        "username", "otp", "two factor", "2fa", "email not received", "verification",
        "account suspended", "access denied", "unauthorized", "portal access",
    ],
    "Tech Support": [
        "bug", "error", "crash", "api", "performance", "slow", "500", "404",
        "not working", "broken", "loading", "timeout", "latency", "outage",
        "integration", "webhook", "script", "code", "exception", "failure",
        "glitch", "unresponsive", "freezing", "data loss", "sync",
    ],
    "Product": [
        "feature", "feature request", "suggestion", "improvement", "enhance",
        "would like", "could you add", "nice to have", "roadmap", "feedback",
        "idea", "proposal", "request for", "add support", "missing feature",
        "wishlist", "future update",
    ],
    "Escalations": [
        "escalate", "manager", "urgent", "critical", "sla", "sla breach",
        "breach", "not resolved", "still waiting", "unacceptable", "legal",
        "complaint", "unresolved", "days without", "weeks without",
    ],
}

_URGENCY_HIGH = [
    "urgent", "critical", "immediately", "asap", "right now", "emergency",
    "production down", "cannot work", "blocked", "deadline", "losing money",
    "customers affected", "business impact", "high priority", "escalate",
]
_URGENCY_LOW = [
    "when possible", "not urgent", "whenever", "no rush", "low priority",
    "nice to have", "eventually", "future", "suggestion", "feedback",
]


def infer_routing(ticket_body: str, ticket_subject: str = "") -> Dict[str, str]:
    """
    Rule-based routing inference from free-form ticket text.

    Returns expected: {team, priority, urgency}
    """
    text = (ticket_subject + " " + ticket_body).lower()

    # Score each team
    scores: Dict[str, int] = {team: 0 for team in _ROUTING_RULES}
    for team, keywords in _ROUTING_RULES.items():
        for kw in keywords:
            if kw in text:
                scores[team] += 1

    # Pick highest-scoring team; fall back to Escalations if no signal
    best_team = max(scores, key=lambda t: scores[t])
    if scores[best_team] == 0:
        best_team = "Escalations"

    # Urgency / priority
    if any(kw in text for kw in _URGENCY_HIGH):
        priority, urgency = "high", "high"
    elif any(kw in text for kw in _URGENCY_LOW):
        priority, urgency = "low", "low"
    else:
        priority, urgency = "medium", "medium"

    return {"team": best_team, "priority": priority, "urgency": urgency}


# ── Preset scenario bank ─────────────────────────────────────────────────────

SCENARIOS: Dict[str, List[Dict[str, Any]]] = {
    "easy": [
        {
            "scenario_id": "E001",
            "subject": "Invoice #INV-2024-0892 shows incorrect charge of $250",
            "body": "Hi, I received my monthly invoice today and it shows a charge of $250. My subscription plan is $100/month. This is clearly a billing error. Please correct the invoice and process the refund of $150 immediately. I am a long-standing premium customer and expect this resolved today.",
            "customer_tier": "premium",
            "team_status": _BALANCED_TEAMS,
            "expected_team": "Billing", "expected_priority": "high", "expected_urgency": "high",
        },
        {
            "scenario_id": "E002",
            "subject": "Cannot reset my password — reset emails not arriving",
            "body": "I have been locked out of my account since yesterday. I tried the Forgot Password flow three times but the reset email never arrives — not in spam either. I need access urgently as I have a project deadline today.",
            "customer_tier": "standard",
            "team_status": _BALANCED_TEAMS,
            "expected_team": "Account", "expected_priority": "high", "expected_urgency": "high",
        },
        {
            "scenario_id": "E003",
            "subject": "API returning 500 errors on all POST /orders requests",
            "body": "Our integration with your API started throwing 500 Internal Server Error on every POST /orders call as of 14:00 UTC today. GET endpoints work fine. This is breaking our production checkout flow and affecting thousands of end customers. We need a hotfix immediately.",
            "customer_tier": "enterprise",
            "team_status": _BALANCED_TEAMS,
            "expected_team": "Tech Support", "expected_priority": "high", "expected_urgency": "high",
        },
        {
            "scenario_id": "E004",
            "subject": "Feature request: bulk CSV export for reports",
            "body": "We would love to have a CSV export option on the Reports dashboard. Currently we copy-paste data manually. This would be a great addition for our finance team. Not urgent — just wanted to share the feedback.",
            "customer_tier": "standard",
            "team_status": _BALANCED_TEAMS,
            "expected_team": "Product", "expected_priority": "low", "expected_urgency": "low",
        },
        {
            "scenario_id": "E005",
            "subject": "Charged twice for the same order — need immediate refund",
            "body": "I was charged $89.99 twice for order #ORD-55123 placed this morning. Both charges appear on my credit card statement. Please reverse one of the charges as soon as possible.",
            "customer_tier": "standard",
            "team_status": _BALANCED_TEAMS,
            "expected_team": "Billing", "expected_priority": "high", "expected_urgency": "high",
        },
    ],
    "medium": [
        {
            "scenario_id": "M001",
            "subject": "Can't log in AND have a question about last month's invoice",
            "body": "I've been trying to log in to my account for two days. The OTP is not arriving on my registered phone. Also, last month's invoice shows a charge I don't recognise — item PRO_ADD_ON for $45. My primary concern right now is getting back into my account.",
            "customer_tier": "premium",
            "team_status": _MODERATE_TEAMS,
            "expected_team": "Account", "expected_priority": "medium", "expected_urgency": "medium",
        },
        {
            "scenario_id": "M002",
            "subject": "Dashboard extremely slow and some widgets not loading",
            "body": "Our analytics dashboard has been painfully slow since Friday's update. The main chart widget takes over 30 seconds to render, and the User Activity panel never loads at all. Meanwhile the Billing tab works fine.",
            "customer_tier": "standard",
            "team_status": _MODERATE_TEAMS,
            "expected_team": "Tech Support", "expected_priority": "medium", "expected_urgency": "medium",
        },
        {
            "scenario_id": "M003",
            "subject": "Team member lost admin rights after role change",
            "body": "We changed one of our team member's role from Admin to Editor yesterday and now they can't access several sections. We need the permissions reviewed — either fix the role definition or restore their specific access.",
            "customer_tier": "premium",
            "team_status": _MODERATE_TEAMS,
            "expected_team": "Account", "expected_priority": "medium", "expected_urgency": "medium",
        },
        {
            "scenario_id": "M004",
            "subject": "Subscription plan downgrade not reflected and still being charged",
            "body": "I downgraded from Business to Starter three weeks ago. The change shows as Pending in my account settings, but I was still charged the full Business plan rate of $299 this month. I'd like the downgrade applied and a prorated refund.",
            "customer_tier": "premium",
            "team_status": _MODERATE_TEAMS,
            "expected_team": "Billing", "expected_priority": "medium", "expected_urgency": "medium",
        },
        {
            "scenario_id": "M005",
            "subject": "Webhook delivery failures and unclear error logs",
            "body": "Our webhooks for the order.completed event have been failing intermittently since Tuesday. The delivery log shows HTTP 502 responses about 30% of the time. We have retries enabled but some events are still being dropped.",
            "customer_tier": "standard",
            "team_status": _MODERATE_TEAMS,
            "expected_team": "Tech Support", "expected_priority": "medium", "expected_urgency": "medium",
        },
    ],
    "hard": [
        {
            "scenario_id": "H001",
            "subject": "Things seem off with the service since the update",
            "body": "Not sure how to describe this but something doesn't feel right since yesterday's update. Some parts of the dashboard load slowly, a few features don't respond at all, and I can't tell if my recent settings changes were actually saved. It's intermittent.",
            "customer_tier": "standard",
            "team_status": _STRAINED_TEAMS,
            "expected_team": "Escalations", "expected_priority": "medium", "expected_urgency": "medium",
        },
        {
            "scenario_id": "H002",
            "subject": "Cannot access invoice history — says 'plan does not include billing history'",
            "body": "When I navigate to Billing > Invoice History I see the message 'Your current plan does not include billing history'. But I'm on the Business plan, same as my colleague who can view their invoices without any issue. I have not changed my plan or missed any payments. My account settings show the correct subscription tier.",
            "customer_tier": "premium",
            "team_status": _STRAINED_TEAMS,
            "expected_team": "Account", "expected_priority": "medium", "expected_urgency": "medium",
        },
        {
            "scenario_id": "H003",
            "subject": "All users cannot authenticate since v4.2 deployment",
            "body": "Since your v4.2 release deployed at 08:00 UTC, our entire organization of 280 users cannot log in. The authentication endpoint returns HTTP 503. We have confirmed this is not a network issue on our end. We need this fixed immediately.",
            "customer_tier": "enterprise",
            "team_status": _MODERATE_TEAMS,
            "expected_team": "Tech Support", "expected_priority": "high", "expected_urgency": "high",
        },
        {
            "scenario_id": "H004",
            "subject": "Data export failing — could be permissions or a bug",
            "body": "When I try to export our usage report as PDF, I get a generic Export failed message. I've tried Chrome and Firefox, cleared cache, and it still fails. My colleague with the same role can export just fine.",
            "customer_tier": "premium",
            "team_status": _STRAINED_TEAMS,
            "expected_team": "Account", "expected_priority": "medium", "expected_urgency": "medium",
        },
        {
            "scenario_id": "H005",
            "subject": "Unexpected charges and service degradation at the same time",
            "body": "We've had a rough week — two separate issues. First, unexpected charges on this month's invoice for Overage Units we didn't incur. Second, our API response times have been 3-4x slower than normal this week.",
            "customer_tier": "enterprise",
            "team_status": _STRAINED_TEAMS,
            "expected_team": "Escalations", "expected_priority": "high", "expected_urgency": "high",
        },
        {
            "scenario_id": "H006",
            "subject": "Urgent: billing overcharge AND API failures — both unresolved from last week",
            "body": "We have two open issues that require immediate attention. First: we were overcharged $2,000 on the latest invoice for Overage Units we did not use — a ticket was opened 8 days ago with no resolution. Second: our order processing API has been returning intermittent 500 errors for the past 6 hours, affecting live transactions. Both issues are still unresolved.",
            "customer_tier": "enterprise",
            "team_status": _STRAINED_TEAMS,
            "expected_team": "Escalations", "expected_priority": "high", "expected_urgency": "high",
        },
        {
            "scenario_id": "H007",
            "subject": "All reports deleted — 2 years of business data gone",
            "body": "I logged in this morning to find that all 847 of my historical reports have been completely wiped. I run a small business and this data is irreplaceable — it represents 2 years of analytics. I did not delete anything and I am the only administrator. I cannot determine whether this is a system bug, a storage failure, or some form of unauthorized access.",
            "customer_tier": "standard",
            "team_status": _STRAINED_TEAMS,
            "expected_team": "Escalations", "expected_priority": "high", "expected_urgency": "high",
        },
        {
            "scenario_id": "H008",
            "subject": "We urgently need bulk user import — enterprise rollout blocked",
            "body": "Our enterprise deployment is scheduled for Monday and we need to onboard 500 users. The manual user creation process in the UI is far too slow. We need a CSV bulk import feature immediately. Please escalate this to your product team as our entire rollout depends on it.",
            "customer_tier": "enterprise",
            "team_status": _STRAINED_TEAMS,
            "expected_team": "Product", "expected_priority": "high", "expected_urgency": "high",
        },
        {
            "scenario_id": "H009",
            "subject": "Account suspended — payment issue or system glitch?",
            "body": "My account shows 'Suspended — payment overdue' but I paid my invoice 3 days ago and have the bank confirmation. My card was successfully charged. I cannot access any of my data or projects. I have tried logging out and logging back in but the suspension banner remains.",
            "customer_tier": "standard",
            "team_status": _STRAINED_TEAMS,
            "expected_team": "Billing", "expected_priority": "high", "expected_urgency": "high",
        },
        {
            "scenario_id": "H010",
            "subject": "Refund approved 5 days ago — still not reflected in dashboard or bank",
            "body": "I received an email 5 days ago confirming my $199 refund was approved and would appear within 3-5 business days. Today is day 5 and the invoice still shows 'Paid — no refund'. My bank statement also does not show the credit. Could you confirm the refund is still being processed or if there was an error?",
            "customer_tier": "standard",
            "team_status": _STRAINED_TEAMS,
            "expected_team": "Billing", "expected_priority": "medium", "expected_urgency": "medium",
        },
    ],
}


# ── Scoring helpers ───────────────────────────────────────────────────────────

def _is_overloaded(team_name: str, team_status: List[Dict]) -> bool:
    for t in team_status:
        if t["name"] == team_name:
            return t["queue_length"] > 10
    return False


def _better_alternative_exists(chosen: str, team_status: List[Dict]) -> bool:
    return any(t["queue_length"] <= 10 for t in team_status if t["name"] != chosen)


def _compute_score(action: TicketRouterAction, expected: Dict, team_status: List[Dict]) -> float:
    s = 0.0
    if action.primary_team == expected["team"]:
        s += 0.6
    if action.priority == expected["priority"]:
        s += 0.2
    if action.urgency == expected["urgency"]:
        s += 0.2
    if _is_overloaded(action.primary_team, team_status) and _better_alternative_exists(action.primary_team, team_status):
        s -= 0.2
    # Clamp strictly within (0, 1) — hackathon requires scores not equal to 0.0 or 1.0
    return round(max(0.01, min(0.99, s)), 4)


def _compute_reward(action: TicketRouterAction, expected: Dict, team_status: List[Dict]) -> float:
    r = 0.0
    if action.primary_team == expected["team"]:
        r += 0.6
    else:
        r -= 0.3
    if action.priority == expected["priority"]:
        r += 0.2
    if action.urgency == expected["urgency"]:
        r += 0.2
    if _is_overloaded(action.primary_team, team_status) and _better_alternative_exists(action.primary_team, team_status):
        r -= 0.2
    return round(r, 4)


# ── Environment ───────────────────────────────────────────────────────────────

class TicketRouterEnvironment(Environment):
    """
    Customer Support Ticket Router.

    DYNAMIC mode — provide any ticket body:
        obs = env.reset(ticket_body="I can't login...", customer_tier="standard")

    PRESET mode — use curated benchmark scenarios:
        obs = env.reset(task_type="hard", seed=2)

    Single-step episode. Call reset() before each new ticket.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_scenario: Optional[Dict] = None
        self._done: bool = False

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        # Dynamic mode params
        ticket_body: Optional[str] = None,
        ticket_subject: Optional[str] = None,
        customer_tier: str = "standard",
        # Preset mode params
        task_type: str = "easy",
        **kwargs: Any,
    ) -> TicketRouterObservation:
        """
        Load a ticket scenario.

        Dynamic mode  → provide ticket_body (any free-form text).
        Preset mode   → provide task_type + seed.
        """
        self._done = False
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        if ticket_body:
            # ── Dynamic mode ──────────────────────────────────────────────
            subject = ticket_subject or ticket_body[:80].split(".")[0].strip()
            inferred = infer_routing(ticket_body, subject)
            scenario = {
                "scenario_id": f"DYN-{self._state.episode_id[:8].upper()}",
                "subject": subject,
                "body": ticket_body,
                "customer_tier": customer_tier,
                "team_status": _BALANCED_TEAMS,
                "resolution_history": _COMMON_HISTORY,
                "expected_team": inferred["team"],
                "expected_priority": inferred["priority"],
                "expected_urgency": inferred["urgency"],
                "mode": "dynamic",
                "inferred_routing": inferred,
            }
        else:
            # ── Preset mode ───────────────────────────────────────────────
            if task_type not in SCENARIOS:
                task_type = "easy"
            pool = SCENARIOS[task_type]
            idx = (seed or 0) % len(pool)
            scenario = dict(pool[idx])
            scenario["resolution_history"] = _COMMON_HISTORY
            scenario["mode"] = "preset"

        self._current_scenario = scenario
        return TicketRouterObservation(
            ticket_subject=scenario["subject"],
            ticket_body=scenario["body"],
            customer_tier=scenario["customer_tier"],
            team_status=scenario["team_status"],
            resolution_history=scenario["resolution_history"],
            task_type=task_type if not ticket_body else "dynamic",
            scenario_id=scenario["scenario_id"],
            done=False,
            reward=None,
        )

    # ── step ──────────────────────────────────────────────────────────────────

    def step(
        self,
        action: TicketRouterAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> TicketRouterObservation:
        if self._current_scenario is None:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("Episode finished. Call reset() to start a new one.")

        self._state.step_count += 1
        self._done = True

        scenario = self._current_scenario
        team_status = scenario["team_status"]
        expected = {
            "team":     scenario["expected_team"],
            "priority": scenario["expected_priority"],
            "urgency":  scenario["expected_urgency"],
        }

        reward = _compute_reward(action, expected, team_status)
        score  = _compute_score(action, expected, team_status)

        return TicketRouterObservation(
            ticket_subject=scenario["subject"],
            ticket_body=scenario["body"],
            customer_tier=scenario["customer_tier"],
            team_status=team_status,
            resolution_history=scenario["resolution_history"],
            task_type=scenario.get("task_type", "dynamic"),
            scenario_id=scenario["scenario_id"],
            done=True,
            reward=reward,
            metadata={
                "score":                   score,
                "expected_team":           expected["team"],
                "expected_priority":       expected["priority"],
                "expected_urgency":        expected["urgency"],
                "chosen_team":             action.primary_team,
                "chosen_priority":         action.priority,
                "chosen_urgency":          action.urgency,
                "team_correct":            action.primary_team == expected["team"],
                "priority_correct":        action.priority == expected["priority"],
                "urgency_correct":         action.urgency == expected["urgency"],
                "overload_penalty":        (
                    _is_overloaded(action.primary_team, team_status)
                    and _better_alternative_exists(action.primary_team, team_status)
                ),
                "mode": scenario.get("mode", "preset"),
            },
        )

    # ── state ─────────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state
