# tests/test_grader_edge_cases.py
"""
Edge-case tests for the grader (scoring) layer.

Covers boundary conditions that the main test_scoring.py does not exercise:
  - Unusual queue lengths (negative, zero, boundary at 10/11)
  - Empty and all-overloaded team status lists
  - All teams with identical queue lengths
  - Score clamping precision
  - Overload penalty trigger conditions
  - Reward floor for wrong-team actions
  - Pydantic model validation of invalid literals
  - TicketRouterObservation JSON round-trip serialization
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.ticket_router_environment import (
    _compute_score,
    _compute_reward,
    _is_overloaded,
    _better_alternative_exists,
)
from models import TicketRouterAction, TicketRouterObservation


# ── Fixtures ──────────────────────────────────────────────────────────────────

EXPECTED_BILLING_HIGH = {"team": "Billing", "priority": "high", "urgency": "high"}
EXPECTED_ACCOUNT_MED  = {"team": "Account", "priority": "medium", "urgency": "medium"}

BALANCED_STATUS = [
    {"name": "Billing",      "queue_length": 3,  "avg_resolution_time_min": 14, "specialization": "Billing"},
    {"name": "Tech Support", "queue_length": 4,  "avg_resolution_time_min": 28, "specialization": "Tech"},
    {"name": "Account",      "queue_length": 2,  "avg_resolution_time_min": 11, "specialization": "Account"},
    {"name": "Product",      "queue_length": 5,  "avg_resolution_time_min": 38, "specialization": "Product"},
    {"name": "Escalations",  "queue_length": 1,  "avg_resolution_time_min": 20, "specialization": "Escalations"},
]

# Billing overloaded; everything else fine
BILLING_OVERLOADED = [
    {"name": "Billing",      "queue_length": 15, "avg_resolution_time_min": 40, "specialization": "Billing"},
    {"name": "Tech Support", "queue_length": 4,  "avg_resolution_time_min": 28, "specialization": "Tech"},
    {"name": "Account",      "queue_length": 2,  "avg_resolution_time_min": 11, "specialization": "Account"},
    {"name": "Product",      "queue_length": 5,  "avg_resolution_time_min": 38, "specialization": "Product"},
    {"name": "Escalations",  "queue_length": 1,  "avg_resolution_time_min": 20, "specialization": "Escalations"},
]

ALL_OVERLOADED = [
    {"name": "Billing",      "queue_length": 15, "avg_resolution_time_min": 40, "specialization": "Billing"},
    {"name": "Tech Support", "queue_length": 20, "avg_resolution_time_min": 80, "specialization": "Tech"},
    {"name": "Account",      "queue_length": 11, "avg_resolution_time_min": 30, "specialization": "Account"},
    {"name": "Product",      "queue_length": 13, "avg_resolution_time_min": 55, "specialization": "Product"},
    {"name": "Escalations",  "queue_length": 12, "avg_resolution_time_min": 25, "specialization": "Escalations"},
]

ALL_SAME_QUEUE = [
    {"name": team, "queue_length": 5, "avg_resolution_time_min": 15, "specialization": team}
    for team in ["Billing", "Tech Support", "Account", "Product", "Escalations"]
]


# ── _is_overloaded edge cases ─────────────────────────────────────────────────

class TestIsOverloaded:

    def test_queue_11_is_overloaded(self):
        status = [{"name": "Billing", "queue_length": 11}]
        assert _is_overloaded("Billing", status) is True

    def test_queue_10_is_not_overloaded(self):
        # Boundary: strictly > 10 required
        status = [{"name": "Billing", "queue_length": 10}]
        assert _is_overloaded("Billing", status) is False

    def test_negative_queue_is_not_overloaded(self):
        status = [{"name": "Billing", "queue_length": -1}]
        assert _is_overloaded("Billing", status) is False

    def test_zero_queue_is_not_overloaded(self):
        status = [{"name": "Billing", "queue_length": 0}]
        assert _is_overloaded("Billing", status) is False

    def test_team_not_in_status_returns_false(self):
        # Team absent from status list → treated as not overloaded
        assert _is_overloaded("Billing", []) is False
        assert _is_overloaded("Billing", [{"name": "Account", "queue_length": 99}]) is False


# ── _better_alternative_exists edge cases ─────────────────────────────────────

class TestBetterAlternativeExists:

    def test_empty_status_no_alternative(self):
        assert _better_alternative_exists("Billing", []) is False

    def test_all_overloaded_no_alternative(self):
        assert _better_alternative_exists("Billing", ALL_OVERLOADED) is False

    def test_one_alternative_below_threshold(self):
        assert _better_alternative_exists("Billing", BILLING_OVERLOADED) is True

    def test_all_same_queue_below_threshold(self):
        # queue=5 for all → alternatives exist (none overloaded)
        assert _better_alternative_exists("Billing", ALL_SAME_QUEUE) is True


# ── _compute_score edge cases ─────────────────────────────────────────────────

class TestComputeScoreEdgeCases:

    def test_perfect_score_clamped_to_099(self):
        # 0.6 + 0.2 + 0.2 = 1.0, clamped to 0.99
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        assert _compute_score(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS) == 0.99

    def test_all_wrong_clamped_to_001(self):
        # 0.0 → clamped to 0.01
        action = TicketRouterAction(primary_team="Product", priority="low", urgency="low")
        assert _compute_score(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS) == 0.01

    def test_wrong_team_correct_priority_urgency(self):
        # 0 + 0.2 + 0.2 = 0.40
        action = TicketRouterAction(primary_team="Product", priority="high", urgency="high")
        assert _compute_score(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS) == 0.40

    def test_correct_team_only(self):
        # 0.6 + 0 + 0 = 0.60
        action = TicketRouterAction(primary_team="Billing", priority="low", urgency="low")
        assert _compute_score(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS) == 0.60

    def test_overload_penalty_triggers_when_alternative_exists(self):
        # Billing queue=15 and alternatives exist → penalty applies
        # 0.6 + 0.2 + 0.2 - 0.2 = 0.8
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        assert _compute_score(action, EXPECTED_BILLING_HIGH, BILLING_OVERLOADED) == 0.80

    def test_overload_penalty_not_applied_when_all_overloaded(self):
        # All teams overloaded → no better alternative → no penalty
        # 0.6 + 0.2 + 0.2 = 1.0 → clamped to 0.99
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        assert _compute_score(action, EXPECTED_BILLING_HIGH, ALL_OVERLOADED) == 0.99

    def test_empty_team_status_no_penalty(self):
        # No status info → _is_overloaded returns False → no penalty
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        assert _compute_score(action, EXPECTED_BILLING_HIGH, []) == 0.99

    def test_all_same_queue_no_overload_no_penalty(self):
        # All queues=5 → none overloaded → no penalty
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        assert _compute_score(action, EXPECTED_BILLING_HIGH, ALL_SAME_QUEUE) == 0.99

    def test_wrong_team_overloaded_still_penalized(self):
        # Chose wrong team that is overloaded with alternatives → penalty applies
        # 0 + 0.2 + 0.2 - 0.2 = 0.2
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        expected = {"team": "Account", "priority": "high", "urgency": "high"}
        score = _compute_score(action, expected, BILLING_OVERLOADED)
        assert score == 0.20

    def test_score_always_in_range(self):
        # Exhaustive check: all combos of valid team/priority/urgency → score in [0.01, 0.99]
        teams     = ["Billing", "Tech Support", "Account", "Product", "Escalations"]
        levels    = ["low", "medium", "high"]
        expected  = EXPECTED_BILLING_HIGH
        for team in teams:
            for priority in levels:
                for urgency in levels:
                    action = TicketRouterAction(
                        primary_team=team, priority=priority, urgency=urgency
                    )
                    score = _compute_score(action, expected, BALANCED_STATUS)
                    assert 0.01 <= score <= 0.99, (
                        f"Score {score} out of [0.01, 0.99] for {team}/{priority}/{urgency}"
                    )


# ── _compute_reward edge cases ────────────────────────────────────────────────

class TestComputeRewardEdgeCases:

    def test_wrong_team_wrong_priority_urgency_reward(self):
        # -0.3 + 0 + 0 = -0.3
        action = TicketRouterAction(primary_team="Product", priority="low", urgency="low")
        assert _compute_reward(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS) == -0.3

    def test_correct_team_wrong_priority_urgency_reward(self):
        # +0.6 + 0 + 0 = 0.6
        action = TicketRouterAction(primary_team="Billing", priority="low", urgency="low")
        assert _compute_reward(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS) == 0.6

    def test_perfect_reward_is_one(self):
        # +0.6 + 0.2 + 0.2 = 1.0 (reward is NOT clamped unlike score)
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        assert _compute_reward(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS) == 1.0

    def test_overload_penalty_in_reward(self):
        # 0.6 + 0.2 + 0.2 - 0.2 = 0.8
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        assert _compute_reward(action, EXPECTED_BILLING_HIGH, BILLING_OVERLOADED) == 0.8

    def test_worst_case_reward(self):
        # -0.3 + 0 + 0 - 0.2 = -0.5  (wrong team, wrong P/U, overloaded with alt)
        action = TicketRouterAction(primary_team="Billing", priority="low", urgency="low")
        expected = {"team": "Account", "priority": "high", "urgency": "high"}  # P/U don't match
        reward = _compute_reward(action, expected, BILLING_OVERLOADED)
        assert reward == round(-0.3 - 0.2, 4)


# ── Pydantic model validation ─────────────────────────────────────────────────

class TestPydanticModelValidation:

    def test_valid_action_accepted(self):
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="low")
        assert action.primary_team == "Billing"

    def test_invalid_team_raises(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            TicketRouterAction(primary_team="UnknownTeam", priority="high", urgency="high")

    def test_invalid_priority_raises(self):
        with pytest.raises(Exception):
            TicketRouterAction(primary_team="Billing", priority="critical", urgency="high")

    def test_invalid_urgency_raises(self):
        with pytest.raises(Exception):
            TicketRouterAction(primary_team="Billing", priority="high", urgency="extreme")


# ── TicketRouterObservation serialization ─────────────────────────────────────

class TestObservationSerialization:

    def test_observation_round_trip(self):
        obs = TicketRouterObservation(
            ticket_subject="Test subject",
            ticket_body="Test body content",
            customer_tier="enterprise",
            team_status=BALANCED_STATUS,
            resolution_history=[
                {"team": "Billing", "issue_type": "invoice", "success": True, "resolution_time_min": 10}
            ],
            task_type="easy",
            scenario_id="E001",
        )
        data = obs.model_dump()
        obs2 = TicketRouterObservation.model_validate(data)
        assert obs2.ticket_subject == obs.ticket_subject
        assert obs2.ticket_body == obs.ticket_body
        assert obs2.customer_tier == obs.customer_tier
        assert obs2.scenario_id == obs.scenario_id
        assert obs2.task_type == obs.task_type
        assert len(obs2.team_status) == len(obs.team_status)

    def test_observation_default_customer_tier(self):
        obs = TicketRouterObservation()
        assert obs.customer_tier == "standard"

    def test_observation_empty_team_status_default(self):
        obs = TicketRouterObservation()
        assert obs.team_status == []


# ── Floating point precision ──────────────────────────────────────────────────

class TestFloatPrecision:

    def test_perfect_score_strictly_below_one(self):
        import math
        # 0.6 + 0.2 + 0.2 = 1.0 internally → clamped to exactly 0.99
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        score = _compute_score(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS)
        assert score == 0.99
        assert not math.isclose(score, 1.0)

    def test_all_wrong_score_strictly_above_zero(self):
        import math
        # 0.0 internally → clamped to exactly 0.01
        action = TicketRouterAction(primary_team="Product", priority="low", urgency="low")
        score = _compute_score(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS)
        assert score == 0.01
        assert not math.isclose(score, 0.0)

    def test_score_rounded_to_4_decimal_places(self):
        # 0.6 + 0.2 + 0.2 - 0.2 = 0.8 → no rounding ambiguity
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        score = _compute_score(action, EXPECTED_BILLING_HIGH, BILLING_OVERLOADED)
        assert score == round(score, 4)
        assert score == 0.80

    def test_reward_perfect_is_exactly_one(self):
        # Reward is NOT clamped; perfect reward = 1.0 exactly
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        reward = _compute_reward(action, EXPECTED_BILLING_HIGH, BALANCED_STATUS)
        assert reward == 1.0
