# tests/test_environment.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.ticket_router_environment import TicketRouterEnvironment, SCENARIOS, infer_routing
from models import TicketRouterAction

VALID_TEAMS    = ["Billing", "Tech Support", "Account", "Product", "Escalations"]
VALID_LEVELS   = ["low", "medium", "high"]


class TestReset:
    def setup_method(self):
        self.env = TicketRouterEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset(task_type="easy", seed=0)
        assert obs is not None
        assert obs.done == False
        assert obs.reward is None

    def test_reset_easy_populates_fields(self):
        obs = self.env.reset(task_type="easy", seed=0)
        assert obs.task_type == "easy"
        assert len(obs.ticket_subject) > 0
        assert len(obs.ticket_body) > 0
        assert obs.customer_tier in ("standard", "premium", "enterprise")
        assert len(obs.team_status) == 5
        assert len(obs.resolution_history) == 3

    def test_reset_medium(self):
        obs = self.env.reset(task_type="medium", seed=0)
        assert obs.task_type == "medium"
        assert obs.scenario_id.startswith("M")

    def test_reset_hard(self):
        obs = self.env.reset(task_type="hard", seed=0)
        assert obs.task_type == "hard"
        assert obs.scenario_id.startswith("H")

    def test_reset_invalid_task_type_defaults_to_easy(self):
        obs = self.env.reset(task_type="impossible_level", seed=0)
        assert obs.task_type == "easy"

    def test_reset_seed_selects_correct_scenario(self):
        for seed in range(3):
            obs = self.env.reset(task_type="easy", seed=seed)
            expected_id = SCENARIOS["easy"][seed]["scenario_id"]
            assert obs.scenario_id == expected_id

    def test_reset_all_seeds_produce_different_scenarios(self):
        ids = [self.env.reset(task_type="easy", seed=s).scenario_id for s in range(5)]
        assert len(set(ids)) == 5   # all 5 scenarios are distinct

    def test_reset_clears_done_state(self):
        obs = self.env.reset(task_type="easy", seed=0)
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        self.env.step(action)
        # Reset must allow a new episode
        obs2 = self.env.reset(task_type="easy", seed=1)
        assert obs2.done == False

    def test_reset_dynamic_mode(self):
        obs = self.env.reset(
            ticket_body="I cannot log in. Password reset email never arrives.",
            customer_tier="standard",
        )
        assert obs.task_type == "dynamic"
        assert obs.scenario_id.startswith("DYN-")
        assert "Password" in obs.ticket_body or "log" in obs.ticket_body.lower()

    def test_reset_dynamic_infers_subject_from_body(self):
        body = "I was charged twice this month. Please refund."
        obs = self.env.reset(ticket_body=body)
        assert len(obs.ticket_subject) > 0

    def test_reset_team_status_has_required_fields(self):
        obs = self.env.reset(task_type="easy", seed=0)
        for team in obs.team_status:
            assert "name" in team
            assert "queue_length" in team
            assert "avg_resolution_time_min" in team
            assert "specialization" in team

    def test_reset_resolution_history_has_required_fields(self):
        obs = self.env.reset(task_type="easy", seed=0)
        for entry in obs.resolution_history:
            assert "team" in entry
            assert "issue_type" in entry
            assert "success" in entry
            assert "resolution_time_min" in entry


class TestStep:
    def setup_method(self):
        self.env = TicketRouterEnvironment()

    def test_step_without_reset_raises(self):
        env = TicketRouterEnvironment()
        try:
            env.step(TicketRouterAction(primary_team="Billing", priority="medium", urgency="medium"))
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_step_returns_observation(self):
        self.env.reset(task_type="easy", seed=0)
        result = self.env.step(
            TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        )
        assert result is not None
        assert result.done == True

    def test_step_after_done_raises(self):
        self.env.reset(task_type="easy", seed=0)
        action = TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        self.env.step(action)
        try:
            self.env.step(action)
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass

    def test_step_reward_not_none(self):
        self.env.reset(task_type="easy", seed=0)
        result = self.env.step(
            TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        )
        assert result.reward is not None

    def test_step_score_in_valid_range(self):
        self.env.reset(task_type="easy", seed=0)
        result = self.env.step(
            TicketRouterAction(primary_team="Product", priority="low", urgency="low")
        )
        score = result.metadata["score"]
        assert 0.0 <= score <= 1.0

    def test_step_metadata_contains_expected_fields(self):
        self.env.reset(task_type="easy", seed=0)
        result = self.env.step(
            TicketRouterAction(primary_team="Billing", priority="high", urgency="high")
        )
        meta = result.metadata
        for key in ("score", "expected_team", "expected_priority", "expected_urgency",
                    "chosen_team", "chosen_priority", "chosen_urgency",
                    "team_correct", "priority_correct", "urgency_correct", "overload_penalty"):
            assert key in meta, f"Missing metadata key: {key}"

    def test_step_team_correct_flag_true(self):
        scenario = SCENARIOS["easy"][0]   # E001 → Billing/high/high
        self.env.reset(task_type="easy", seed=0)
        result = self.env.step(TicketRouterAction(
            primary_team=scenario["expected_team"],
            priority=scenario["expected_priority"],
            urgency=scenario["expected_urgency"],
        ))
        assert result.metadata["team_correct"] == True

    def test_step_team_correct_flag_false(self):
        scenario = SCENARIOS["easy"][0]
        self.env.reset(task_type="easy", seed=0)
        wrong_team = next(t for t in VALID_TEAMS if t != scenario["expected_team"])
        result = self.env.step(TicketRouterAction(
            primary_team=wrong_team, priority="medium", urgency="medium"
        ))
        assert result.metadata["team_correct"] == False

    def test_step_state_increments(self):
        self.env.reset(task_type="easy", seed=0)
        self.env.step(TicketRouterAction(primary_team="Billing", priority="high", urgency="high"))
        assert self.env.state.step_count == 1


class TestPerfectScore:
    """Perfect routing actions must score 1.0 on every benchmark scenario."""

    SEEDS = [0, 1, 2]

    def _perfect_score(self, task_type: str, seed: int) -> float:
        env = TicketRouterEnvironment()
        scenario = SCENARIOS[task_type][seed]
        env.reset(task_type=task_type, seed=seed)
        result = env.step(TicketRouterAction(
            primary_team=scenario["expected_team"],
            priority=scenario["expected_priority"],
            urgency=scenario["expected_urgency"],
        ))
        return result.metadata["score"]

    def test_perfect_score_easy_seed0(self):
        assert self._perfect_score("easy", 0) == 0.99

    def test_perfect_score_easy_seed1(self):
        assert self._perfect_score("easy", 1) == 0.99

    def test_perfect_score_easy_seed2(self):
        assert self._perfect_score("easy", 2) == 0.99

    def test_perfect_score_medium_seed0(self):
        assert self._perfect_score("medium", 0) == 0.99

    def test_perfect_score_medium_seed1(self):
        assert self._perfect_score("medium", 1) == 0.99

    def test_perfect_score_medium_seed2(self):
        assert self._perfect_score("medium", 2) == 0.99

    def test_perfect_score_hard_seed0(self):
        assert self._perfect_score("hard", 0) == 0.99

    def test_perfect_score_hard_seed1(self):
        assert self._perfect_score("hard", 1) == 0.99

    def test_perfect_score_hard_seed2(self):
        assert self._perfect_score("hard", 2) == 0.99

    def test_mean_score_perfect_actions(self):
        """All 9 benchmark scenarios with correct actions → mean score = 0.99 (clamped from 1.0)."""
        all_scores = []
        for task_type in ["easy", "medium", "hard"]:
            for seed in self.SEEDS:
                all_scores.append(self._perfect_score(task_type, seed))
        mean = sum(all_scores) / len(all_scores)
        assert mean == 0.99, f"Expected mean=0.99, got {mean:.4f}"


class TestDynamicModeRouting:
    """
    Dynamic mode infers expected routing from ticket text via keyword rules.

    Tests verify that infer_routing() drives the expected values stored in the
    scenario, and that stepping with the correct inferred action yields score=0.99.
    """

    def setup_method(self):
        self.env = TicketRouterEnvironment()

    def _reset_and_step(self, ticket_body: str, customer_tier: str = "standard"):
        """
        Reset in dynamic mode and immediately step with the inferred routing action.
        Returns (obs, step_result) so tests can inspect both.
        """
        obs = self.env.reset(ticket_body=ticket_body, customer_tier=customer_tier)
        inferred = infer_routing(obs.ticket_body, obs.ticket_subject)
        action = TicketRouterAction(
            primary_team=inferred["team"],
            priority=inferred["priority"],
            urgency=inferred["urgency"],
        )
        result = self.env.step(action)
        return obs, result

    def test_dynamic_billing_keywords(self):
        obs, result = self._reset_and_step(
            "I was overcharged on my invoice this month. Please process a refund."
        )
        assert obs.task_type == "dynamic"
        assert result.metadata["expected_team"] == "Billing"
        assert result.metadata["score"] == 0.99

    def test_dynamic_account_keywords(self):
        _, result = self._reset_and_step(
            "I cannot login. My password reset email never arrives."
        )
        assert result.metadata["expected_team"] == "Account"
        assert result.metadata["score"] == 0.99

    def test_dynamic_tech_support_keywords(self):
        _, result = self._reset_and_step(
            "The API is returning 500 errors on every call. There is a crash in production."
        )
        assert result.metadata["expected_team"] == "Tech Support"
        assert result.metadata["score"] == 0.99

    def test_dynamic_product_keywords(self):
        _, result = self._reset_and_step(
            "I would like to suggest a new feature for CSV export. No rush, whenever."
        )
        assert result.metadata["expected_team"] == "Product"
        assert result.metadata["score"] == 0.99

    def test_dynamic_escalations_fallback_no_keywords(self):
        # Text with no team or urgency keywords → Escalations fallback
        obs, result = self._reset_and_step("Hi there, just checking in.")
        assert obs.task_type == "dynamic"
        assert result.metadata["expected_team"] == "Escalations"
        assert result.metadata["score"] == 0.99

    def test_dynamic_high_urgency_keywords(self):
        _, result = self._reset_and_step(
            "This is an emergency. Production down, we are completely blocked, need help asap."
        )
        assert result.metadata["expected_priority"] == "high"
        assert result.metadata["expected_urgency"] == "high"
        assert result.metadata["score"] == 0.99

    def test_dynamic_low_urgency_keywords(self):
        _, result = self._reset_and_step(
            "I would like a new feature. No rush, whenever you have time."
        )
        assert result.metadata["expected_priority"] == "low"
        assert result.metadata["expected_urgency"] == "low"
        assert result.metadata["score"] == 0.99

    def test_dynamic_enterprise_tier_preserved(self):
        obs, result = self._reset_and_step(
            "Invoice shows an incorrect charge of $500.", customer_tier="enterprise"
        )
        assert obs.customer_tier == "enterprise"
        assert result.metadata["expected_team"] == "Billing"

    def test_dynamic_multi_intent_billing_wins_over_account(self):
        # "invoice" + "charge" = 2 Billing keywords; "login" = 1 Account keyword → Billing wins
        _, result = self._reset_and_step(
            "My invoice shows a wrong charge. Also I cannot login to review it."
        )
        assert result.metadata["expected_team"] == "Billing"
        assert result.metadata["score"] == 0.99

    def test_dynamic_keyword_stuffing_routes_correctly(self):
        # Repeating "refund" many times still counts as 1 Billing keyword (presence check, not count)
        _, result = self._reset_and_step("refund refund refund refund refund")
        assert result.metadata["expected_team"] == "Billing"
        assert result.metadata["score"] == 0.99
        # Confirm wrong-team action still scores low — stuffing doesn't bypass the grader
        self.env.reset(ticket_body="refund refund refund refund refund")
        wrong = self.env.step(
            TicketRouterAction(primary_team="Tech Support", priority="medium", urgency="medium")
        )
        assert wrong.metadata["score"] < 0.6  # wrong team → no +0.6

    def test_dynamic_billing_enterprise_high_urgency(self):
        # Enterprise billing ticket with urgency keywords → Billing/high/high
        _, result = self._reset_and_step(
            "We have been overcharged $5,000 on our enterprise invoice. "
            "This must be reversed immediately — our accounting closes today.",
            customer_tier="enterprise",
        )
        assert result.metadata["expected_team"] == "Billing"
        assert result.metadata["expected_priority"] == "high"
        assert result.metadata["expected_urgency"] == "high"
        assert result.metadata["score"] == 0.99

    def test_dynamic_inference_vs_preset_consistency(self):
        # Dynamic mode on E001 body should infer the same team as the preset expected
        scenario = SCENARIOS["easy"][0]   # E001: Billing / high / high
        inferred = infer_routing(scenario["body"], scenario["subject"])
        assert inferred["team"] == scenario["expected_team"], (
            f"Dynamic infer_routing disagrees with preset: "
            f"got {inferred['team']}, expected {scenario['expected_team']}"
        )
        # Step with the inferred action → should score 0.99
        obs, result = self._reset_and_step(scenario["body"])
        assert result.metadata["expected_team"] == scenario["expected_team"]
        assert result.metadata["score"] == 0.99
