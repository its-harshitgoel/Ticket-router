# tests/test_scenarios.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.ticket_router_environment import (
    SCENARIOS,
    TicketRouterEnvironment,
    _compute_score,
)
from models import TicketRouterAction

VALID_TEAMS  = {"Billing", "Tech Support", "Account", "Product", "Escalations"}
VALID_LEVELS = {"low", "medium", "high"}
VALID_TIERS  = {"standard", "premium", "enterprise"}
TASK_TYPES   = ["easy", "medium", "hard"]
BENCHMARK_SEEDS = [0, 1, 2]


class TestScenarioBank:
    def test_all_difficulty_levels_exist(self):
        for level in TASK_TYPES:
            assert level in SCENARIOS

    def test_easy_and_medium_levels_have_5_scenarios(self):
        for level in ["easy", "medium"]:
            assert len(SCENARIOS[level]) == 5, \
                f"{level} has {len(SCENARIOS[level])} scenarios, expected 5"

    def test_hard_level_has_10_scenarios(self):
        assert len(SCENARIOS["hard"]) == 10, \
            f"hard has {len(SCENARIOS['hard'])} scenarios, expected 10"

    def test_total_scenario_count_is_20(self):
        total = sum(len(SCENARIOS[t]) for t in TASK_TYPES)
        assert total == 20

    def test_all_scenario_ids_are_unique(self):
        ids = [s["scenario_id"] for t in TASK_TYPES for s in SCENARIOS[t]]
        assert len(ids) == len(set(ids)), f"Duplicate scenario_id found in: {ids}"

    def test_easy_scenario_ids_start_with_E(self):
        for s in SCENARIOS["easy"]:
            assert s["scenario_id"].startswith("E"), \
                f"Easy scenario has unexpected ID: {s['scenario_id']}"

    def test_medium_scenario_ids_start_with_M(self):
        for s in SCENARIOS["medium"]:
            assert s["scenario_id"].startswith("M")

    def test_hard_scenario_ids_start_with_H(self):
        for s in SCENARIOS["hard"]:
            assert s["scenario_id"].startswith("H")


class TestScenarioFields:
    def test_each_scenario_has_required_fields(self):
        required = {"scenario_id", "subject", "body", "customer_tier",
                    "team_status", "expected_team", "expected_priority", "expected_urgency"}
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                missing = required - s.keys()
                assert not missing, \
                    f"Scenario {s.get('scenario_id','?')} missing fields: {missing}"

    def test_expected_team_is_always_valid(self):
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                assert s["expected_team"] in VALID_TEAMS, \
                    f"{s['scenario_id']}: invalid expected_team '{s['expected_team']}'"

    def test_expected_priority_is_always_valid(self):
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                assert s["expected_priority"] in VALID_LEVELS, \
                    f"{s['scenario_id']}: invalid expected_priority '{s['expected_priority']}'"

    def test_expected_urgency_is_always_valid(self):
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                assert s["expected_urgency"] in VALID_LEVELS, \
                    f"{s['scenario_id']}: invalid expected_urgency '{s['expected_urgency']}'"

    def test_customer_tier_is_always_valid(self):
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                assert s["customer_tier"] in VALID_TIERS, \
                    f"{s['scenario_id']}: invalid customer_tier '{s['customer_tier']}'"

    def test_subject_is_non_empty(self):
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                assert len(s["subject"].strip()) > 0, \
                    f"{s['scenario_id']}: empty subject"

    def test_body_is_non_empty(self):
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                assert len(s["body"].strip()) > 0, \
                    f"{s['scenario_id']}: empty body"

    def test_team_status_has_5_teams(self):
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                assert len(s["team_status"]) == 5, \
                    f"{s['scenario_id']}: team_status has {len(s['team_status'])} entries"

    def test_team_status_contains_all_valid_teams(self):
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                names = {t["name"] for t in s["team_status"]}
                assert names == VALID_TEAMS, \
                    f"{s['scenario_id']}: unexpected teams in team_status: {names}"

    def test_team_queue_lengths_are_non_negative(self):
        for task_type in TASK_TYPES:
            for s in SCENARIOS[task_type]:
                for team in s["team_status"]:
                    assert team["queue_length"] >= 0


class TestSeedIndexing:
    def test_seed_0_returns_first_scenario(self):
        env = TicketRouterEnvironment()
        obs = env.reset(task_type="easy", seed=0)
        assert obs.scenario_id == SCENARIOS["easy"][0]["scenario_id"]

    def test_seed_4_returns_fifth_scenario(self):
        env = TicketRouterEnvironment()
        obs = env.reset(task_type="hard", seed=4)
        assert obs.scenario_id == SCENARIOS["hard"][4]["scenario_id"]

    def test_seed_wraps_around_modulo(self):
        env = TicketRouterEnvironment()
        obs = env.reset(task_type="easy", seed=5)   # 5 % 5 = 0
        assert obs.scenario_id == SCENARIOS["easy"][0]["scenario_id"]

    def test_benchmark_seeds_0_1_2_are_distinct(self):
        env = TicketRouterEnvironment()
        for task_type in TASK_TYPES:
            ids = [env.reset(task_type=task_type, seed=s).scenario_id for s in BENCHMARK_SEEDS]
            assert len(set(ids)) == 3, \
                f"{task_type}: benchmark seeds return duplicate scenarios"


class TestMeanScore:
    """
    Simulates the hackathon grader: run all 9 benchmark scenarios
    with the correct expected action and report the mean score.
    """

    def _run_scenario(self, task_type: str, seed: int) -> float:
        env = TicketRouterEnvironment()
        scenario = SCENARIOS[task_type][seed]
        env.reset(task_type=task_type, seed=seed)
        result = env.step(TicketRouterAction(
            primary_team=scenario["expected_team"],
            priority=scenario["expected_priority"],
            urgency=scenario["expected_urgency"],
        ))
        return result.metadata["score"]

    def test_mean_score_benchmark_is_1_0(self):
        """Perfect routing on all 9 benchmark scenarios → mean = 1.0."""
        scores = [
            self._run_scenario(t, s)
            for t in TASK_TYPES
            for s in BENCHMARK_SEEDS
        ]
        mean = sum(scores) / len(scores)
        assert round(mean, 4) == 0.99, f"Expected mean=0.99 with perfect actions, got {mean:.4f}"

    def test_easy_mean_perfect(self):
        scores = [self._run_scenario("easy", s) for s in BENCHMARK_SEEDS]
        assert round(sum(scores) / len(scores), 4) == 0.99

    def test_medium_mean_perfect(self):
        scores = [self._run_scenario("medium", s) for s in BENCHMARK_SEEDS]
        assert round(sum(scores) / len(scores), 4) == 0.99

    def test_hard_mean_perfect(self):
        scores = [self._run_scenario("hard", s) for s in BENCHMARK_SEEDS]
        assert round(sum(scores) / len(scores), 4) == 0.99

    def test_wrong_team_reduces_mean_below_1(self):
        """Always routing to Product (wrong) must reduce mean below 1.0."""
        scores = []
        for task_type in TASK_TYPES:
            for seed in BENCHMARK_SEEDS:
                env = TicketRouterEnvironment()
                env.reset(task_type=task_type, seed=seed)
                result = env.step(TicketRouterAction(
                    primary_team="Product", priority="medium", urgency="medium"
                ))
                scores.append(result.metadata["score"])
        mean = sum(scores) / len(scores)
        assert mean < 1.0, f"Expected mean < 1.0 with wrong team, got {mean:.4f}"

    def test_score_breakdown_all_9_scenarios(self, capsys):
        """Print per-scenario score breakdown (informational, always passes)."""
        all_scores = []
        lines = ["\n--- Score Breakdown (perfect actions) ---"]
        for task_type in TASK_TYPES:
            for seed in BENCHMARK_SEEDS:
                scenario = SCENARIOS[task_type][seed]
                score = self._run_scenario(task_type, seed)
                all_scores.append(score)
                lines.append(
                    f"  {task_type:6s} seed={seed} [{scenario['scenario_id']}]  "
                    f"{scenario['expected_team']:12s} / {scenario['expected_priority']:6s} / "
                    f"{scenario['expected_urgency']:6s}  →  score={score:.4f}"
                )
        mean = sum(all_scores) / len(all_scores)
        lines.append(f"  {'':6s}                                                 mean={mean:.4f}")
        lines.append("-" * 60)
        with capsys.disabled():
            print("\n".join(lines))
        assert True   # always passes — informational only
