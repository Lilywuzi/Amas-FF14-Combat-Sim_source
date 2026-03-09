import unittest
from types import SimpleNamespace

from ama_xiv_combat_sim.rotation_solver.vpr_mvp_solver import (
    VPRFrequencyPriorModel,
    VPRGenerationConfig,
    build_vpr_samples_from_rows,
    generate_vpr_action_plan,
)


class TestVPRMVPSolver(unittest.TestCase):
    def test_build_samples_filters_non_vpr(self):
        rows = [
            SimpleNamespace(t=0.0, skill_name="Steel Fangs", job_class="VPR"),
            SimpleNamespace(t=2.5, skill_name="Dreadwinder", job_class="VPR"),
            SimpleNamespace(t=3.0, skill_name="Battle Litany", job_class="DRG"),
        ]

        samples = build_vpr_samples_from_rows(rows, player_job="VPR")
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].state.last_action, "<START>")
        self.assertEqual(samples[1].state.last_action, "Steel Fangs")

    def test_frequency_prior_and_generation(self):
        rows = [
            SimpleNamespace(t=0.0, skill_name="Steel Fangs", job_class="VPR"),
            SimpleNamespace(t=2.5, skill_name="Dreadwinder", job_class="VPR"),
            SimpleNamespace(t=5.0, skill_name="Steel Fangs", job_class="VPR"),
        ]
        samples = build_vpr_samples_from_rows(rows, player_job="VPR")

        model = VPRFrequencyPriorModel()
        model.fit(samples)

        actions = generate_vpr_action_plan(
            model,
            seed_actions=["Steel Fangs"],
            start_time_s=7.5,
            gcd_s=2.5,
            config=VPRGenerationConfig(horizon=3, top_k=2),
        )

        self.assertEqual(len(actions), 4)
        self.assertTrue(all(isinstance(x, str) for x in actions))


if __name__ == "__main__":
    unittest.main()
