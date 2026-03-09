from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence



@dataclass(frozen=True)
class VPRState:
    """Compact state used by the MVP prior model."""

    elapsed_time_s: float
    gcd_delta_s: float
    in_burst_window: bool
    last_action: str


@dataclass(frozen=True)
class VPRActionSample:
    state: VPRState
    action: str


@dataclass(frozen=True)
class VPRGenerationConfig:
    horizon: int = 24
    top_k: int = 3
    default_fallback_action: str = "Steel Fangs"


class VPRFrequencyPriorModel:
    """A tiny imitation-learning baseline using conditional action frequencies."""

    def __init__(self):
        self._global_counts: Counter[str] = Counter()
        self._counts_by_context: dict[tuple[str, bool], Counter[str]] = defaultdict(Counter)

    @staticmethod
    def _context_key(state: VPRState) -> tuple[str, bool]:
        return (state.last_action, state.in_burst_window)

    def fit(self, samples: Sequence[VPRActionSample]) -> None:
        self._global_counts.clear()
        self._counts_by_context = defaultdict(Counter)

        for sample in samples:
            key = self._context_key(sample.state)
            self._counts_by_context[key][sample.action] += 1
            self._global_counts[sample.action] += 1

    def top_k_actions(self, state: VPRState, k: int = 3) -> list[str]:
        key = self._context_key(state)
        context_counts = self._counts_by_context.get(key)
        if context_counts and len(context_counts) > 0:
            return [action for action, _ in context_counts.most_common(k)]
        return [action for action, _ in self._global_counts.most_common(k)]


def _get_burst_window(elapsed_time_s: float) -> bool:
    return (elapsed_time_s % 120.0) <= 20.0


class RotationRowLike(Protocol):
    t: float
    skill_name: str
    job_class: str | None


def build_vpr_samples_from_rows(rows: Iterable[RotationRowLike], player_job: str = "VPR") -> list[VPRActionSample]:
    samples: list[VPRActionSample] = []
    previous_t: float | None = None
    last_action = "<START>"

    for row in rows:
        row_job = row.job_class if row.job_class is not None and row.job_class != "" else player_job
        if row_job != player_job:
            continue

        t = float(row.t)
        gcd_delta_s = 0.0 if previous_t is None else max(0.0, t - previous_t)

        state = VPRState(
            elapsed_time_s=t,
            gcd_delta_s=gcd_delta_s,
            in_burst_window=_get_burst_window(t),
            last_action=last_action,
        )
        samples.append(VPRActionSample(state=state, action=row.skill_name))

        previous_t = t
        last_action = row.skill_name

    return samples


def build_vpr_samples_from_csv(csv_path: str, player_job: str = "VPR") -> list[VPRActionSample]:
    from ama_xiv_combat_sim.simulator.rotation_import_utils.csv_utils import CSVUtils

    meta_fields, skiprows = CSVUtils.read_meta_fields(csv_path)
    rows = CSVUtils.read_rotation_from_csv(csv_path, skiprows=skiprows)
    if "stats" in meta_fields:
        # metadata is parsed by CSVUtils in the main sim path; this function only needs action rows.
        pass
    return build_vpr_samples_from_rows(rows, player_job=player_job)


def generate_vpr_action_plan(
    model: VPRFrequencyPriorModel,
    seed_actions: Sequence[str],
    start_time_s: float,
    gcd_s: float,
    config: VPRGenerationConfig = VPRGenerationConfig(),
) -> list[str]:
    actions = list(seed_actions)
    last_action = actions[-1] if actions else "<START>"

    for i in range(config.horizon):
        t = start_time_s + i * gcd_s
        state = VPRState(
            elapsed_time_s=t,
            gcd_delta_s=gcd_s,
            in_burst_window=_get_burst_window(t),
            last_action=last_action,
        )
        candidates = model.top_k_actions(state, k=config.top_k)
        if len(candidates) == 0:
            next_action = config.default_fallback_action
        else:
            next_action = candidates[0]

        actions.append(next_action)
        last_action = next_action

    return actions


def evaluate_action_plan_expected_damage(
    actions: Sequence[str],
    stats,
    skill_library,
    num_samples: int = 100000,
    enable_autos: bool = True,
) -> float:
    from ama_xiv_combat_sim.simulator.damage_simulator import DamageSimulator
    from ama_xiv_combat_sim.simulator.timeline_builders.damage_builder import DamageBuilder
    from ama_xiv_combat_sim.simulator.timeline_builders.rotation_builder import RotationBuilder

    rb = RotationBuilder(
        stats,
        skill_library,
        ignore_trailing_dots=True,
        enable_autos=enable_autos,
        use_strict_skill_naming=False,
    )
    for action in actions:
        rb.add_next(action)

    db = DamageBuilder(stats, skill_library)
    sim = DamageSimulator(stats, db.get_damage_instances(rb.get_skill_timing()), num_samples)
    return float(sim.get_expected_damage())
