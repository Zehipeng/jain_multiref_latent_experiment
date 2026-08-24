from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceCandidate:
    candidate_index: int
    prompt_index: int
    prompt: str
    generation_seed: int


def build_candidate_schedule(
    prompts: list[str], generation_seeds: list[int], max_candidates: int
) -> list[ReferenceCandidate]:
    """Pair candidate seeds with prompts, cycling prompts when needed."""
    if not prompts:
        raise ValueError("At least one reference prompt is required")
    if not generation_seeds:
        raise ValueError("At least one generation seed is required")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    candidate_count = min(len(generation_seeds), max_candidates)
    return [
        ReferenceCandidate(
            candidate_index=index,
            prompt_index=index % len(prompts),
            prompt=prompts[index % len(prompts)],
            generation_seed=int(generation_seeds[index]),
        )
        for index in range(candidate_count)
    ]


def accepts_reference(
    p_value: float | None, threshold: float, require_detected: bool
) -> bool:
    if not require_detected:
        return True
    return p_value is not None and p_value <= threshold
