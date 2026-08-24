import pytest

from rmlp.reference_bank import accepts_reference, build_candidate_schedule


def test_candidate_schedule_cycles_prompts_and_preserves_seeds() -> None:
    schedule = build_candidate_schedule(
        prompts=["first", "second"], generation_seeds=[4, 8, 15], max_candidates=3
    )
    assert [candidate.prompt for candidate in schedule] == ["first", "second", "first"]
    assert [candidate.generation_seed for candidate in schedule] == [4, 8, 15]


def test_candidate_schedule_respects_max_candidates() -> None:
    schedule = build_candidate_schedule(
        prompts=["prompt"], generation_seeds=[0, 1, 2], max_candidates=2
    )
    assert [candidate.generation_seed for candidate in schedule] == [0, 1]


@pytest.mark.parametrize(
    ("p_value", "required", "expected"),
    [
        (0.05, True, True),
        (0.0500001, True, False),
        (None, True, False),
        (None, False, True),
    ],
)
def test_accepts_reference(
    p_value: float | None, required: bool, expected: bool
) -> None:
    assert accepts_reference(p_value, threshold=0.05, require_detected=required) is expected
