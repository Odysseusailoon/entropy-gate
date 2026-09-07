"""Shared answer parsing, distribution metrics, seeds and quality checks."""
import pytest
from egs.answer_parser import INVALID, normalize_number, parse_answer, parse_gold
from egs.net.config import Config
from egs.metrics import histogram, js_divergence
from egs.backend import seed_for


@pytest.mark.parametrize('raw, expected', [
    ('Final answer: $1,234.50.', '1234.5'), ('Final answer: -7', '-7'),
    ('Final answer: -1/2.', '-0.5'), ('Final answer: 2/6', '1/3'),
    ('Final answer: $-42', '-42'), ('Final answer: .125', '0.125'),
    ('Final answer: 0.0000001', '0.0000001'), ('Final answer: 9007199254740993', '9007199254740993'),
    ('**Final answer:** 42', '42'), ('Final answer: **42**', '42'),
    ('Final answer: 42\nFinal answer: nonsense', INVALID),
    ('Final answer: 42\nFinal answer:\n99', INVALID),
    ('Reasoning: 42', INVALID), ('Final answer 42', INVALID),
    ('Final answer: 1/0', INVALID), ('Final answer: 1/2/3', INVALID),
    ('Final answer: 42 apples', INVALID), ('Final answer: 1e3', INVALID),
    ('Final answer: 12,34', INVALID), ('Final answer: NaN', INVALID),
    ('Final answer: 3\nFinally, 9', '3'), ('Final answer: 3\nFinal answer: 4', '4'),
])
def test_parser(raw, expected):
    assert parse_answer(raw).answer == expected


def test_exact_rational_equivalence():
    assert normalize_number('5/2') == normalize_number('2.500') == '2.5'
    assert normalize_number('-0') == '0'
    assert parse_gold('x #### 1,234') == '1234'
    assert parse_answer('Final answer: 1\nFinal answer: nope').n_matches == 2


def test_metrics_keep_invalid_and_js_bits():
    assert histogram(['42', INVALID])[INVALID] == .5
    assert js_divergence({'a': 1}, {'b': 1}) == 1
    assert js_divergence({'a': 1}, {'a': 1}) == 0
    with pytest.raises(ValueError):
        js_divergence({'a': .8}, {'a': 1})
    with pytest.raises(ValueError):
        histogram(['a'], [-1])


def test_five_percent_invalid_stops_main_experiment():
    from egs.checks import quality
    assert not quality([INVALID] + ['42'] * 19, Config())['passed']
    assert quality([INVALID] + ['42'] * 20, Config())['passed']


def test_seed_namespaces_are_stable_and_separate():
    assert seed_for(1, "reference", 2) != seed_for(1, "iid", 2)
    assert seed_for(1, "iid", 2) == seed_for(1, "iid", 2)
