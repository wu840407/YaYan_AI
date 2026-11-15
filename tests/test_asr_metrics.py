"""CER building blocks in scripts/evaluate_asr.py.

Every CER number in the README comes from these functions, so they are checked against an
independent reference implementation, not only against hand-picked examples.
"""
import random
from functools import lru_cache

import pytest

import evaluate_asr as ea


def reference_distance(a: str, b: str) -> int:
    """Textbook recursive Levenshtein distance, written independently of the code under test."""

    @lru_cache(maxsize=None)
    def d(i: int, j: int) -> int:
        if i == 0:
            return j
        if j == 0:
            return i
        return min(d(i - 1, j) + 1, d(i, j - 1) + 1, d(i - 1, j - 1) + (a[i - 1] != b[j - 1]))

    return d(len(a), len(b))


# ---------------------------------------------------------------- edit_distance


@pytest.mark.parametrize(
    "ref, hyp, expected",
    [
        ("今天天氣", "今天天氣", (0, 0, 0)),
        ("", "", (0, 0, 0)),
        ("", "你好", (0, 0, 2)),
        ("你好", "", (0, 2, 0)),
        ("今天天氣", "今天天汽", (1, 0, 0)),   # one substitution
        ("今天天氣好", "今天氣好", (0, 1, 0)),  # one deletion
        ("今天好", "今天很好", (0, 0, 1)),      # one insertion
    ],
)
def test_edit_distance_known_cases(ref, hyp, expected):
    assert ea.edit_distance(ref, hyp) == expected


def test_edit_distance_total_matches_reference_on_random_strings():
    rng = random.Random(0)
    alphabet = "ab甲乙"
    for _ in range(500):
        ref = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        hyp = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 12)))
        sub, dele, ins = ea.edit_distance(ref, hyp)
        assert sub + dele + ins == reference_distance(ref, hyp), (ref, hyp)
        # Every reference character is matched, substituted or deleted, and every
        # hypothesis character is matched, substituted or inserted.
        assert dele - ins == len(ref) - len(hyp), (ref, hyp)
        assert min(sub, dele, ins) >= 0


# ---------------------------------------------------------------- normalize


def test_normalize_removes_punctuation_and_spaces():
    assert ea.normalize("你好，世界！ 今天。") == "你好世界今天"


def test_normalize_empty_input():
    assert ea.normalize("") == ""
    assert ea.normalize(None) == ""


def test_normalize_drops_romanization_in_parentheses():
    # Common Voice Taiwanese transcripts write "Han characters (romanization)".
    assert ea.normalize("食飽（tsia̍h-pá）未") == "食飽未"
    assert ea.normalize("愛（Ài）") == "愛"  # upper-case tone-marked letters too


def test_normalize_keeps_chinese_inside_parentheses():
    assert ea.normalize("我（們）好") == "我們好"


def test_normalize_can_keep_romanization():
    out = ea.normalize("食飽（tsia̍h）", strip_romanization=False)
    assert out.startswith("食飽") and "tsia" in out


# ---------------------------------------------------------------- timestamp_error


def test_timestamp_error_on_matching_words():
    ref = [{"text": "今", "start": 0.0}, {"text": "天", "start": 0.5}, {"text": "好", "start": 1.0}]
    hyp = [{"text": "今", "start": 0.1}, {"text": "天", "start": 0.6}, {"text": "好", "start": 1.1}]
    out = ea.timestamp_error(ref, hyp)
    assert out["matched_words"] == 3
    assert out["mean_abs_err_s"] == pytest.approx(0.1)


def test_timestamp_error_only_matches_in_order():
    # The hypothesis has an extra word; matching must not jump backwards.
    ref = [{"text": "好", "start": 1.0}, {"text": "天", "start": 2.0}]
    hyp = [{"text": "天", "start": 0.0}, {"text": "好", "start": 1.0}, {"text": "天", "start": 2.5}]
    out = ea.timestamp_error(ref, hyp)
    assert out["matched_words"] == 2
    assert out["mean_abs_err_s"] == pytest.approx(0.25)


def test_timestamp_error_returns_none_without_matches():
    assert ea.timestamp_error([], [{"text": "a", "start": 0}]) is None
    assert ea.timestamp_error([{"text": "甲", "start": 0}], [{"text": "乙", "start": 0}]) is None


# ---------------------------------------------------------------- line prefixes


def test_strip_line_prefix():
    text = "[SPEAKER_00 00:01-00:05] 你好\n[00:06-00:09] 再見"
    assert ea.strip_line_prefix(text) == "你好\n再見"
    assert ea.strip_line_prefix("") == ""


# ---------------------------------------------------------------- --set overrides


def test_set_override_parses_yaml_values():
    pairs = ea._parse_set(["asr.lid_backend=firered", "asr.enable_vad=false", "audio.max_chunk_seconds=20"])
    cfg = {"asr": {"lid_backend": "voxlingua", "enable_vad": True}, "audio": {"max_chunk_seconds": 30}}
    ea._apply_set(cfg, pairs)
    assert cfg == {"asr": {"lid_backend": "firered", "enable_vad": False}, "audio": {"max_chunk_seconds": 20}}


def test_set_override_rejects_typos():
    cfg = {"asr": {"lid_backend": "voxlingua"}}
    with pytest.raises(SystemExit):
        ea._apply_set(cfg, ea._parse_set(["asr.lid_backnd=firered"]))   # unknown key
    with pytest.raises(SystemExit):
        ea._apply_set(cfg, ea._parse_set(["asrr.lid_backend=firered"]))  # unknown section
    with pytest.raises(SystemExit):
        ea._parse_set(["asr.lid_backend"])                               # no "="
    assert cfg == {"asr": {"lid_backend": "voxlingua"}}
