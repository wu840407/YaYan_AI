"""Scoring helpers in scripts/evaluate_llm.py and scripts/evaluate_translation.py."""
import random

import pytest

import evaluate_asr as ea
import evaluate_llm as el
import evaluate_translation as et


def test_llm_edit_distance_agrees_with_asr_edit_distance():
    # The two scripts implement Levenshtein separately; they must agree.
    rng = random.Random(1)
    for _ in range(300):
        a = "".join(rng.choice("abc丙丁") for _ in range(rng.randint(0, 15)))
        b = "".join(rng.choice("abc丙丁") for _ in range(rng.randint(0, 15)))
        assert el.edit_distance(a, b) == sum(ea.edit_distance(a, b)), (a, b)


def test_score_one_flags_format_and_wording_problems():
    src = "[00:01-00:05] 這個資訊\n[00:06-00:09] 好"
    out = "以下是結果：[00:01-00:05] 这个信息"
    res = el.score_one(src, out, terms=["資訊", "好"])
    assert res["tag_keep"] == pytest.approx(0.5)    # one of two timestamp tags survived
    assert res["line_match"] == pytest.approx(0.5)  # one of two lines survived
    assert res["has_preamble"] is True
    assert res["cn_terms"] == ["信息"]
    assert res["simplified_chars"] == 2             # 这、个
    assert res["term_hit"] == pytest.approx(0.0)


def test_score_one_on_a_clean_output():
    src = "[00:01-00:05] 今天開會"
    res = el.score_one(src, src, terms=[])
    assert res["tag_keep"] == 1.0
    assert res["line_match"] == 1.0
    assert res["edit_ratio"] == 0.0
    assert res["term_hit"] is None
    assert res["has_preamble"] is False


def test_chrf_bounds():
    assert et.chrf("今天天氣很好", "今天天氣很好") == pytest.approx(1.0)
    assert et.chrf("今天天氣", "明後雨雪") == 0.0
    assert et.chrf("", "今天") == 0.0
    partial = et.chrf("今天天氣很好", "今天天氣不好")
    assert 0.0 < partial < 1.0


def test_recall():
    assert et.recall("abcd", "abxd") == pytest.approx(0.75)
    assert et.recall("", "abc") == 0.0


def test_translation_normalize_drops_tags_and_punctuation():
    assert et.normalize("[00:01-00:05] Hello，世界!") == "hello世界"
