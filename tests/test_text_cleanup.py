"""Output clean-up: LLM artifacts (yayan/llm/postprocess.py) and ASR repetition loops
(yayan/observability.py)."""
from yayan import observability
from yayan.llm import postprocess

collapse = observability.collapse_repetition


# ---------------------------------------------------------------- LLM artifacts


def test_strip_think_block_and_polite_prefix():
    assert postprocess.strip_llm_artifacts("<think>推理過程</think>\n好的，今天天氣很好") == "今天天氣很好"


def test_strip_code_fence():
    assert postprocess.strip_llm_artifacts("```text\n今天開會\n```") == "今天開會"


def test_strip_translation_preamble():
    assert postprocess.strip_llm_artifacts("以下是翻譯：你好") == "你好"


def test_strip_leaves_normal_text_alone():
    assert postprocess.strip_llm_artifacts("今天開會，明天放假。") == "今天開會，明天放假。"
    assert postprocess.strip_llm_artifacts("") == ""


def test_to_taiwan_traditional_without_converter(monkeypatch):
    monkeypatch.setattr(postprocess, "_get_converter", lambda: None)
    assert postprocess.to_taiwan_traditional("<think>x</think>今天") == "今天"


def test_to_taiwan_traditional_uses_converter(monkeypatch):
    class FakeConverter:
        def convert(self, text):
            return text.replace("信息", "資訊")

    monkeypatch.setattr(postprocess, "_get_converter", lambda: FakeConverter())
    assert postprocess.to_taiwan_traditional("好的，信息") == "資訊"


# ---------------------------------------------------------------- repetition loops


def test_long_loop_is_cut_to_one_copy():
    out, cut = collapse("謝謝大家" * 30 + "結束")
    assert (out, cut) == ("謝謝大家結束", True)


def test_single_character_loop_keeps_two():
    out, cut = collapse("啊" * 60 + "好")
    assert (out, cut) == ("啊啊好", True)


def test_short_repetition_is_left_alone():
    assert collapse("哈哈哈") == ("哈哈哈", False)
    text = "你好" * 20  # 40 characters of content: below the 50-character threshold
    assert collapse(text) == (text, False)


def test_numbers_are_never_collapsed():
    for text in ("1234" * 20, "九五二七" * 20, "one two " * 20):
        assert collapse(text) == (text, False), text
