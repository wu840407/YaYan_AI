"""Engine routing in yayan/asr/router.py.

The real engines need GPUs and model weights, so they are replaced with stubs that report
which engine was chosen. What is tested is the decision, which is the part the README's
routing results depend on.
"""
import numpy as np
import pytest

from yayan.asr import router

AUDIO = np.zeros(1600, dtype=np.float32)


@pytest.fixture
def engines(monkeypatch):
    calls = []

    def stub(name):
        def run(audio, routing, chunk_start_sec):
            calls.append((name, routing))
            return name

        return run

    monkeypatch.setattr(router, "_dolphin_transcribe", stub("dolphin"))
    monkeypatch.setattr(router, "_whisper_transcribe", stub("whisper"))
    monkeypatch.setattr(router, "_taigi_transcribe", stub("taigi"))
    asr = router.CONFIG["asr"]
    monkeypatch.setitem(asr, "enable_taigi_asr", False)
    monkeypatch.setitem(asr, "force_whisper_routings", [])
    monkeypatch.setitem(asr, "taigi_routings", ["nan", "nan-cs", "nan-hn"])
    return calls


def test_auto_and_empty_go_to_dialect_engine(engines):
    assert router.transcribe(AUDIO, "auto") == "dolphin"
    assert router.transcribe(AUDIO, None) == "dolphin"
    assert router.transcribe(AUDIO, "") == "dolphin"


def test_routing_key_is_case_insensitive(engines):
    assert router.transcribe(AUDIO, "ZH") == "dolphin"
    assert engines[-1] == ("dolphin", "zh")


@pytest.mark.parametrize("key", sorted(router.DOLPHIN_ROUTINGS))
def test_every_dialect_key_goes_to_dialect_engine(engines, key):
    assert router.transcribe(AUDIO, key) == "dolphin"


@pytest.mark.parametrize("key", ["ja", "ko", "en", "fr", "vi", "ar"])
def test_other_languages_go_to_whisper(engines, key):
    assert router.transcribe(AUDIO, key) == "whisper"


def test_taiwanese_engine_when_enabled(engines, monkeypatch):
    monkeypatch.setitem(router.CONFIG["asr"], "enable_taigi_asr", True)
    assert router.transcribe(AUDIO, "nan") == "taigi"
    assert router.transcribe(AUDIO, "zh") == "dolphin"


def test_taiwanese_engine_failure_falls_back_to_dialect_engine(engines, monkeypatch):
    def broken(audio, routing, chunk_start_sec):
        raise RuntimeError("weights missing")

    monkeypatch.setitem(router.CONFIG["asr"], "enable_taigi_asr", True)
    monkeypatch.setattr(router, "_taigi_transcribe", broken)
    assert router.transcribe(AUDIO, "nan") == "dolphin"


@pytest.mark.parametrize("value", [["yue"], "yue", [" YUE "]])
def test_force_whisper_list_overrides_dialect_engine(engines, monkeypatch, value):
    monkeypatch.setitem(router.CONFIG["asr"], "force_whisper_routings", value)
    assert router.transcribe(AUDIO, "yue") == "whisper"
    assert router.transcribe(AUDIO, "wuu") == "dolphin"


def test_default_taiwanese_keys_when_config_is_empty(monkeypatch):
    monkeypatch.setitem(router.CONFIG["asr"], "taigi_routings", None)
    assert router._taigi_routings() == router._DEFAULT_TAIGI_ROUTINGS


def test_routing_table_in_config_agrees_with_router():
    """configs/default.yaml documents a routing table; it must match what the router does."""
    table = router.CONFIG["asr"]["routing"]
    for key, alias in table.items():
        goes_to_dialect = key in router.DOLPHIN_ROUTINGS or key == "auto"
        if alias == "YaYan_ASR_Dialect":
            assert goes_to_dialect, key
        elif alias == "YaYan_ASR_Global":
            assert not goes_to_dialect, key
        else:
            pytest.fail(f"unexpected alias {alias!r} for {key!r}")
