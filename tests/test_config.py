"""Configuration files and the prompt-override mechanism in yayan/config.py."""
import pytest

from yayan import config


def test_every_referenced_model_alias_exists():
    aliases = config.ALIASES["models"]
    asr = config.CONFIG["asr"]
    referenced = {asr["default_alias"], config.CONFIG["llm"]["alias"], *asr["routing"].values()}
    missing = sorted(a for a in referenced if a not in aliases)
    assert not missing, missing


def test_model_path_rejects_unknown_alias():
    with pytest.raises(KeyError):
        config.model_path("No_Such_Model")


def test_model_path_uses_models_root(monkeypatch):
    monkeypatch.setitem(config.CONFIG["paths"], "models_root", "/tmp/models")
    expected = config.ALIASES["models"]["YaYan_ASR_Dialect"]["local_dir"]
    assert config.model_path("YaYan_ASR_Dialect").as_posix() == f"/tmp/models/{expected}"


@pytest.mark.parametrize("name", ["translate", "refine"])
def test_bundled_prompts_load(name, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_PROMPTS_DIR", tmp_path / "none")
    assert config.load_prompt(name).strip()


def test_local_prompt_overrides_bundled_one(monkeypatch, tmp_path):
    (tmp_path / "translate.txt").write_text("local prompt", encoding="utf-8")
    monkeypatch.setattr(config, "LOCAL_PROMPTS_DIR", tmp_path)
    assert config.load_prompt("translate") == "local prompt"
    assert config.load_prompt("refine") != "local prompt"  # other prompts are unaffected


def test_missing_prompt_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "LOCAL_PROMPTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError):
        config.load_prompt("no_such_prompt")


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("YAYAN_MODELS_ROOT", "/srv/models")
    monkeypatch.setenv("YAYAN_SERVER_PORT", "9999")
    cfg = config._override_from_env({"paths": {"models_root": "/data/ai_models"}, "server": {"port": 7860}})
    assert cfg["paths"]["models_root"] == "/srv/models"
    assert cfg["server"]["port"] == 9999
