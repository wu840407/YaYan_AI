"""集中載入 YaYan-AI 設定檔，並提供模型別名查詢。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(filename: str) -> Dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"設定檔不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _override_from_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    env_root = os.environ.get("YAYAN_MODELS_ROOT")
    if env_root:
        cfg["paths"]["models_root"] = env_root
    env_input = os.environ.get("YAYAN_INPUT_DIR")
    if env_input:
        cfg["paths"]["input_dir"] = env_input
    env_output = os.environ.get("YAYAN_OUTPUT_DIR")
    if env_output:
        cfg["paths"]["output_dir"] = env_output
    env_port = os.environ.get("YAYAN_SERVER_PORT")
    if env_port:
        cfg["server"]["port"] = int(env_port)
    return cfg


CONFIG: Dict[str, Any] = _override_from_env(_load_yaml("default.yaml"))
ALIASES: Dict[str, Any] = _load_yaml("model_aliases.yaml")


def models_root() -> Path:
    return Path(CONFIG["paths"]["models_root"])


def model_path(alias: str) -> Path:
    if alias not in ALIASES["models"]:
        raise KeyError(f"未知的模型別名: {alias}")
    local_dir = ALIASES["models"][alias]["local_dir"]
    return models_root() / local_dir


def real_id(alias: str) -> str:
    return ALIASES["models"][alias]["real_id"]


def public_name(alias: str) -> str:
    return alias


# 本機覆寫的提示詞（不進 git）：部署端要用不同措辭時放這裡，同名檔優先於 configs/prompts/。
# 可用環境變數 YAYAN_LOCAL_PROMPTS_DIR 指到別處；目錄不存在就一律用預設提示詞。
LOCAL_PROMPTS_DIR = Path(os.environ.get("YAYAN_LOCAL_PROMPTS_DIR") or (CONFIG_DIR / "local" / "prompts"))


def load_prompt(name: str) -> str:
    local = LOCAL_PROMPTS_DIR / f"{name}.txt"
    if local.is_file():
        return local.read_text(encoding="utf-8")
    path = CONFIG_DIR / "prompts" / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt 檔不存在: {path}")
    return path.read_text(encoding="utf-8")
