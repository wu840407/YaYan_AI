"""YaYan_Reasoner / YaYan_Translator LLM 用戶端。

支援三個後端：
- transformers + bitsandbytes 4-bit  (預設，Turing 穩定)
- vllm in-process                    (較快，但會獨佔 process 的 CUDA_VISIBLE_DEVICES)
- openai-compatible (vLLM serve)     (推薦：LLM 與 ASR process 隔離，雙卡乾淨)

backend 設定於 configs/default.yaml -> llm.backend
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

import torch

from ..config import CONFIG, model_path, load_prompt

logger = logging.getLogger("YaYan.LLM")


class LlmClient:
    def __init__(self, alias: str = "YaYan_Reasoner"):
        self.alias = alias
        self.local_dir: Path = model_path(alias)
        if not self.local_dir.exists():
            raise FileNotFoundError(f"{alias} 不存在: {self.local_dir}")

        self.device = CONFIG["devices"]["llm_gpu"]
        self.backend = CONFIG["llm"]["backend"].lower()
        self._tokenizer = None
        self._model = None
        self._vllm = None
        self._oai = None
        self._load()

    # ------------------------------------------------------------------ #
    # Loaders
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        if self.backend == "vllm":
            self._load_vllm()
        elif self.backend == "openai":
            self._load_openai()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cfg = CONFIG["llm"]
        quant = (cfg.get("quantization") or "").lower()

        logger.info(f"載入 {self.alias}（transformers, quant={quant or 'none'}）…")

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.local_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: Dict = dict(
            device_map=self.device,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float16,  # Turing 不支援 bfloat16
        )

        if quant == "awq":
            # ★ 強制改走 GEMV（純 CUDA）路徑，避開 Turing 不支援的 triton 3.0 API
            from transformers import AwqConfig
            load_kwargs["quantization_config"] = AwqConfig(
                bits=4,
                version="gemv",        # 不要用 gemm（會走 triton）
                zero_point=True,
                group_size=128,
                do_fuse=False,         # 關掉 fused kernel，更穩
            )
        elif quant == "gptq":
            pass
        elif quant in ("4bit", "nf4"):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.local_dir), **load_kwargs
        )

    def _load_vllm(self) -> None:
        # 先嘗試 import；失敗早退，不要污染環境變數
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError(
                "vLLM 未安裝。請改 backend: transformers / openai，"
                "或安裝：pip install vllm==0.6.4.post1"
            ) from e

        # Turing 必設
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")
        # vLLM in-process 不認 device_map，靠 CUDA_VISIBLE_DEVICES 限制看到的卡
        # 副作用：同一 process 裡的 ASR 也會受影響，請改用 openai backend 避免衝突
        gpu_idx = self.device.split(":")[-1] if ":" in self.device else "0"
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx
            logger.warning(
                f"已設定 CUDA_VISIBLE_DEVICES={gpu_idx}（in-process vLLM 限制）；"
                "若 ASR 需用其他卡，請改用 backend: openai"
            )

        logger.info(f"載入 {self.alias}（vLLM in-process）…")
        cfg = CONFIG["llm"]
        quant = cfg.get("quantization")
        self._vllm = LLM(
            model=str(self.local_dir),
            quantization=quant if quant in ("awq", "gptq", "fp8") else None,
            dtype=cfg["dtype"],
            max_model_len=cfg["max_model_len"],
            gpu_memory_utilization=cfg["gpu_memory_utilization"],
            enforce_eager=cfg["enforce_eager"],
            tensor_parallel_size=cfg.get("tensor_parallel_size", 1),
            trust_remote_code=True,
        )

        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.local_dir),
            trust_remote_code=True,
            local_files_only=True,
        )

    def _load_openai(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai SDK 未安裝。請：pip install 'openai>=1.40.0'"
            ) from e

        cfg = CONFIG["llm"]
        base_url = cfg.get("openai_base_url", "http://127.0.0.1:8001/v1")
        served_name = cfg.get("served_model_name", "YaYan_Reasoner")
        logger.info(f"連線至 OpenAI-compatible server: {base_url} (model={served_name})")

        self._oai = OpenAI(base_url=base_url, api_key="dummy")
        self._served_name = served_name

        # 仍保留本地 tokenizer，需要時可用來計算 token 數 / 預組 chat template
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.local_dir),
            trust_remote_code=True,
            local_files_only=True,
        )

    # ------------------------------------------------------------------ #
    # Public chat
    # ------------------------------------------------------------------ #

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        cfg = CONFIG["llm"]
        max_new_tokens = max_new_tokens or cfg["max_new_tokens"]
        temperature = temperature if temperature is not None else cfg["temperature"]

        # OpenAI server 模式：直接走 messages，不在本地 render template
        if self.backend == "openai":
            return self._chat_openai(messages, max_new_tokens, temperature)

        # transformers / vllm in-process：本地 render，並關掉 Qwen3 thinking
        template_kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            text_input = self._tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_kwargs
            )
        except TypeError:
            # 非 Qwen3 系列模型沒有 enable_thinking 參數
            text_input = self._tokenizer.apply_chat_template(
                messages, **template_kwargs
            )

        if self.backend == "vllm":
            return self._chat_vllm(text_input, max_new_tokens, temperature)
        return self._chat_transformers(text_input, max_new_tokens, temperature)

    # ------------------------------------------------------------------ #
    # Backend implementations
    # ------------------------------------------------------------------ #

    def _chat_transformers(
        self, text_input: str, max_new_tokens: int, temperature: float
    ) -> str:
        cfg = CONFIG["llm"]
        inputs = self._tokenizer(
            [text_input], return_tensors="pt", padding=True
        ).to(self.device)

        do_sample = bool(cfg.get("do_sample", True)) and temperature > 0
        gen_kwargs: Dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.eos_token_id,
            attention_mask=inputs.attention_mask,
        )
        if do_sample:
            gen_kwargs.update(
                temperature=temperature,
                top_p=cfg.get("top_p", 0.9),
                top_k=cfg.get("top_k", 20),
                repetition_penalty=cfg.get("repetition_penalty", 1.05),
            )

        with torch.no_grad():
            output_ids = self._model.generate(inputs.input_ids, **gen_kwargs)

        new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _chat_vllm(
        self, text_input: str, max_new_tokens: int, temperature: float
    ) -> str:
        from vllm import SamplingParams
        cfg = CONFIG["llm"]
        params = SamplingParams(
            temperature=temperature,
            top_p=cfg.get("top_p", 0.9),
            top_k=cfg.get("top_k", 20),
            repetition_penalty=cfg.get("repetition_penalty", 1.05),
            max_tokens=max_new_tokens,
        )
        outputs = self._vllm.generate([text_input], params)
        return outputs[0].outputs[0].text.strip()

    def _chat_openai(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        cfg = CONFIG["llm"]
        resp = self._oai.chat.completions.create(
            model=self._served_name,
            messages=messages,
            temperature=temperature,
            top_p=cfg.get("top_p", 0.9),
            max_tokens=max_new_tokens,
            extra_body={
                "top_k": cfg.get("top_k", 20),
                "repetition_penalty": cfg.get("repetition_penalty", 1.05),
                # 關閉 Qwen3 thinking（vLLM serve 端要求 chat_template_kwargs）
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------------------ #
    # High-level helpers
    # ------------------------------------------------------------------ #

    def translate(self, raw_text: str, source_language: str) -> str:
        prompt = load_prompt("translate").replace("{source_language}", source_language)
        return self.chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": raw_text},
        ])

    def refine(self, raw_text: str, user_edit: str, source_language: str) -> str:
        prompt = (
            load_prompt("refine")
            .replace("{raw_text}", raw_text)
            .replace("{user_edit}", user_edit)
            .replace("{source_language}", source_language)
        )
        return self.chat([
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_edit},
        ])
