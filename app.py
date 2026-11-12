import os
import sys

# ==========================================
# 0. 強制設定模型路徑 (一定要在 import torch 之前)
# ==========================================
# 設定 Hugging Face 模型快取路徑為當前專案底下的 models_cache
os.environ["HF_HOME"] = os.path.abspath("./models_cache")

import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    BitsAndBytesConfig
)

# --- 硬體檢查 ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 系統啟動中... 使用裝置: {device}")
print(f"📂 模型儲存路徑: {os.environ['HF_HOME']}")

# ==========================================
# 1. 載入 ASR 模型 (Whisper-Large-v3)
# ==========================================
print("⏳ 正在載入 Whisper-Large-v3 (語音識別)...")
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device=device,
)

# ==========================================
# 2. 載入 LLM 模型 (Qwen2.5-7B)
# ==========================================
print("⏳ 正在載入 Qwen2.5-7B (翻譯與潤飾)...")

llm_model_id = "Qwen/Qwen2.5-7B-Instruct"

# 4-bit 量化配置 (3090 省顯存關鍵)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_id,
    quantization_config=bnb_config, 
    device_map="auto",              
    trust_remote_code=True
)

# ==========================================
# 3. 定義核心處理邏輯
# ==========================================
def process_audio(audio_path, source_dialect, target_style):
    if audio_path is None:
        return "請先錄音或上傳檔案！", ""

    print(f"🎤 收到音訊: {audio_path} | 來源: {source_dialect}")
    
    # --- 步驟 A: ASR 識別 ---
    # 對於維吾爾語，我們可以嘗試讓 whisper 自動偵測，或是強制指定 "ug"
    # 但為了通用性，這裡使用自動偵測模式 (task="transcribe")
    try:
        asr_result = asr_pipe(
            audio_path, 
            generate_kwargs={"task": "transcribe"},
            return_timestamps=True
        )
        raw_text = asr_result["text"]
        print(f"📝 Whisper 識別結果: {raw_text}")
    except Exception as e:
        return f"識別錯誤: {str(e)}", ""

    # --- 步驟 B: LLM 翻譯/潤飾 ---
    system_instruction = f"""
    你是由「雅言 (YaYan-AI)」專案開發的方言轉換專家。
    
    你的核心任務是：
    1. 接收使用者的語音識別文字，原文語言是「{source_dialect}」。
    2. 理解其語意，並將其精確轉換為優雅、標準的「{target_style}」。
    3. 如果原文是維吾爾語，請將其翻譯為流暢的正體中文。
    4. 修正語音識別可能產生的同音錯字或贅字。
    
    請注意：輸出必須精準、流暢且符合正體中文規範。直接輸出轉換結果即可。
    """

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"語音識別原文：{raw_text}"}
    ]

    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.3,
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    final_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return raw_text, final_response

# ==========================================
# 4. 建立 Gradio 介面
# ==========================================
with gr.Blocks(title="YaYan-AI 雅言系統") as demo:
    gr.Markdown("# 🏺 YaYan-AI (雅言) - 本地化方言轉換系統")
    gr.Markdown("基於 RTX 3090 | Whisper-Large-v3 | Qwen-2.5-7B")
    
    with gr.Row():
        with gr.Column(scale=1):
            # 錄音輸入
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="請按此說話或上傳檔案")
            
            # 選項 (已新增維吾爾語)
            dialect_dropdown = gr.Dropdown(
                choices=["台灣口語/台灣國語", "廣東話 (粵語)", "四川話", "上海話", "維吾爾語", "其他方言"], 
                value="台灣口語/台灣國語", 
                label="輸入語言 (來源)"
            )
            style_dropdown = gr.Radio(
                choices=["標準新聞書面語 (正體)", "流暢口語 (正體)", "精簡摘要"], 
                value="標準新聞書面語 (正體)", 
                label="輸出風格"
            )
            
            submit_btn = gr.Button("開始轉換 🚀", variant="primary")

        with gr.Column(scale=1):
            # 輸出區
            raw_text_output = gr.Textbox(label="Whisper 原始識別", lines=3, interactive=False)
            final_text_output = gr.Textbox(label="雅言 AI 轉換結果", lines=5, interactive=False)

    # 綁定事件
    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, dialect_dropdown, style_dropdown],
        outputs=[raw_text_output, final_text_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)