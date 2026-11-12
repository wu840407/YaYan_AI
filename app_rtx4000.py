import os
import sys
import torch
import gradio as gr
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline, 
    BitsAndBytesConfig
)

# ==========================================
# 0. 強制設定模型路徑
# ==========================================
# 設定 Hugging Face 模型快取路徑
os.environ["HF_HOME"] = os.path.abspath("./models_cache")

# ==========================================
# 1. 硬體資源分配 (雙卡核心邏輯)
# ==========================================
# 檢查是否有兩張顯卡
if torch.cuda.device_count() >= 2:
    print(f"🚀 偵測到雙顯卡環境！啟動雙卡分工模式...")
    device_asr = "cuda:0"  # 第一張卡負責聽 (Whisper)
    device_llm = "cuda:1"  # 第二張卡負責想 (Llama)
else:
    print(f"⚠️ 警告：僅偵測到單卡，將使用混合模式...")
    device_asr = "cuda:0"
    device_llm = "cuda:0"

print(f"📂 模型儲存路徑: {os.environ['HF_HOME']}")
print(f"🎤 ASR Device: {device_asr}")
print(f"🧠 LLM Device: {device_llm}")

# ==========================================
# 設定離線模型路徑 (指向您的大硬碟)
# ==========================================
# 假設您已經把模型下載到 /data/ai_models/ 裡面
OFFLINE_MODEL_PATH_LLM = "/data/ai_models/Llama-3.1-8B-Instruct"
OFFLINE_MODEL_PATH_WHISPER = "openai/whisper-large-v3" 
# 注意: Whisper 如果要離線，建議也先下載到 /data/ai_models/whisper-large-v3 
# 然後把上面改成 "/data/ai_models/whisper-large-v3"

# ==========================================
# 2. 載入 ASR 模型 (Whisper-Large-v3) -> GPU 0
# ==========================================
print(f"⏳ 正在 GPU 0 載入 Whisper-Large-v3...")
try:
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=OFFLINE_MODEL_PATH_WHISPER,  # <--- 這裡可以是本地路徑
        # model="openai/whisper-large-v3",
        torch_dtype=torch.float16,
        device=device_asr, # 指定第一張卡
    )
except Exception as e:
    print(f"❌ Whisper 載入失敗: {e}")
    sys.exit(1)
    
# ==========================================
# 3. 載入 LLM (Llama 3.1) - 完全離線讀取
# ==========================================
print(f"⏳ 正在 GPU 1 載入 Llama 3.1 (讀取路徑: {OFFLINE_MODEL_PATH_LLM})...")

# 檢查路徑是否存在，避免報錯
if not os.path.exists(OFFLINE_MODEL_PATH_LLM):
    print(f"❌ 錯誤：找不到模型路徑 {OFFLINE_MODEL_PATH_LLM}")
    print("請確認您已將模型下載到該資料夾，或暫時開啟網路下載。")
    sys.exit(1)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(
    OFFLINE_MODEL_PATH_LLM,  # <--- 直接讀本地資料夾
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    OFFLINE_MODEL_PATH_LLM,  # <--- 直接讀本地資料夾
    quantization_config=bnb_config, 
    device_map=device_llm,
    trust_remote_code=True,
    local_files_only=True      # <--- 關鍵！強制不連網
)

# ==========================================
# 4. 定義核心處理邏輯
# ==========================================
def process_audio(audio_path, source_dialect, target_style):
    if audio_path is None:
        return "請先錄音或上傳檔案！", ""

    print(f"🎤 收到音訊: {audio_path} | 來源: {source_dialect}")
    
    # --- 步驟 A: ASR 識別 (GPU 0) ---
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

    # --- 步驟 B: LLM 翻譯/潤飾 (GPU 1) ---
    # Llama 3 的 System Prompt 寫法
    system_instruction = f"""
    You are an expert dialect translator named 'YaYan-AI'.
    
    Your task is to:
    1. Receive text transcribed from '{source_dialect}'.
    2. Translate and refine it into '{target_style}'.
    3. If the source is Uyghur, translate it into standard Traditional Chinese.
    4. Correct any homophone errors from the ASR process.
    
    Output ONLY the translated text in Traditional Chinese. Do not explain.
    """

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Original Text: {raw_text}"}
    ]

    # 使用 apply_chat_template 自動處理 Llama 3 的特殊標籤
    text_input = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # 注意：這裡要把 inputs 搬到 device_llm (GPU 1)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device_llm)

    generated_ids = llm_model.generate(
        model_inputs.input_ids,
        max_new_tokens=1024,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    final_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return raw_text, final_response

# ==========================================
# 5. 建立 Gradio 介面
# ==========================================
with gr.Blocks(title="YaYan-AI 雅言系統 (Server Edition)") as demo:
    gr.Markdown("# 🏺 YaYan-AI (雅言) - 離線部署版")
    gr.Markdown("Based on **Dual RTX 4000 Ada** | **雅言 YaYan 自主研發語音辨識模型** | **雅言 YaYan 自主研發翻譯模型**")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="錄音輸入")
            
            dialect_dropdown = gr.Dropdown(
                choices=["台灣口語/台灣國語", "廣東話 (粵語)", "四川話", "上海話", "維吾爾語", "其他方言"], 
                value="台灣口語/台灣國語", 
                label="來源語言"
            )
            style_dropdown = gr.Radio(
                choices=["標準內容摘要 (正體)", "逐字精準翻譯", "語意重點分析"],
                value="標準內容摘要 (正體)",
                label="輸出模式"
            )
            
            submit_btn = gr.Button("開始分析 🚀", variant="primary")

        with gr.Column(scale=1):
            raw_text_output = gr.Textbox(label="Whisper 原始轉錄 (GPU 0)", lines=3, interactive=False)
            final_text_output = gr.Textbox(label="Llama 3.1 研判結果 (GPU 1)", lines=5, interactive=False)

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, dialect_dropdown, style_dropdown],
        outputs=[raw_text_output, final_text_output]
    )

if __name__ == "__main__":
    # Server 版通常需要開啟 share=False 並且綁定 0.0.0.0
    demo.launch(server_name="0.0.0.0", server_port=7860)