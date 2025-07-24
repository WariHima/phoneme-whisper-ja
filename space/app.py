import gradio as gr
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# suppress_tokensの読み込み
def load_suppress_tokens(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [int(token) for token in f.read().split()]

suppress_tokens = load_suppress_tokens("./whisper-ja-anime-v0.3/kanji_tokens.txt") 
print("suppress_tokens loaded")

# 初始化组件
def load_components():
    processor = WhisperProcessor.from_pretrained(
        "efwkjn/whisper-ja-anime-v0.3",
        #"openai/whisper-large-v3-turbo", 
        language="Japanese",
        task="transcribe"
    )
    
    model = WhisperForConditionalGeneration.from_pretrained(
        "WariHIma/furigna-accent-whisper-v0.1-lora", 
        device_map="auto" if torch.cuda.is_available() else "cpu"
    )
    
    # 设置强制解码参数
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="japanese", 
        task="transcribe"
    )
    model.config.forced_decoder_ids = forced_decoder_ids
    model.eval()
    
    return processor, model

processor, model = load_components()

# 语音转文字函数
def transcribe_audio(audio_path):
    try:
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        
        # 提取特征
        inputs = processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(model.device)
        
        # 生成预测
        with torch.no_grad():
            generated_ids = model.generate(inputs, max_length=256, suppress_tokens=suppress_tokens)
        
        # 解码结果
        text = processor.tokenizer.batch_decode(
            generated_ids, 
            suppress_tokens=suppress_tokens,
            skip_special_tokens=True
        )[0]
        
        return text
    
    except Exception as e:
        return f"Error: {str(e)}"

# 创建Gradio界面
demo = gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.Textbox(label="output"),
    title="ASR",
    description="furigana-accent-whisper demo, hf space code from https://huggingface.co/spaces/AkitoP/whisper-japanese-prosodic-jsut5000_only",
    allow_flagging="never"
)

demo.launch()