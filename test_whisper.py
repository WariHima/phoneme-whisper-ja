from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from peft import PeftModel, PeftConfig
import torch

def infer_with_peft_lora_whisper(
    model_name_or_path: str,
    lora_adapter_path: str,
    audio_file_path: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    PEFT LoRAでファインチューニングされたWhisperモデルを読み込み、音声ファイルからテキストを推論します。

    Args:
        model_name_or_path (str): オリジナルのWhisperモデルのパスまたはHugging FaceモデルID
                                  例: "openai/whisper-small"
        lora_adapter_path (str): LoRAアダプターのパス
                                 例: "./my_whisper_lora_model" (保存したLoRAアダプターのディレクトリ)
        audio_file_path (str): 推論したい音声ファイルのパス
        device (str): 使用するデバイス ("cuda" または "cpu")
    """
    print(f"使用デバイス: {device}")

    # 1. オリジナルのWhisperモデルとプロセッサをロード
    print(f"オリジナルのWhisperモデルをロード中: {model_name_or_path}")
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path)

    # 2. LoRAアダプターをロード
    print(f"LoRAアダプターをロード中: {lora_adapter_path}")
    # PeftConfigをロードすることで、ベースモデルのどの層にLoRAを適用したかなどの情報が得られる
    # config = PeftConfig.from_pretrained(lora_adapter_path)
    model = PeftModel.from_pretrained(model, lora_adapter_path)

    # モデルを評価モードに設定し、指定されたデバイスに移動
    model.eval()
    model.to(device)
    print("モデルとアダプターのロードが完了しました。")

    # 3. 音声ファイルをロードして前処理
    print(f"音声ファイルをロード中: {audio_file_path}")
    # 音声ファイルをロードするためのダミーデータ（実際にはlibrosaなどを使用）
    # ここでは例としてダミーのwaveformを使用します。
    # 実際の使用では、以下のように音声ファイルを読み込みます:
    # import librosa
    # audio, sampling_rate = librosa.load(audio_file_path, sr=16000)
    # input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_features

    # デモンストレーションのためのダミー音声データ
    # 実際の音声ファイルがない場合、適当な形状のテンソルで代用
    # 注意: 実際の音声ファイルを使ってください
    # 例: 16kHzで5秒の音声
    dummy_audio_waveform = torch.randn(1, 16000 * 5)
    input_features = processor(dummy_audio_waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    print("音声ファイルの前処理が完了しました。")

    # 4. 推論を実行
    print("推論を実行中...")
    with torch.no_grad():
        generated_ids = model.generate(input_features, max_new_tokens=128) # max_new_tokensは生成するトークンの最大数
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print("\n--- 推論結果 ---")
    print(f"転写されたテキスト: {transcription}")
    
    return transcription

if __name__ == "__main__":
    # --- 使用例 ---
    # 以下のパスは実際の環境に合わせて変更してください。
    
    # 1. オリジナルのWhisperモデルの指定
    # 小さなモデルから試すことをお勧めします
    # 例えば "openai/whisper-tiny", "openai/whisper-small" など
    BASE_MODEL = "openai/whisper-large-v3-turbo" 
    
    # 2. LoRAアダプターを保存したディレクトリのパス
    # これはあなたがLoRAでファインチューニングしたモデルを保存したディレクトリです。
    # 例: trainer.save_model("my_whisper_lora_model") で保存した場合のディレクトリ名
    LORA_ADAPTER_DIR = "./accent-whisper-ja-lora/checkpoint-190" # 仮のパス、ご自身のパスに変更してください
    
    # 3. 推論したい音声ファイルのパス
    # 実際の音声ファイルを用意してください。
    # 例えば、サンプルとしてHugging Face Datasetsから取得するか、
    # 自分で録音した短い音声ファイルなど
    AUDIO_FILE = "./dataset_example/wav/vsm_jvnv_f2_0.wav" 

    # 注意: `AUDIO_FILE`が存在しない場合、ダミーデータで実行されますが、
    # 実際の推論結果は得られません。`librosa`などで音声ファイルを読み込む部分を
    # 実際のコードに置き換える必要があります。
    
    try:
        # この部分でダミーファイル作成（実行時に毎回作る必要はありません）
        # 実際の推論には不要ですが、librosaなしで動作確認するため
        import soundfile as sf
        import numpy as np
        import os
        if not os.path.exists(AUDIO_FILE):
            print(f"警告: 音声ファイル '{AUDIO_FILE}' が見つかりません。ダミーデータを使用します。")
            # 16kHzで5秒間のサイン波を生成
            samplerate = 16000
            duration = 5 # seconds
            frequency = 440 # Hz
            t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
            data = 0.5 * np.sin(2. * np.pi * frequency * t)
            sf.write(AUDIO_FILE, data, samplerate)
            print(f"ダミー音声ファイル '{AUDIO_FILE}' を作成しました。")
            
        
        if not os.path.exists(LORA_ADAPTER_DIR):
            print(f"警告: LoRAアダプターディレクトリ '{LORA_ADAPTER_DIR}' が見つかりません。")
            print("LoRAアダプターがトレーニングされ、このパスに保存されていることを確認してください。")
            print("このまま実行するとエラーになる可能性があります。")
            
        # 推論実行
        transcribed_text = infer_with_peft_lora_whisper(
            model_name_or_path=BASE_MODEL,
            lora_adapter_path=LORA_ADAPTER_DIR,
            audio_file_path=AUDIO_FILE
        )
        
    except ImportError:
        print("\n--- 依存関係のインストールのお願い ---")
        print("以下のコマンドで必要なライブラリをインストールしてください:")
        print("pip install transformers peft accelerate bitsandbytes datasets librosa soundfile")
        print("特に librosa と soundfile は音声ファイルの読み込みに必要です。")
        print("もしCUDAを使用する場合は accelerate と bitsandbytes も推奨されます。")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("LoRAアダプターのパスが正しいか、モデルが正しく保存されているか確認してください。")
        print("また、`audio_file_path`が実際に存在する音声ファイルであるか確認してください。")