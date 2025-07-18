import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset
from pydub import AudioSegment
from transformers import (
    Seq2SeqTrainer,
    Trainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

from mora_list import __MORA_LIST_MINIMUM, __MORA_LIST_ADDITIONAL

# --- ユーティリティ関数 ---
def is_wav_file(file: Path) -> bool:
    """指定されたファイルが.wav拡張子を持っているかを確認します。"""
    return file.suffix.lower() == ".wav"


def is_lab_file(file: Path) -> bool:
    """指定されたファイルが.lab拡張子を持っているかを確認します。"""
    return file.suffix.lower() == ".lab"


# --- PyOpenJTalk G2P プロソディ抽出 ---
# プロソディ特徴量を抽出するための正規表現パターン
_PYOPENJTALK_G2P_PROSODY_A1_PATTERN = re.compile(r"/A:([0-9\-]+)\+")
_PYOPENJTALK_G2P_PROSODY_A2_PATTERN = re.compile(r"\+(\d+)\+")
_PYOPENJTALK_G2P_PROSODY_A3_PATTERN = re.compile(r"\+(\d+)/")
_PYOPENJTALK_G2P_PROSODY_E3_PATTERN = re.compile(r"!(\d+)_")
_PYOPENJTALK_G2P_PROSODY_F1_PATTERN = re.compile(r"/F:(\d+)_")
_PYOPENJTALK_G2P_PROSODY_P3_PATTERN = re.compile(r"\-(.*?)\+")

CUR_REMAP_LIST = __MORA_LIST_MINIMUM + __MORA_LIST_ADDITIONAL 
REMAP_LIST = []

for i in CUR_REMAP_LIST: 
    if i[1] == "":
        phone = " " + i[2]
    else:
        phone = " " + " ".join( ( i[1],i[2]) )
    REMAP_LIST.append( (phone , i[0]) )


def _numeric_feature_by_regex(pattern: re.Pattern[str], s: str) -> int:
    """正規表現パターンを使用して数値特徴量を抽出するヘルパー関数。"""
    match = pattern.search(s)
    if match is None:
        return -50  # 特徴量が見つからない場合のデフォルト値
    return int(match.group(1))


def pyopenjtalk_g2p_prosody(text: str, drop_unvoiced_vowels: bool = True) -> list[str]:
    """
    入力された完全文脈ラベルから音素とプロソディ記号のシーケンスを抽出します。

    このアルゴリズムは、`Prosodic features control by symbols as input of
    sequence-to-sequence acoustic modeling for neural TTS`_ に基づいており、r9y9の調整が加えられています。
    (ESPnetの実装からの引用)。

    Args:
        text (str): 入力テキスト（完全文脈ラベル文字列、.labファイルの内容を想定）。
        drop_unvoiced_vowels (bool): 無声母音を削除するかどうか（小文字に変換）。

    Returns:
        List[str]: 音素とプロソディ記号のリスト。

    .. _`Prosodic features control by symbols as input of sequence-to-sequence acoustic
        modeling for neural TTS`: https://doi.org/10.1587/transinf.2020EDP7104
    """
    labels = text.splitlines()  # .labファイルが複数行であると想定
    N = len(labels)
    phones = []

    for n in range(N):
        lab_curr = labels[n]

        # 現在の音素の抽出
        cur_phone_match = _PYOPENJTALK_G2P_PROSODY_P3_PATTERN.search(lab_curr)
        if cur_phone_match is None:
            warnings.warn(f"ラベルから音素を抽出できませんでした: {lab_curr}")
            continue

        p3 = cur_phone_match.group(1)

        # 無声母音の処理
        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        # 'sil' と 'pau' 音素の処理
        if p3 == "sil":
            if n == 0:
                continue
                #phones.append("^")  # 発話の開始
            elif n == N - 1:
                e3 = _numeric_feature_by_regex(_PYOPENJTALK_G2P_PROSODY_E3_PATTERN, lab_curr)
                if e3 == 0:
                    continue
                    #phones.append("$")  # 発話の終了
                elif e3 == 1:
                    phones.append("?")  # 疑問形
            continue
        elif p3 == "pau":
            #phones.append("_")  # ポーズ
            phones.append("①")  # ポーズ
            continue
        else:
            phones.append(p3)  # 通常の音素

        # アクセントタイプと位置情報の抽出
        a1 = _numeric_feature_by_regex(_PYOPENJTALK_G2P_PROSODY_A1_PATTERN, lab_curr)
        a2 = _numeric_feature_by_regex(_PYOPENJTALK_G2P_PROSODY_A2_PATTERN, lab_curr)
        a3 = _numeric_feature_by_regex(_PYOPENJTALK_G2P_PROSODY_A3_PATTERN, lab_curr)
        f1 = _numeric_feature_by_regex(_PYOPENJTALK_G2P_PROSODY_F1_PATTERN, lab_curr)

        # 次の音素のa2を先読み（ラベルの末尾でない場合）
        a2_next = -50
        if n + 1 < N:
            a2_next = _numeric_feature_by_regex(_PYOPENJTALK_G2P_PROSODY_A2_PATTERN, labels[n + 1])

        # プロソディ記号の追加
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNcl":
            phones.append("#")  # アクセント句の境界
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            #phones.append("]")  # ピッチ下降
            phones.append("↓")  # ピッチ下降

        elif a2 == 1 and a2_next == 2:
            #phones.append("[")  # ピッチ上昇
            phones.append("↑")  # ピッチ上昇
            
    phone_text = f' {" ".join(phones)}'

    for i in REMAP_LIST:
        phone_text = phone_text.replace(i[0], i[1])

    phone_text_list = phone_text.split(" ")
    phone_text = "".join(phone_text_list)

    print(phone_text)
    return phone_text
    #return phones


# --- データコレーター ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    動的なパディングを伴う音声-シーケンス変換タスク用のデータコレーター。
    バッチ内の最長シーケンスに合わせて、入力特徴量（音声）とラベル（トークン化されたテキスト）の両方をパディングします。
    """

    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # 入力音声特徴量のパディング
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # トークン化されたラベルのパディング
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 損失計算で無視されるように、パディングトークンのIDを-100に置き換える
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # ラベルの先頭にBOSトークンがある場合は削除
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# --- データセットの準備 ---
def prepare_dataset(batch: Dict[str, Any], processor: WhisperProcessor) -> Dict[str, Any]:
    """
    Whisperモデル用のデータバッチを準備します。
    音声をロードしてリサンプリングし、ログメル特徴量を計算し、ターゲットテキストをトークン化します。
    """
    # 音声データのロードとリサンプリング
    audio = batch["audio"]

    # ログメル入力特徴量の計算
    input_features = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    # ターゲットテキストをラベルIDにエンコード
    labels = processor.tokenizer(batch["sentence"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
        "length": len(input_features) # input_featuresの長さを追加
    }


def compute_metrics(pred, processor: WhisperProcessor, metric_wer: Any):
    """
    単語誤り率（WER）メトリクスを計算します。
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # 正しいデコードのために、-100をpad_token_idに置き換える
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # 予測とラベルをデコード
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # WERを計算
    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# --- メインスクリプト ---
def main():
    # --- 設定 ---
    # 入力ディレクトリの定義
    WAV_DIR = Path("./0_999/wav")
    LAB_DIR = Path("./0_999/lab")
    OUTPUT_DIR = "accent-whisper-ja-lora"
    MODEL_NAME = "efwkjn/whisper-ja-anime-v0.3"

    # トレーニング引数
    TRAIN_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 1
    LEARNING_RATE = 1e-5
    WARMUP_STEPS = 5
    MAX_STEPS = 1000
    SAVE_EVAL_STEPS = 10
    LOGGING_STEPS = 25
    EVAL_BATCH_SIZE = 8
    GENERATION_MAX_LENGTH = 225
    LOAD_IN_8BIT = True

    # --- データロードと準備 ---
    print("データセットをロードして準備しています...")
    wav_files = sorted([file for file in WAV_DIR.rglob("*") if is_wav_file(file)])
    lab_files = sorted([file for file in LAB_DIR.rglob("*") if is_lab_file(file)])

    if len(wav_files) != len(lab_files):
        raise ValueError("WAVファイルとLABファイルの数が一致しません。")

    dataset_list = []
    for i in range(len(wav_files)):
        wavfile = wav_files[i]
        labfile = lab_files[i]

        # LABファイルの内容を読み込み、pyopenjtalk_g2p_prosodyで処理
        lab_content = labfile.read_text(encoding="utf-8")
        text = pyopenjtalk_g2p_prosody(lab_content)
        text = " ".join(text)

        # soundfileで音声データをNumPy配列としてロードし、pydubでフレームレートを取得
        data, samplerate = sf.read(str(wavfile))
        # Whisperのfeature extractorが期待するように、データをfloat32に変換
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # AudioSegmentを使用してframe_rateを簡単に取得。sf.readもsamplerateを返す
        # sf.readからのsamplerateを優先する（実際の音声データのレートであるため）
        sound = AudioSegment.from_file(wavfile, "wav")
        actual_samplerate = sound.frame_rate # soundfileのsamplerateと一致するはず

        dataset_list.append(
            {"audio": {"array": data, "sampling_rate": actual_samplerate}, "sentence": text}
        )

    # Hugging Face Datasetの作成
    common_voice = Dataset.from_list(dataset_list)

    # オーディオ列を希望のサンプリングレートにキャスト（Whisperは16kHzを期待）
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

    # プロセッサ、メトリクスの初期化
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language="Japanese", task="transcribe"
    )
    metric = evaluate.load("wer")

    # 準備関数でデータセットをマップ
    # processorとmetricをcompute_metricsに渡すためにラムダ式を使用
    common_voice = common_voice.map(
        lambda batch: prepare_dataset(batch, processor),
        num_proc=1,  # システムの能力に応じてnum_procを調整
        # remove_columns引数は、元の列を削除したい場合に利用（例：remove_columns=common_voice.column_names）
    )

    # 訓練/検証セットの分割（最初はcommon_voiceが単一の分割であると想定）
    # データが小さい場合、簡単な分割が必要になる場合があります。
    # デモンストレーションのため、全データセットを「train」として使用し、一部を「validation」として使用します。
    # 実際のシナリオでは、専用の訓練/検証セットがあるか、明示的に分割します。
    if "train" not in common_voice.column_names and "validation" not in common_voice.column_names:
        print("訓練/検証セットを分割中（80/20）...")
        common_voice = common_voice.train_test_split(test_size=0.2)


    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # --- モデルの初期化 ---
    print("モデルを初期化しています...")
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME,  load_in_4bit=LOAD_IN_8BIT)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ja", task="transcribe"
    )
    
    model.config.suppress_tokens = []

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model) 
    config = LoraConfig(r=32, lora_alpha=64,
                        target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
    
    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    # --- トレーニング引数 ---
    print("トレーニング引数を設定しています...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_checkpointing=True,
        fp16=True,
        group_by_length=True,
        eval_strategy="steps",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        predict_with_generate=True,
        generation_max_length=GENERATION_MAX_LENGTH,
        save_steps=SAVE_EVAL_STEPS,
        eval_steps=SAVE_EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    # --- トレーナーの初期化とトレーニング ---
    print("トレーナーを初期化し、トレーニングを開始しています...")
    trainer = Seq2SeqTrainer(
    #trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"], # train_test_splitにより"test"が検証セットとして使用される
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor, metric),
        tokenizer=processor.feature_extractor, # テキスト処理には`processor.tokenizer`を使用すべきだが、DataCollatorの`processor.feature_extractor.pad`からすると、現状でも問題ない可能性あり
    )

    trainer.train()
    print("トレーニング完了！")

if __name__ == "__main__":
    main()