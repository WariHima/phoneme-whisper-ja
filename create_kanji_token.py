from transformers import WhisperTokenizer
import re
from pathlib import Path

# Whisperのトークナイザをロード
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3-turbo", language="ja", task="transcribe")

# 正規表現で漢字を含むかどうかを判断
def contains_kanji(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

# トークナイザの語彙全体から漢字を含むトークンを抽出
kanji_tokens = []
for token_id in range(tokenizer.vocab_size):
    token_text = tokenizer.decode([token_id], skip_special_tokens=True)
    if contains_kanji(token_text):
        kanji_tokens.append((token_id, token_text))

# 結果表示（例：最初の10件）
for token_id, token_text in kanji_tokens[:10]:
    print(f"ID: {token_id}, Token: '{token_text}'")

token_ids = []
for token_id, token_text in kanji_tokens:
    token_ids.append(token_id)

Path("./kanji_tokens.txt").write_text("\n".join(map(str, token_ids)))


# 合計何個あるか
print(f"\nTotal kanji tokens: {len(kanji_tokens)}")
