
from huggingface_hub import create_repo, HfApi

repo_id = "WariHima/furigna-accent-whisper-v0.1-lora"
#create_repo(repo_id)

api = HfApi()
api.upload_folder(
    folder_path="./accent-whisper-ja-lora/checkpoint-2500",
    repo_id=repo_id,
)
