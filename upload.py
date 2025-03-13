from huggingface_hub import upload_folder

upload_folder(
    folder_path="arxiv-summarization",
    repo_id="Talina06/arxiv-summarization",
    commit_message="Uploading fine-tuned summarization model based on google/flan-t5-small"
)

upload_folder(
    folder_path="arxiv-search",
    repo_id="Talina06/arxiv-search",
    commit_message="Uploading fine-tuned model based on sentence-transformers/all-MiniLM-L6-v2"
)
