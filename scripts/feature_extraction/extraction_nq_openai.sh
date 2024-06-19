python feature_extraction_openai.py \
    --dataset_name nq_10 \
    --input_path 10_total_documents/nq-open-10_total_documents_gold_at_0.jsonl.gz \
    --model_name text-embedding-3-large \
    --emb_save_path dataset/nq_10_openai_embedding_large.pkl \
    --dataset_save_path dataset/nq-open-10_total_documents_gold_openai_large.pkl