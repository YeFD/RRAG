python runner.py \
    --dataset_name musique \
    --train_data_path retrieval/dataset/musique_ans_v1.0_train_bert.pkl \
    --test_data_path retrieval/dataset/musique_ans_v1.0_dev_bert.pkl \
    --model_name meta-llama/Llama-2-7b-hf \
    --use_training\
    --freeze_llm \
    --save_model \
    --output_dir your/rrag/path/Rrag-Llama-2-7b \
    --num_k 20 \
    --use_rrag \
    --use_evaluation \
    --save_results