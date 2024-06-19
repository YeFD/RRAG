python finetune_retriever.py \
    --dataset_name musique \
    --train_data_path musique/musique_ans_v1.0_train.jsonl \
    --test_data_path musique/musique_ans_v1.0_dev.jsonl \
    --model_name google-bert/bert-base-uncased \
    --train_batch_size 32 \
    --num_epoch 4
