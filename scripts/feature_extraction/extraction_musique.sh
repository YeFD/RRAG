python feature_extraction.py \
    --dataset_name musique \
    --train_data_path musique/musique_ans_v1.0_train.jsonl \
    --test_data_path musique/musique_ans_v1.0_dev.jsonl \
    --save_train_path dataset/musique_ans_v1.0_train_bert.pkl \
    --save_test_path dataset/musique_ans_v1.0_dev_bert.pkl \
    --model_name google-bert/bert-base-uncased