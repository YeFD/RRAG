python feature_extraction.py \
    --dataset_name hotpotqa \
    --train_data_path hotpotqa/hotpot_train_v1.1.json \
    --test_data_path hotpotqa/hotpot_dev_distractor_v1.json \
    --save_train_path dataset/hotpot_train_v1.1_bert.pkl \
    --save_test_path dataset/hotpot_dev_distractor_v1_bert.pkl \
    --model_name google-bert/bert-base-uncased