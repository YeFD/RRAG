python feature_extraction.py \
    --dataset_name 2wiki \
    --train_data_path 2wikimultihop/train.json \
    --test_data_path 2wikimultihop/dev.json \
    --save_train_path dataset/2wikimultihop_train_bert.pkl \
    --save_test_path dataset/2wikimultihop_dev_bert.pkl \
    --model_name google-bert/bert-base-uncased