python finetune_retriever.py \
    --dataset_name 2wiki \
    --train_data_path 2wikimultihop/train.json \
    --test_data_path 2wikimultihop/dev.json \
    --model_name google-bert/bert-base-uncased \
    --train_batch_size 32 \
    --num_epoch 4
