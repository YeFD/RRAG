python finetune_retriever.py \
    --dataset_name hotpotqa \
    --train_data_path hotpotqa/hotpot_train_v1.1_bert.pkl \
    --test_data_path hotpotqa/hotpot_dev_distractor_v1_bert.pkl \
    --model_name google-bert/bert-base-uncased \
    --train_batch_size 32 \
    --num_epoch 4
