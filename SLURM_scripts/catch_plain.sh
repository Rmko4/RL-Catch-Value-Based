#!/bin/bash

for i in {2..4}; do
    run_name="train${i}"
    echo "Training $run_name"

    python train_agent.py \
        --run_name "$run_name" \
        --max_epochs 100 \
        --batch_size 128 \
        --batches_per_step 1 \
        --optimizer RMSprop \
        --learning_rate 0.001 \
        --gamma 0.99 \
        --epsilon_start 0.5 \
        --epsilon_end 0.01 \
        --epsilon_decay_rate 200 \
        --buffer_capacity 10000 \
        --replay_warmup_steps 10 \
        --soft_update_tau 0.05 \
        --hidden_size 256 \
        --n_filters 16 \
        --algorithm Dueling_architecture

    echo "Training $run_name completed"
done
