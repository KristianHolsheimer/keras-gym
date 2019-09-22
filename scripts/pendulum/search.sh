#!/bin/bash
trap "kill 0" EXIT

rm -rf /tmp/tensorboard

for id in `seq 1 10`; do
    CUDA_VISIBLE_DEVICES=-1 python3 -Wignore ppo.py \
        --id $id \
        --learning_rate 1e-3 \
        --entropy_beta 0.01 \
        --bootstrap_n 5 \
        --gamma 0.9 \
        --batch_size 512 \
        --minibatch_size 32 \
        --target_sync_tau 0.1 \
        --value_loss_weight 1 &
done

tensorboard --logdir /tmp/tensorboard 2> /dev/null

wait