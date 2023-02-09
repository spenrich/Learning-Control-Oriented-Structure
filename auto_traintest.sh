#!/bin/bash

# Automate training and testing.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

for system in PlanarBirotor PlanarSpacecraft
do
    for seed in {0..4}
    do
        for N in 50 100 200 500 1000
        do
            echo "system=$system, seed=$seed, N=$N"
            python train.py $system $seed $N --epochs=1000 --holdout_frac=0.1 --sample_locally
            python test.py $system $seed $N --clip_ctrl --zoh
            echo
        done
    done
done
