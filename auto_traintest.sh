#!/bin/bash

# Automate training and testing.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

for system in PlanarBirotor PlanarSpacecraft
do
    for seed in {0..4}
    do
        # Using data uniformly sampled from the state and input spaces
        echo "========== With uniformly sampled data =========="
        for N in 50 100 200 500 1000
        do
            echo "system=$system, seed=$seed, N=$N"
            python train.py $system $seed $N --epochs=50000 --sample_locally
            python test.py $system $seed $N --clip_ctrl --zoh
            echo
        done

        # Using data in the form of trajectories
        echo "============= With trajectory data =============="
        for N in 1 2 5 10 20 50 100
        do
            echo "system=$system, seed=$seed, N=$N"
            python train.py $system $seed $N --epochs=50000 --sample_locally --traj
            python test.py $system $seed $N --clip_ctrl --zoh --traj
            echo
        done
    done
done
