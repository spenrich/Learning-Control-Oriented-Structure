#!/bin/bash

# Automate testing of the PVTOL in tracking the double loop-the-loop.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

for seed in {0..4}
do
    for N in 50 100 200 500 1000
    do
        echo "seed=$seed, N=$N"
        python test_double_loop.py $seed $N --clip_ctrl --freq=150 --zoh
    done
done
