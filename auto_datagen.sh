#!/bin/bash

# Automate generation of test trajectories.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

for system in PlanarBirotor PlanarSpacecraft
do
    echo "system=$system"
    python data_generation.py $system --seed=151 --num_traj=1000 --freq=100 --zoh
done
