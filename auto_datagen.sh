#!/bin/bash

# Automate generation of training and test trajectories.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

for system in PlanarBirotor PlanarSpacecraft
do
    # Generate trajectory training data
    echo "Generating training trajectories (system=$system)"
    python trajectory_generation.py $system --seed=151 --num_traj=1000 --sim --freq=100 --zoh

    # Generate test trajectories
    echo "Generating test trajectories (system=$system)"
    python trajectory_generation.py $system --seed=42 --num_traj=100
done
