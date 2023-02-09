#!/bin/bash

# Automate generation of test trajectories.

# Author: Spencer M. Richards
#         Autonomous Systems Lab (ASL), Stanford
#         (GitHub: spenrich)

for system in PlanarBirotor PlanarSpacecraft
do
    echo "system=$system"
    python trajectory_generation.py $system
done
