#!/bin/env bash

tau=tau_1
echo "Running Mackey-Glass system with d shift-add"
python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_d=True
echo "Running Mackey-Glass system with mlp shift-add"
python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_mlp=True
echo "Running Mackey-Glass system with ALL shift-add"
python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_b=True --shift_add_c=True --shift_add_d=True --shift_add_mlp=True
echo "Running Mackey-Glass system with ALL shift-add"
python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_b=True --shift_add_d=True --shift_add_mlp=True