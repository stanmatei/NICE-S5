#!/bin/env bash

# Run the MackeyGlass system experiments
tau=tau_1
echo "Running Mackey-Glass system witn no discretization"
python3 run_experiment.py --experiment mackey_glass --tau $tau
echo "Running Mackey-Glass system with b shift-add"
python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_b=True
echo "Running Mackey-Glass system with c shift-add"
python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_c=True