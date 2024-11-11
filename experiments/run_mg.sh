#!/bin/env bash

# Run the MackeyGlass system experiments
TAUS=`ls -1 dynamical/data/MackeyGlass/`

for r in 1 2 3 4 5; do # Repeat the experiment 5 times
   for tau in $TAUS; do # For each tau
      echo "Running Mackey-Glass system baseline"
      python3 run_experiment.py --experiment mackey_glass --tau $tau
      echo "Running Mackey-Glass system with b shift-add"
      python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_b=True
      echo "Running Mackey-Glass system with c shift-add"
      python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_c=True
      echo "Running Mackey-Glass system with d shift-add"
      python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_mlp=True
   done
done