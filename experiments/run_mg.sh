#!/bin/env bash

# Run the MackeyGlass system experiments
#TAUS=`ls -1 dynamical/data/MackeyGlass/`
TAUS="tau_1 tau_9 tau_17 tau_25 tau_33 tau_41 tau_49 tau_57"

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

      for beta in "0.5 0.25 0.1"; do
         echo "Running Mackey-Glass system with d shift-add and sparsity"
         python3 run_experiment.py --experiment mackey_glass --tau $tau --sparse_relu=True --beta=$beta
      done
   done
done