#!/bin/env bash

# Run the MackeyGlass system experiments
TAUS=`ls -1 dynamical/data/MackeyGlass/`

for tau in $TAUS; do
   echo "Running Mackey-Glass system with b shift-add"
   python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_b=True
   echo "Running Mackey-Glass system with c shift-add"
   python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_c=True
   echo "Running Mackey-Glass system with d shift-add"
   python3 run_experiment.py --experiment mackey_glass --tau $tau --shift_add_d=True
done

# TAUS="tau_1 tau_128 tau_255"
# for tau in $TAUS; do
#    for a_bits in $A_BITS; do
#       for ssm_act_bits in $SSM_ACT_BITS; do
#         echo "Running Mackey-Glass system with a_bits=$a_bits, ssm_act_bits=$ssm_act_bits, tau=$tau"
#         python3 run_experiment.py --a_bits $a_bits --ssm_act_bits $ssm_act_bits --experiment mackey_glass --tau $tau
#       done
#    done
# done
