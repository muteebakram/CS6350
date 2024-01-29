#!/bin/sh

printf "Baseline\n"
printf "******************************************************************\n"

python3 ./baseline.py

printf "\n\nFull Trees\n"
printf "******************************************************************\n"

python3 ./full_trees.py

printf "\n\nLimiting Depth\n"
printf "******************************************************************\n"

python3 ./limiting_depth.py

printf "\n\nGrad only: Attribute cost\n"
printf "******************************************************************\n"

python3 ./grad_only.py
