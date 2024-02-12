#!/bin/sh

printf "1. Simple Perceptron\n"
printf "******************************************************************\n\n"
python3 ./simple_perceptron.py

printf "\n2. Decaying Learning Rate\n"
printf "******************************************************************\n\n"
python3 ./decaying_perceptron.py

printf "\n3. Margin Perceptron\n"
printf "******************************************************************\n\n"
python3 ./margin_perceptron.py
