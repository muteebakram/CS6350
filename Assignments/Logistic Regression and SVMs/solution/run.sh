#!/bin/sh

mkdir logs figs
rm logs/*

printf "1. SVM\n"
printf "******************************************************************\n"
python3 ./svm.py

printf "\n2. Logistic Regression\n"
printf "******************************************************************\n"
python3 ./logistic_regression.py

printf "\n3. SVM Over Trees\n"
printf "******************************************************************\n"
python3 ./svm_over_trees.py > logs/svm_over_trees.log
