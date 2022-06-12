#!/bin/bash

: << "USAGE"
./classification_trace.sh 
USAGE


for i in {1..3} ## number of input: 3
do 
  for j in {1..10} ## umber of algorithms: 10
  do
    valgrind --tool=callgrind --log-file=classification/train/classification-input$i-algorithm$j.txt --simulate-wb=yes python3 modify_plot_classifier_comparison.py $j $i
  done
done
