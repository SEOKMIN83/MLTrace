#!/bin/bash

: << "USAGE"
./clustering_trace.sh
USAGE

for i in {1..6}  ## number of input:6
do
  for j in {1..10}  ## number of algorithms:10
  do
     valgrind --tool=callgrind --log-file=clustering/train/clustering-input$i-algorithm$j.txt --simulate-wb=yes python3 modify_plot_cluster_comparison.py $j $i 
  done
done


