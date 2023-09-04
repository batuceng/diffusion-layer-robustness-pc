#!/bin/bash
for l in {0..100..10}; 
do for k in {0..100..10};
do for j in {0..100..10};
do for i in {0..100..10};
do python test_distance.py -t_list $i,$j,$k,$l;
done;done;done;done;