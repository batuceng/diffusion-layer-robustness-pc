#!/bin/bash
for i in {0..30..5}
do for j in {0..20..5}
do for k in {40..80..5}; 
do python test_attack.py --t-list $i,$j,$k --layers 0,1,2 -model dgcnn;
done;done;done