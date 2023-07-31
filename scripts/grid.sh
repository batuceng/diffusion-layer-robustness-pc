#!/bin/bash
for i in {0..80..10}
do for j in {0..80..10}
do for k in {0..80..10}; 
do python test_pgd_attack.py --t-list $i,$j,$k,0 --layers 0,1,2,3;
done;done;done