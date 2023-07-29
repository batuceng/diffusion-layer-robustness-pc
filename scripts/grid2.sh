#!/bin/bash
for i in {0..30..10}
do for j in {0..55..5}
do for k in {0..150..30}; 
do python test_pgd_attack.py --t-list $i,$j,$k --layers 0,1,2 --test-size 128;
done;done;done