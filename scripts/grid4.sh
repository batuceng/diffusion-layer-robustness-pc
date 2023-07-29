#!/bin/bash
for i in {0..30..3}
do for j in {0..30..3}
do for k in {50..80..3}; 
do python test_pgd_attack.py --t-list $i,$j,$k --layers 0,1,2;
done;done;done