#!/bin/bash

for i in {0..200..5}
do python test_pgd_attack.py --t-list $i,0,0 --layers 0,1,2;
done

for j in {0..200..5}
do python test_pgd_attack.py --t-list 0,$j,0 --layers 0,1,2;
done

for k in {0..200..5}
do python test_pgd_attack.py --t-list 0,0,$k --layers 0,1,2;
done
