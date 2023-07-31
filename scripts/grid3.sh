#!/bin/bash

for i in {0..200..5}
do python test_attack.py --t-list $i,0,0,0 --layers 0,1,2,3 -model dgcnn;
done

for j in {0..200..5}
do python test_attack.py --t-list 0,$j,0,0 --layers 0,1,2,3 -model dgcnn;
done

for k in {0..200..5}
do python test_attack.py --t-list 0,0,$k,0 --layers 0,1,2,3 -model dgcnn;
done

for l in {0..200..5}
do python test_attack.py --t-list 0,0,0,$l --layers 0,1,2,3 -model dgcnn;
done