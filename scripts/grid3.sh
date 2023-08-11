#!/bin/bash

for i in {0..200..5}
do python test_distance.py -t_list $i,0,0,0 --model dgcnn;
done

for j in {0..200..5}
do python test_distance.py -t_list 0,$j,0,0 --model dgcnn;
done

for k in {0..200..5}
do python test_distance.py -t_list 0,0,$k,0 --model dgcnn;
done

for l in {0..200..5}
do python test_distance.py -t_list 0,0,0,$l --model dgcnn;
done