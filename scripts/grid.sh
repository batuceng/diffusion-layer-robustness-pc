#!/bin/bash

bs=256
echo "model: $1, attack: $2, batchsize: $bs";

for l0 in {0..100..10}
do 
    for l1 in {0..100..10}
    do 
        for l2 in {0..100..10}
        do 
            for l3 in {0..100..10}
            do 
                if [ "$1" = "pointnet" ] || [ "$1" = "pointnet2" ]
                then
                    python test_distance.py -t_list $l0,$l1,$l2,$l3,0 --model $1 --attack $2 --batch-size $bs
                else
                    for l4 in {0..100..10}
                    do 
                        python test_distance.py -t_list $l0,$l1,$l2,$l3,$l4 --model $1 --attack $2 --batch-size $bs
                    done
                fi
            done
        done
    done
done