#!/bin/bash

bs=256
echo "model: $1, attack: $atck, batchsize: $bs";

for atck in "pgd" "pgdl2" "cw" "add" "knn" "drop";
do
    for l0 in {0..200..5}
    do 
        python test_distance.py -t_list $l0,0,0,0,0 --model $1 --attack $atck --batch-size $bs
    done

    for l1 in {0..200..5}
    do 
        python test_distance.py -t_list 0,$1,0,0,0 --model $1 --attack $atck --batch-size $bs
    done

    for l2 in {0..200..5}
    do 
        python test_distance.py -t_list 0,0,$l2,0,0 --model $1 --attack $atck --batch-size $bs
    done

    for l3 in {0..200..5}
    do 
        python test_distance.py -t_list 0,0,0,$l3,0 --model $1 --attack $atck --batch-size $bs
    done

    if [ "$1" = "pointnet" ] || [ "$1" = "pointnet2" ]
    then
        echo ""
    else
        for l4 in {0..200..5}
        do 
            python test_distance.py -t_list 0,0,0,0,$l4 --model $1 --attack $atck --batch-size $bs
        done
    fi
done