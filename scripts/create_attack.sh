#!/bin/bash
for atk in 'add' 'knn' 'cw'; 
do for mdel in 'curvenet';
do
echo $mdel $atk
python attack_dataset.py --batch_size 16 -model $mdel $atk  ;
done;done