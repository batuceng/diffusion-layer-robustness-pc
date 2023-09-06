#!/bin/bash
for atk in 'add' 'cw' 'drop' 'knn' 'pgd' 'pgdl2'; 
do for mdel in 'pointnet' 'pointnet2' 'dgcnn' 'pct' 'pointmlp' 'curvenet';
do
echo $mdel $atk
python attack_dataset.py --batch_size 32 -model $mdel $atk;
done;done
