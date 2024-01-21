#!/bin/bash
for atk in 'cw' 'knn'; 
do for mdel in 'pointnet' 'pointnet2' 'dgcnn' 'pct' 'pointmlp' 'curvenet';
do
echo $mdel $atk
python attack_dataset.py --batch_size 32 -model $mdel $atk;
done;done