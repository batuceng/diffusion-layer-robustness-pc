#!/bin/bash
for atk in 'add' 'cw' 'drop' 'knn' 'pgd' 'pgdl2'; 
do for mdel in 'pointnet' 'pointnet2' 'dgcnn' 'pct' 'pointmlp' 'curvenet';
do
echo $mdel $atk
cd ./defense/DUP_Net/
python dupnet_defense.py --batch-size 32 --model $mdel --attack $atk;
done;done
