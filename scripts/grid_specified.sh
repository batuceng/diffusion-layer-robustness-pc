#!/bin/bash
for mdel in 'pointnet' 'pointnet2' 'dgcnn' 'pct' 'pointmlp' 'curvenet';
do for atk in 'add' 'cw' 'drop' 'knn' 'pgd' 'pgdl2';
do
echo $mdel $atk
python scripts/random_search.py  -model $mdel -attack $atk -cuda_device 0 --search_method "specified";
done;done