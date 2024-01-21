#!/bin/bash
for atk in 'cw' 'knn'; 
do for mdel in 'pointnet' 'pointnet2' 'dgcnn' 'pct' 'pointmlp' 'curvenet';
do
echo $mdel $atk
python ./scripts/random_search.py -model $mdel -attack $atk -cuda_device 0;
done;done