#!/bin/bash
for atk in 'pgd' 'pgdl2' 'drop'; 
do for mdel in 'pointnet' 'pointnet2' 'dgcnn' 'pct' 'pointmlp' 'curvenet';
do
echo $mdel $atk
python attack_dataset.py -model $mdel $atk ;
done;done