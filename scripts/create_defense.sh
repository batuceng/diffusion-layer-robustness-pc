#!/bin/bash
for atk in 'add' 'cw' 'drop' 'knn' 'pgd' 'pgdl2'; 
do for mdel in 'curvenet';
do for def in 'sor';
do
echo $mdel $atk $def
if [ "$def" == 'srs' ];
then
    cd ./defense/SOR_SRS/
    python sor_srs_defense.py --batch-size 32 --model $mdel --attack $atk --defense $def;
    cd ../../
elif [ "$def" == 'sor' ]
then
    cd ./defense/SOR_SRS/
    python sor_srs_defense.py --batch-size 1 --model $mdel --attack $atk --defense $def;
    cd ../../
elif [ "$def" == 'dupnet' ]
then
    cd ./defense/DUP_Net/
    python dupnet_defense.py --batch-size 32 --model $mdel --attack $atk;
    cd ../../
elif [ "$def" == 'ifdefense' ]
then
    cd ./defense/IF_DEFENSE/ConvONet/
    python if_defense.py --batch-size 32 --model $mdel --attack $atk;
    cd ../../../
else
    echo "Invalid defense"
fi
done done done