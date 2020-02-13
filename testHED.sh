#!/bin/bash
dataset='histograph-ak'
# dataset='histograph-gw'
# data_path=../../Datasets/histograph/01_GW/01_Keypoint/
data_path=../../Datasets/histograph/03_AK/01_Keypoint/
bz="--batch_size 64"
ngpu="--ngpu 0"
prefetch="--prefetch 4"
set_partition="--set_partition cv1"

tau_n=4
tau_e=16
alpha=0.5
beta=0.1

echo "EXPERIMENT Tn=$tau_n; Te=$tau_e; Alpha=$alpha; Beta=$beta"
python src/testHED.py $dataset $data_path $set_partition $bz $prefetch $ngpu --tau_n $tau_n --tau_e $tau_e --alpha $alpha --beta $beta
