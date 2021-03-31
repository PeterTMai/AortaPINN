#!/bin/bash

#$ -S /bin/bash

#$ -l tmem=40G
#$ -l h_vmem=40G

#$ -l h_rt=28:00:00
#$ -l gpu=true
#$ -R y
# -S /bin/bash
# -j y
# -pe gpu 4
# -N Aorta_weighted_coarse
# -wd /home/petermai/AortaPINN/AortaPINN

hostname
date
echo ""
source /share/apps/source_files/python/python-3.7.2.source
source /share/apps/source_files/cuda/cuda-10.1.source
cd /home/petermai/AortaPINN/AortaPINN/Source
python3 Aorta_w_val.py > aorta_w_val
