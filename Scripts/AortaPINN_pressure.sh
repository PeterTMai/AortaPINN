#!/bin/bash

#$ -S /bin/bash

#$ -l tmem=15.9G
#$ -l h_vmem=15.9G

#$ -l h_rt=5:00:00
#$ -l gpu=true
#$ -R y
# -S /bin/bash
# -j y
# -pe gpu 2
# -N AortaPINN_pressure
# -wd /home/petermai/AortaPINN/AortaPINN

hostname
date
echo ""
source /share/apps/source_files/python/python-3.7.2.source
source /share/apps/source_files/cuda/cuda-10.1.source
cd /home/petermai/AortaPINN/AortaPINN/Source
python3 Aorta_pressure.py > aorta_nondim_pressure
git 
