#!/bin/bash

#$ -S /bin/bash

#$ -l tmem=63.9G
#$ -l h_vmem=63.9G

#$ -l h_rt=5:00:00
#$ -l gpu=true
#$ -R y
# -S /bin/bash
# -j y
# -pe gpu 2
# -N Aorta_weighted
# -wd /home/petermai/AortaPINN/AortaPINN

hostname
date
echo ""
source /share/apps/source_files/python/python-3.7.2.source
source /share/apps/source_files/cuda/cuda-10.1.source
cd /home/petermai/AortaPINN/AortaPINN/Source
python3 Aorta_w.py > aorta_w
