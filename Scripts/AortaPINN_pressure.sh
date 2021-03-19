#!/bin/bash

#$ -S /bin/bash

#$ -l tmem=15.9G
#$ -l h_vmem=15.9G

#$ -l h_rt=100:00:00

# -S /bin/bash
# -j y
# -pe smp 4
# -N AortaPINN_pressure
# -wd /home/petermai/AortaPINN/AortaPINN

hostname
date
echo ""
source /share/apps/source_files/python/python-3.8.5.source
cd /home/petermai/AortaPINN/AortaPINN/Source
python3 Aorta_pressure.py > aorta_nondim_pressure
git 
