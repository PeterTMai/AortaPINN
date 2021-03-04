#!/bin/bash

#$ -S /bin/bash

#$ -l tmem=15.9G
#$ -l h_vmem=15.9G

#$ -l h_rt=20:00:00

# -S /bin/bash
# -j y
# -N AortaPINN_initial_test
# -wd /home/petermai/AortaPINN/AortaPINN

hostname
date
echo ""
source /share/apps/source_files/python/python-3.8.5.source

python3 Source/Aorta.py > aorta
