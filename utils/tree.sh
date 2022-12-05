#!/bin/sh
#$ -cwd
#$ -l s_core=1
#$ -l h_rt=0:10:00

sleep 60
qsub tree.sh