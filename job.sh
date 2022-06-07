#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=00:10:00

module load python/3.9.2 cuda/11.6.1 cudnn/8.3

#bash init.sh
source .venv/bin/activate
python train_RNN/train_RNN.py && $(curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w?value1=`date +%H%M%S`; true) || curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w?value1=`ERROR`
deactivate
echo finish
