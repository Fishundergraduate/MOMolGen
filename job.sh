#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=2:00:00

module load python/3.9.2 cuda/11.2.146 cudnn/8.1 nccl/2.8.4 tensorflow/2.8.0

#bash init.sh
#source .venv/bin/activate
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
python train_RNN/train_RNN.py  && $(curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w?value1=`date +%H%M%S`; true) || curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w?value1=`ERROR`
#deactivate
echo finish
