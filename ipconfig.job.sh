#!/bin/sh
#$ -cwd
#$ -l f_node=2
#$ -l h_rt=0:6:00

sleep 5m 30s  && curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w &
PID=$!
module load python cuda/11.2.146 cudnn/8.1 openmpi
mpirun -n 2 -npernode 1 nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>ipconf.gpu.log &
sleep 40s && curl -X POST https://maker.ifttt.com/trigger/slack_mention/with/key/bD0xPz0Ajd2SWn6JVoww3w
wait $PID