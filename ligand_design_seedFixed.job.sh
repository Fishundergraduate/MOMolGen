#!/bin/sh
#$ -cwd
#$ -l s_gpu=1
#$ -l h_rt=24:00:00

sleep 23h 45m && curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w/ &
PID=$!
module load python cuda/11.2.146 cudnn/8.1 nccl/2.8.4 gcc

#bash init.sh
#source .venv/bin/activate
source ~/.bashrc
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
python -m cProfile -o ./data_test/pylog.prof -s tottime ligand_design/mcts_ligand_copy.py ./data_test/
#deactivate
wait $PID
echo finish