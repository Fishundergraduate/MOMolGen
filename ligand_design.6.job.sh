#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -l h_rt=24:00:00

module load python cuda/11.2.146 cudnn/8.1 nccl/2.8.4 gcc

#bash init.sh
#source .venv/bin/activate
source ~/.bashrc
#nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
#for i in `seq 1 4`
#do
#    n=$(($i+8))
#    cp -pr data6 log$(($n*24))h_6lu7
#    python ligand_design/mcts_ligand.py ./data6/
#    echo finish
#    curl -X POST https://maker.ifttt.com/trigger/JOB_FINISH/with/key/bD0xPz0Ajd2SWn6JVoww3w/
#done

python ligand_design/mcts_ligand.py ./data6/
#deactivate
echo finish