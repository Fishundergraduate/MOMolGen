#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -l h_rt=24:00:00

module load python cuda/11.2.146 cudnn/8.1 nccl/2.8.4 gcc

#bash init.sh
#source .venv/bin/activate
source ~/.bashrc
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.free,memory.used --format=csv -l 10 &>gpu.log &
python ligand_design/mcts_ligand.py ./data1/
#deactivate
echo finish