#!/bin/bash

mkdir log$(($1*24))h_sascore
for i in `seq 16 20`
do
    cp -pr data$(($i)) log$(($1*24))h_sascore
    mv ligand_design.$(($i)).job.sh.* log$(($1*24))h_sascore
    qsub -g tga-science ligand_design.$(($i)).job.sh
    echo submitted $i
done