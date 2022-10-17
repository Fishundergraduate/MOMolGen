#!/bin/bash

mkdir log$(($1*24))h_docking
for i in `seq 22 22`
do
    cp -pr data_test_2 log$(($1*24))h_docking
    mv ligand_design.$(($i)).job.sh.* log$(($1*24))h_docking
    qsub -g tga-science ligand_design.$(($i)).job.sh
    echo submitted $i
done