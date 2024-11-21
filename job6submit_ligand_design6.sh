#!/bin/bash

mkdir log$(($1*24))h_ChemTS_subj
for i in `seq 21 21`
do
    cp -pr data_test_2 log$(($1*24))h_ChemTS_subj
    mv ligand_design.$(($i)).job.sh.* log$(($1*24))h_ChemTS_subj
    qsub -g tga-science ligand_design.$(($i)).job.sh
    echo submitted $i
done