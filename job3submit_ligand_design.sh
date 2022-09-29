mkdir dataSeed$(($1*24))h
for i in `seq 1 1`
do
    cp -pr data_test dataSeed$(($1*24))h
    qsub -g tga-science ligand_design_seedFixed.job.sh
    echo submitted $i
done
