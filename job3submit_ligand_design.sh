mkdir log$(($1*24))h_DataSeed
for i in `seq 1 1`
do
    cp -pr data_test log$(($1*24))h_DataSeed
    qsub -g tga-science ligand_design_seedFixed.job.sh
    echo submitted $i
done
