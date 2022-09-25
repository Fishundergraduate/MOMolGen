mkdir log$(($1*24))h_6lu7
for i in `seq 6 10`
do
    cp -pr data$i log$(($1*24))h_6lu7
    qsub -g tga-science ligand_design.$i.job.sh
    echo submitted $i
done
