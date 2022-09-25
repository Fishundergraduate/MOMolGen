mkdir log$(($1*24))h
for i in `seq 1 5`
do
    cp -pr data$i log$(($1*24))h
    qsub -g tga-science ligand_design.$i.job.sh
    echo submitted $i
done
