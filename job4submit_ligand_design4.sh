mkdir log_DlogPQED$(($1*24))h
for i in `seq 11 15`
do
    cp -pr data$(($i)) log_DlogPQED$(($1*24))h
    qsub -g tga-science ligand_design.$(($i)).job.sh
    echo submitted $i
done