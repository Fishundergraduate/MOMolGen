mkdir log$(($1*24))h_DlogPQED
for i in `seq 11 15`
do
    cp -pr data$(($i)) log$(($1*24))h_DlogPQED
    qsub -g tga-science ligand_design.$(($i)).job.sh
    echo submitted $i
done