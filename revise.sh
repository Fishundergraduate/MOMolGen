for i in `seq 1 20`
do
    python revise.py data$(($i)) $(($i))
    ##rm data$(($i))/output/allLigands
done