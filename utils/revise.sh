for i in `seq 1 20`
do
    python revise.py data$(($i)) $(($i))
    ##rm data$(($i))/output/allLigands
done

#python revise.py data_test 0
#python revise.py data_test_2 0