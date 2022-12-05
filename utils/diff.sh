for i in `seq 0 2`
do
    start=$((${i}*24))
    end=$((${start}+24))
    diff -u log${end}h_6lu7/data6/present/scores.csv log${start}h_6lu7/data6/present/scores.csv | grep ^-- | grep -v ^--- | sed "s/-//" > diff${start}.csv

done