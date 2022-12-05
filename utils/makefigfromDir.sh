echo "The last is log$1h_$2"
for i in `seq 0 24 $1`
do
    ./makefig.sh log${i}h${2} &
    echo "finish log${i}h_$2"
done
wait