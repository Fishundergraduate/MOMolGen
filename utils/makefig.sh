#!/bin/bash

module load python

dataDir=$1
cd $1
startIdx=`ls |grep data| sed -e "s/data//" | sort -n | head -1`
endIdx=`ls |grep data| sed -e "s/data//" | sort -n | tail -1`
cd ..
for  i in `seq $startIdx $endIdx`
do
    curDir="${dataDir}/data${i}"
    cp ${curDir}/present/scores.txt ${curDir}/present/scores.csv
    sed -i -e "s/\[//" -e "s/\]//" ${curDir}/present/scores.csv
    python makeFig.py $curDir &
    
done
wait