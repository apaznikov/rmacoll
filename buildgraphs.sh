#!/bin/sh

NPROC=100
BCAST_TYPES="linear binomial"

cd results

[ ! -d "graphs" ] && mkdir graphs

for ((nproc = 1; nproc <= NPROC; nproc++)); do
    flag="yes"
    for bcast_type in $BCAST_TYPES; do
        if [ ! -f $bcast_type-n$nproc.dat ]; then
            flag="no"
        fi
    done

    if [ $flag == "yes" ]; then
        echo $nproc
        cat ../datasize.gp.tmpl | sed "s/%%NPROC%%/$nproc/g" >datasize.gp.tmp
        gnuplot datasize.gp.tmp
    fi
done

rm datasize.gp.tmp
