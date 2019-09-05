#!/bin/sh

NPROC=100
BCAST_TYPES="linear binomial"

cd results

rm -rf graphs
mkdir graphs

#
# Graphs of time on process number
#

datasizes=`ls binomial-d* | sed "s/[a-z]*-d//" | sed "s/.dat//"`

for datasize in $datasizes; do
    echo "build for datasize $datasize"
    cat ../graph.gp.tmpl | sed "s/%%PARAM%%/$datasize/g" | 
                           sed "s/%%TYPE%%/nproc/g" | 
                           sed "s/%%X_SYMB%%/d/g" |
                           sed "s/%%XTICS%%/set xtics 8/" >graph.gp.tmp
    gnuplot graph.gp.tmp
done

#
# Graphs of time on buffer size
#

nprocs=`ls binomial-n* | sed "s/binomial-n"// | sed "s/.dat//"`

for nproc in $nprocs; do
    echo "build for nproc $nproc"

    cat ../graph.gp.tmpl | sed "s/%%PARAM%%/$nproc/g" | 
                           sed "s/%%TYPE%%/datasize/g" | 
                           sed "s/%%X_SYMB%%/n/g" |
                           sed "s/%%XTICS%%//" >graph.gp.tmp
    gnuplot graph.gp.tmp
done

# rm graph.gp.tmp
