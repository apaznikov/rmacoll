#!/bin/sh

TASKJOB=slurm.job

NNODES=6
PPN=8

BCAST_TYPES="linear binomial"

[ ! -d "results" ] && mkdir results

for ((nnodes = 1; nnodes <= NNODES; nnodes++)); do
    for bcast_type in $BCAST_TYPES; do
        cat slurm.job | sed "s/rmacoll.*$/rmacoll $bcast_type/" \
                      | sed "s/nodes=[0-9]\+/nodes=$nnodes/" > slurm.job.tmp
        sbatch slurm.job.tmp
    done
done

rm slurm.job.tmp
