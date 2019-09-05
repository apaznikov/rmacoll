#!/bin/sh

TASKJOB=slurm.job

NNODES=6

BCAST_TYPES="linear binomial"

if [ -x "$(command -v sbatch)" ]; then
    cmd=sbatch
    batch=slurm
elif [ -x "$(command -v qsub)" ]; then 
    cmd=qsub
    batch=torque
fi

rm -rf results
mkdir results

for ((nnodes = 1; nnodes <= NNODES; nnodes++)); do
    for bcast_type in $BCAST_TYPES; do
        cat $batch.job | sed "s/rmacoll.*$/rmacoll $bcast_type/" \
                       | sed "s/nodes=[0-9]\+/nodes=$nnodes/" > $batch.job.tmp
        $cmd $batch.job.tmp
    done
done

rm $batch.job.tmp
