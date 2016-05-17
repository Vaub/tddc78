#!/bin/bash

ompsalloc ./laplsolv_orig

for i in `seq 1 4`; do
    export OMP_NUM_THREADS=$(($i**2))
    ompsalloc ./laplsolv
done
