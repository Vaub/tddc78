#!/bin/bash

for i in `seq 0 6`; do
    cpu=$((2**$i))
    mpprun -n $i ./lab2_thre images/im4.ppm images/imOutTre.ppm
    
done
