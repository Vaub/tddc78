#!/bin/bash

for i in `seq 1 20`; do

    mpirun -n $i build/Release/lab1 500 images/im2.ppm images/imOut.ppm
    
done
        
