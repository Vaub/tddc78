#!/bin/bash

for i in `seq 1 5`; do

    a = $(($i**2))
    mpirun -n $a build/Release/lab1 500 images/im2.ppm images/imOut.ppm
    
done

mpirun -n 1 build/Release/lab1 1 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 100 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 200 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 300 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 400 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 500 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 1000 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 1500 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 2000 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 2500 images/im2.ppm images/imOut.ppm
mpirun -n 1 build/Release/lab1 3000 images/im2.ppm images/imOut.ppm

        
