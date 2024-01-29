#! /usr/bin/bash

nvcc q10.cu -o q10
for i in {1..10}
do
    ./q10 >/dev/null
    if [ $? -eq 1 ]
    then
        echo At least one iteration failed to correctly compute the block transpose.
        exit 1
    fi
done
echo All iterations successfully computed the block transpose.
