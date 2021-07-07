#!/bin/bash

if [ -z $1 ]
then
    docker run \
        --rm \
        -v ${PWD}:/ccpalign \
        -w /ccpalign \
        ccp-alignment:latest \
            python -m unittest discover
else
    docker run \
        --rm \
        -v ${PWD}:/ccpalign \
        -w /ccpalign \
        ccp-alignment:latest \
            python -m unittest $1
fi
