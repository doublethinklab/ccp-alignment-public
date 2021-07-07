#!/bin/bash

docker run \
    --rm \
    -p 8890:8890 \
    -v ${PWD}:/ccpalign \
    -w /ccpalign \
    ccp-alignment:latest \
        jupyter notebook --ip 0.0.0.0 --port 8890 --allow-root --no-browser
