#!/bin/bash

# echo "TODO: fill in the docker build command"
docker build -t ift6758/serving:$1 . -f Dockerfile.serving
docker build -t ift6758/serving:$1 . -f Dockerfile.streamlit