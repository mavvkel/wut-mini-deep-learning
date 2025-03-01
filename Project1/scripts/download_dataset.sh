#!/bin/bash

if ! test -f ./dataset; then
    mkdir ./dataset;
fi

curl -L -o ./dataset/cinic10.zip\
  https://www.kaggle.com/api/v1/datasets/download/mengcius/cinic10 &&
    echo "Dataset saved at $(readlink -e ./dataset/cinic10.zip)"
