#!/bin/bash

baseLAS="interesting.las"

newLAS="test_out.las"

pdal pipeline ./knn_2d.json \
    --readers.las.filename="$baseLAS" \
    --writers.las.filename="$newLAS" \
    -v 8
