#!/bin/bash

for model in 'gpt-small'
do
    for dataset in 'natural_stories' 'brown' 'provo_skip2zero' 'dundee_skip2zero' 'provo' 'dundee'
    do
        make process_data MODEL=${model} DATASET=${dataset}
        make get_llh_linear MODEL=${model} DATASET=${dataset}
    done

    for dataset in 'provo_skip2zero' 'dundee_skip2zero'
    do
        make get_llh_skip MODEL=${model} DATASET=${dataset}
    done
done
