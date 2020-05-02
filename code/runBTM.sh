#!/bin/bash

K=8   # number of topics

alpha=`echo "scale=3;50/$K"|bc`
beta=0.005
niter=100
save_step=501

input_dir=./temp/
output_dir=./temp/
model_dir=${output_dir}model/
mkdir -p $output_dir/model 

# the input docs for training
doc_pt=${input_dir}health_tweets.txt

# docs after indexing
dwid_pt=${output_dir}doc_wids.txt
# vocabulary file
voca_pt=${output_dir}voca.txt
python3 indexDocs.py $doc_pt $dwid_pt $voca_pt

## learning parameters p(z) and p(w|z)
W=`wc -l < $voca_pt` # vocabulary size
make -C ./btm
./btm/btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir

## infer p(z|d) for each doc
./btm/btm inf sum_b $K $dwid_pt $model_dir

## output top words of each topic
python3 BTMTopics.py $model_dir $K $voca_pt
