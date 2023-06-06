#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo $$ > /home/shared/kunchaya/Msc-thesis/arr_trans.pid
python arrange_transcript.py /home/shared/kunchaya/data/must-c-v1/en-de/result/train/ /home/shared/kunchaya/data/must-c-v1/en-de/data/train/txt/train