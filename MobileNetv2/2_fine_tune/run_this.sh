#!/bin/bash
gpus=0,1,2,3

python main.py --gpu_id ${gpus} 2>&1 | tee log.txt
