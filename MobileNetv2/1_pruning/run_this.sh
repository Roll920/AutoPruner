#!/bin/bash
gpus=4,5,6,7

python main.py --gpu_id ${gpus} 2>&1 | tee log.txt
