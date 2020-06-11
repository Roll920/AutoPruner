#!/bin/bash
gpus=4,5,6,7

for layer in $(seq 0 3)
do
	# 1. prune
	LOG=results/logs/log_${layer}.txt
	python main.py \
		--gpu_id ${gpus} \
		--group_id ${layer} 2>&1 | tee ${LOG}

	# 3. compressed model
	cd compress_model
	python compress_model.py --group_id ${layer}
	cd ..
done

python fine_tune_compressed_model.py --gpu_id ${gpus} 2>&1 | tee "results/logs/fine_tune_log.txt"
