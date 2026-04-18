#!/bin/bash
#PBS -N hici-llama-train
#PBS -P hn98
#PBS -q gpuhopper
#PBS -l walltime=40:00:00
#PBS -l ncpus=48
#PBS -l ngpus=4
#PBS -l mem=1024GB
#PBS -l jobfs=50GB
#PBS -l storage=gdata/hn98+scratch/hn98
#PBS -l wd
#PBS -j oe
#PBS -o /g/data/hn98/Yang/llm-mem/HiCI/Training_out_fuxian/train_hici_llama.log

source /g/data/hn98/mini3/etc/profile.d/conda.sh
conda activate /g/data/hn98/Yang/envs/hici-llama

cd /g/data/hn98/Yang/llm-mem/HiCI

bash train_fine_tune_hici.sh 2>&1 | tee Training_out_fuxian/Llama-2-13b-hici-16k-none-sub.txt
