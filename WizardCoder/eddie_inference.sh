#!/bin/bash

# Grid Engine options (lines prefixed with #$)
# Runtime limit of 1 hour:
#$ -l h_rt=1:00:00
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU:
#$ -q gpu
#$ -pe gpu-a100 1
#
# Request 4 GB system RAM
# the total system RAM available to the job is the value specified here multiplied by
# the number of requested GPUs (above)
#$ -l h_vmem=64G
# -l h_rss=64G
# -l s_vmem=64G
# -l mem_free=64G

# . /etc/profile.d/modules.sh
# Initialaze micromamba ML3.8 environment
source $HOME/.bashrc
micromamba activate $SCRATCH/micromamba/envs/ML3.8
#conda activate CondaML3.8

max_examples=$((2**9*400))
n_examples=$((2**$1*400))
n_partitions=$((max_examples/n_examples))
partition=$((seed%n_partitions))

datasetdir=$SCRATCH/CodeXGLUE/Text-Code/NL-code-search-Adv/dataset
#syntheticdataset=../synthetic_data/d2c_semisynthetic.jsonl

output_dir=./output/
mkdir -p $output_dir

python src/inference_wizardcoder.py \
    --base_model "WizardLM/WizardCoder-15B-V1.0" \
    --input_data_path "$datasetdir/train100.jsonl" \
    --output_data_path "$output_dir/result.jsonl"