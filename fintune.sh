#!/bin/bash

exp_tag="qw"
python fitune.py \
    --base_model 'Qwen/Qwen2.5-0.5B' \
    --data_path 'medical_qa.json' \
    --output_dir './lora-'$exp_tag \
    --prompt_template_name 'med_template' \
    --micro_batch_size 64 \
    --batch_size 128 \
    --wandb_run_name $exp_tag
