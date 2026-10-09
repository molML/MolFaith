#!/bin/bash

#SBATCH --job-name=GCN_Finetuning_Hbond_Donor
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=gpu_a100
#SBATCH --time=1-12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --gpus=1

# Export wandb key
export WANDB_API_KEY=431a756ec63a29a4ab674227aa238645f2596251

# Load modules
module load 2023
module load CUDA/12.1.1
module load 2024
module load Python/3.12.3-GCCcore-13.3.0


# Load venv
source  /home/mhiltscher/code/Faithfulness_Test/.venv/bin/activate

# Define parameters
config_path_or_base="/projects/prjs1679/Experiments/logP/logp_target/CNN/pretraining/hp_tuned_parameters"
base_output_dir="/projects/prjs1679/Experiments/logP/logp_target/CNN/finetuning/hp_tuned_parameters"
model_type="cnn"
betas=(0.1)

for beta in "${betas[@]}"; do
    output_dir="${base_output_dir}/beta_${beta}"

    echo "Running with beta=$beta, output_dir=$output_dir"
    python /home/mhiltscher/code/Faithfulness_Test/faithfulness_test/training/dnn_training/finetuning.py \
        --config_path_or_base="$config_path_or_base" \
        --output_dir="$output_dir" \
        --beta="$beta" \
        --model_type="$model_type" \
        --multiple_splits

    echo "Completed beta=$beta"
done

echo "All beta values processed"
