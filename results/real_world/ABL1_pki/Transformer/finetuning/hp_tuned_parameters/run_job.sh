#!/bin/bash

#SBATCH --job-name=Transformer_Finetuning
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=bme.gpu2.q
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=12G
#SBATCH --gpus=1

# Export wandb key
export WANDB_API_KEY=431a756ec63a29a4ab674227aa238645f2596251

# Load modules
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load Python/3.12.3-GCCcore-13.3.0

# Load venv
source /home/20243926/code/MolFaith/.venv/bin/activate

# Define parameters
config_path_or_base="/home/20243926/experiments/faithfulness_test/P00519_WT_ABL1/Transformer/pretraining/hp_tuned_parameters"
base_output_dir="/home/20243926/experiments/faithfulness_test/P00519_WT_ABL1/Transformer/finetuning/hp_tuned_parameters"
model_type="transformer"
betas=(0.1)

for beta in "${betas[@]}"; do
    output_dir="${base_output_dir}/beta_${beta}"

    echo "Running with beta=$beta, output_dir=$output_dir"
    python /home/20243926/code/MolFaith/mol_faith/training/dnn_training/finetuning.py \
        --config_path_or_base="$config_path_or_base" \
        --output_dir="$output_dir" \
        --beta="$beta" \
        --model_type="$model_type" \
        --multiple_splits

    echo "Completed beta=$beta"
done

echo "All beta values processed"
