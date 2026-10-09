#!/bin/bash

#SBATCH --job-name=GIN_logp_target_pretraining
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=gpu_h100
#SBATCH --time=4-12:00:00
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


# Run script
config_path="/projects/prjs1679/Experiments/logP/logp_target/GIN/pretraining/hp_tuned_parameters/config.yaml"
model_type="gnn"

python /home/mhiltscher/code/Faithfulness_Test/faithfulness_test/training/dnn_training/training.py --config_path=$config_path --model_type=$model_type --multiple_splits