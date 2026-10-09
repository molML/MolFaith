#!/bin/bash

#SBATCH --job-name=GIN_FXA_LogD_f_fid_eval
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=gpu_h100
#SBATCH --time=4-24:00:00
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
source /home/mhiltscher/code/MolFaith/.venv/bin/activate

# Define parameters
model_path="/projects/prjs1679/Experiments/FXA_logs_logd_stratified/logd_results/GIN/finetuning/hp_tuned_parameters/beta_0.1"
output_folder="/projects/prjs1679/Experiments/FXA_logs_logd_stratified/logd_results/GIN/f_fid_eval/beta_0.1"
evaluation_strategies="top_percentage"

xai_methods=(
    "integrated_gradients"
    "kernel_shap"
    "lime"
    "gnn_explainer"
    "graph_mask"
    "shap_sampling"
    "random_explanation"
)

for i in "${!xai_methods[@]}"; do
    method="${xai_methods[$i]}"
    echo "Running explanation method=$method"
    python /home/mhiltscher/code/MolFaith/mol_faith/explanations/f_fidelity_evaluation/f_fidelity_eval.py \
        --model_path="$model_path" \
        --explanation_method="$method" \
        --output_folder="$output_folder" \
        --evaluation_strategies="$evaluation_strategies"
    echo "Completed method=$method"
done
echo "All xAI methods processed"
