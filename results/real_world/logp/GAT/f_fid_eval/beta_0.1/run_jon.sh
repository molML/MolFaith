#!/bin/bash
#SBATCH --job-name=GAT_TopPerc_FFid_LogP_Target
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=gpu_h100
#SBATCH --time=4-24:00:00
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
source /home/mhiltscher/code/Faithfulness_Test/.venv/bin/activate

# Define common parameters
model_path="/projects/prjs1679/Experiments/logP/logp_target/GAT/finetuning/hp_tuned_parameters/beta_0.1"
output_folder="/projects/prjs1679/Experiments/logP/logp_target/GAT/f_fid_eval/beta_0.1"
evaluation_strategies="top_percentage"

# Define all XAI methods to test
xai_methods=(
    "integrated_gradients"
    "kernel_shap"
    "lime"
    "gnn_explainer"
    "graph_mask"
    "shap_sampling"
    "random_explanation"
)

echo "Starting F-Fidelity evaluation for all XAI methods..."
echo "Model path: $model_path"
echo "Base output folder: $output_folder"
echo "Total methods to evaluate: ${#xai_methods[@]}"
echo "----------------------------------------"

# Loop through all XAI methods
for i in "${!xai_methods[@]}"; do
    method="${xai_methods[$i]}"
    
    echo "[$((i+1))/${#xai_methods[@]}] Processing XAI method: $method"
    echo "Start time: $(date)"
    
    # Run F-Fidelity evaluation for current method
    python /home/mhiltscher/code/Faithfulness_Test/faithfulness_test/explanations/f_fidelity_evaluation/f_fidelity_eval.py \
        --model_path="$model_path" \
        --explanation_method="$method" \
        --output_folder="$output_folder" \
        --evaluation_strategies="$evaluation_strategies"
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $method"
    else
        echo "✗ Failed to complete: $method"
        echo "Error occurred at $(date)"
    fi
    
    echo "End time: $(date)"
    echo "----------------------------------------"
done

echo "F-Fidelity evaluation completed for all XAI methods!"
echo "Final completion time: $(date)"