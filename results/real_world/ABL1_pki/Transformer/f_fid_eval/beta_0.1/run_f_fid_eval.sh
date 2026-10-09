#!/bin/bash

# Export wandb key
export WANDB_API_KEY=431a756ec63a29a4ab674227aa238645f2596251

# Load venv
source /home/marcelhiltscher/Documents/Projects/MolFaith/.venv/bin/activate

# Define parameters
model_path="/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/P00519_WT_ABL1/Transformer/finetuning/hp_tuned_parameters/beta_0.1"
output_folder="/home/marcelhiltscher/Documents/Experiments/Faithfulness_Test/P00519_WT_ABL1/Transformer/f_fid_eval/beta_0.1"
evaluation_strategies="top_percentage"

xai_methods=(
    "integrated_gradients"
    "kernel_shap"
    "lime"
    "shap_sampling"
    "random_explanation"
)

for i in "${!xai_methods[@]}"; do
    method="${xai_methods[$i]}"
    echo "Running explanation method=$method"
    python /home/marcelhiltscher/Documents/Projects/MolFaith/mol_faith/explanations/f_fidelity_evaluation/f_fidelity_eval.py \
        --model_path="$model_path" \
        --explanation_method="$method" \
        --output_folder="$output_folder" \
        --evaluation_strategies="$evaluation_strategies"
    echo "Completed method=$method"
done
echo "All xAI methods processed"
