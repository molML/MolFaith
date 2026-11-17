# MolFaith 

MolFaith is a benchmarking framework for evaluating the faithfulness of explainable AI methods in molecular property prediction using fidelity-based metrics across 5 models, 3 training tasks, and 8 feature attribution methods.

This is the repository to the preprint [Explaining What Matters: Faithfulness in Molecular Deep Learning](https://doi.org/10.26434/chemrxiv.10001594/v1). 


## Install

To install this package and its dependencies using `uv` (see corresponding [documentation](https://docs.astral.sh/uv/getting-started/)):

```bash
uv sync
```

This command will:
- Create a virtual environment (if it doesn't exist)
- Install all dependencies from `uv.lock`
- Install the package in editable mode

After installation, activate the virtual environment:
```bash
source .venv/bin/activate
```

Installation should take few seconds only.

## Project Structure

The repository is organized into three main top-level directories:

•	`data/` contains all files related to the three training datasets, including raw data, preprocessed data, and dataset splits.

•	`results/` stores outputs from hyperparameter tuning, model checkpoints (both pretraining and finetuning), and the results of fidelity- and ground-truth–based evaluations.

•	`examples/` provides example configuration files for hyperparameter tuning and model training.

The core implementation of the benchmark is located in the mol_faith/ package, which is structured as follows:

```
mol_faith/
├── data/
│   ├── create_dataset/
│   │   ├── create_gt_data.py
│   │   ├── create_multi_readout_data_splits.py
│   │   ├── create_stratified_data_splits.py
│   │   └── __init__.py
│   ├── data_managers/
│   │   ├── data_manager.py
│   │   ├── graph_data_manager.py
│   │   ├── smiles_data_manager.py
│   │   ├── tokenization_helper.py
│   │   └── __init__.py
│   ├── preprocessing/
│   │   ├── clean_data.py
│   │   ├── filter_data.py
│   │   ├── filter_helper.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── types.py
├── explanations/
│   ├── attribution_providers/
│   │   ├── captum_methods_provider.py
│   │   ├── graphmask.py
│   │   ├── guided_gradcam.py
│   │   ├── pyg_methods_provider.py
│   │   ├── utils.py
│   │   └── __init__.py
│   ├── explainer/
│   │   ├── base_explainer.py
│   │   ├── gnn_explainer.py
│   │   ├── sequence_explainer.py
│   │   ├── processing_modes.py
│   │   ├── utils.py
│   │   └── __init__.py
│   ├── f_fidelity_evaluation/
│   │   ├── f_fidelity_eval.py
│   │   └── __init__.py
│   ├── gt_evaluation/
│   │   ├── gt_eval.py
│   │   └── __init__.py
│   ├── wrappers/
│   │   ├── cnn_wrappers.py
│   │   ├── gnn_wrappers.py
│   │   ├── transformer_wrapper.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── types.py
├── masking/
│   ├── mask_generator.py
│   ├── test_mask_generator.py
│   └── __init__.py
├── model/
│   ├── CNN/
│   │   ├── cnn_model.py
│   │   └── __init__.py
│   ├── GNN/
│   │   ├── base_gnn.py
│   │   ├── gat_model.py
│   │   ├── gcn_model.py
│   │   ├── gin_model.py
│   │   └── __init__.py
│   ├── transformer/
│   │   ├── transformer.py
│   │   └── __init__.py
│   ├── configs/
│   │   ├── dnn_configs.py
│   │   ├── utils.py
│   │   └── __init__.py
│   ├── shared/
│   │   ├── prediction_head.py
│   │   ├── smiles_token_embedding.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── types.py
├── model_evaluation/
│   ├── dnn_evaluation.py
│   ├── model_evaluation.py
│   └── __init__.py
├── notebooks/
│   ├── datasets/
│   │   ├── abl1_pki.ipynb
│   │   ├── fxa_logs_logd.ipynb
│   │   ├── logp_dataset_exploration.ipynb
│   │   └── target_size_analysis.ipynb
│   ├── results_analysis/
│   │   ├── real_world_data/
│   │   │   ├── check_ABL1_results.ipynb
│   │   │   ├── check_logd_results.ipynb
│   │   │   ├── check_logp_result.ipynb
│   │   │   ├── check_logs_results.ipynb
│   │   │   ├── compare_logd_logs_results.ipynb
│   │   │   └── results_combined.ipynb
│   │   ├── synthetic_data/
│   │   │   ├── fid_general.ipynb
│   │   │   ├── fid_topk_thres_analysis.ipynb
│   │   │   ├── fid_vs_gt.ipynb
│   │   │   ├── gt_general.ipynb
│   │   │   └── model_evaluation.ipynb
│   │   ├── shared.py
│   │   └── __init__.py
│   └── __init__.py
├── target_extraction/
│   ├── benzene_substructure_extraction.py
│   ├── hbond_donor_extraction.py
│   ├── largest_conjugated_system_extraction.py
│   ├── shared.py
│   ├── test_benzene_substructure_extraction.py
│   └── __init__.py
├── training/
│   ├── dnn_training/
│   │   ├── checkpoint_handler.py
│   │   ├── finetuning.py
│   │   ├── gnn_trainer.py
│   │   ├── metrics.py
│   │   ├── sequence_model_trainer.py
│   │   ├── trainer.py
│   │   ├── training.py
│   │   └── __init__.py
│   ├── hyperparameter_tuning/
│   │   ├── dnn_hp_tuner.py
│   │   ├── hp_tuner.py
│   │   ├── tune_hyperparams.py
│   │   ├── utils.py
│   │   ├── visualize_hyperparam_tuning.py
│   │   └── __init__.py
│   ├── const.py
│   ├── __init__.py
│   └── types.py
├── utils/
│   ├── adjust_paths.py
│   └── __init__.py
└── __init__.py
```

## Entry Points

The benchmark is organized around four main entry points that build on each other. The output of each step can be used as input for subsequent steps. Each script requires a configuration file specifying model and training parameters, data and output paths, and logging settings via Weights & Biases. Example configuration files are provided to illustrate the expected structure.

The standard workflow is as follows:

**1. Hyperparameter tuning**

 ``` bash
 python mol_faith/training/hyperparameter_tuning/tune_hyperparams.py --help
 ```

**2. Pretraining** 

``` bash 
python mol_faith/training/dnn_training/training.py --help
```

**3. Finetuning** 

``` bash 
python mol_faith/training/dnn_training/finetuning.py --help
```

**4. Explanation evaluation**

F-Fidelity evaluation:
``` bash 
python mol_faith/explanations/f_fidelity_evaluation/f_fidelity_eval.py --help
```

Ground-truth alignment evaluation:
``` bash 
python mol_faith/explanations/gt_evaluation/gt_eval.py --help
```

Depending on the entry point the runtime can be very differnt.

## Lookup table for paper figures and tables

Lookup table of where each figure and table of the preprint (main text and Supplementary Information, SI) is produced. Paths are relative to `mol_faith/`.

Abbreviations: `RW/` = `notebooks/results_analysis/real_world_data/`, `SYN/` = `notebooks/results_analysis/synthetic_data/`, `DS/` = `notebooks/datasets/`. The `RW/check_<task>_results.ipynb` notebooks exist once per real-world task: `check_logp_result`, `check_logd_results`, `check_logs_results` and `check_ABL1_results`.

### Main text

| Item | Content | Where |
|---|---|---|
| Fig. 1 | Study overview (schematic) | No code, drawn manually |
| Table 1 | Task summary: no. molecules, atoms, GT atoms, GT fraction | `DS/target_size_analysis.ipynb` (GT atoms and fraction); `RW/results_combined.ipynb` → "Molecule count and size per dataset" (real-world molecules and atoms); `DS/logp_dataset_exploration.ipynb` (logP and synthetic task data creation) |
| Fig. 2a, b | Fid+ violin/box plots by model and by xAI method (real-world) | `RW/results_combined.ipynb` → "Boxplots" / "Violin plots" |
| Fig. 2c | PCA of Fid+ (real-world) | `RW/results_combined.ipynb` → "PCA" |
| Fig. 3a | Std. of Fid+ across molecules per method and model | `RW/check_<task>_results.ipynb` → "Summary table & standard deviation" |
| Fig. 3b, c | ΔFid+ binned by Fid+ at k=20 (SHAP vs GraphMask, GAT; ABL1 and FXa logS) | `RW/check_ABL1_results.ipynb` and `RW/check_logs_results.ipynb` → "Parameter k analysis" |
| Fig. 4a | Schematic of the four faithfulness scenarios | No code, drawn manually |
| Fig. 4b | Kendall τ between GT accuracy and Fid+ | `SYN/fid_vs_gt.ipynb` → "Correlation Fid+ vs GT accuracy" |
| Table 2 | Kendall τ with GT size and GT accuracy/Fid+ edge cases | `SYN/fid_vs_gt.ipynb` → "Correlation with GT size" and "GT accuracy vs Fid+ mismatches" |

### Supplementary Information

| Item | Content | Where |
|---|---|---|
| Table 1 | Summary of xAI methods | No code |
| **Table 2** | **Model performance (RMSE, MAE, Q²F3), all tasks, before finetuning** | **Computed with `model_evaluation/dnn_evaluation.py` → `DNNEvaluator.evaluate_multi_model_performance()` (writes `multi_model_comparison.csv` via `save_multi_model_results_to_csv()`). Formatted with `format_multi_model_comparison_table()` in `SYN/model_evaluation.ipynb` (synthetic) and in the "Model comparison" section of `RW/check_<task>_results.ipynb` (real-world)** |
| Fig. 1 | GT atom fraction of the synthetic tasks | `DS/target_size_analysis.ipynb` |
| Fig. 2 | PCA with loadings (synthetic) | `SYN/fid_general.ipynb` → "PCA" |
| Fig. 3 | PCA with loadings (real-world) | `RW/results_combined.ipynb` → "PCA" |
| Fig. 4 | Fid+ by model, by xAI method and PCA (synthetic) | `SYN/fid_general.ipynb` → "Boxplots" / "Violin plots" / "PCA" |
| Fig. 5 | Std. of Fid+ (synthetic) | `SYN/fid_general.ipynb` → "Standard deviation" |
| Fig. 6, 7 | Effect of k on median F-Fid+ (synthetic) | `SYN/fid_topk_thres_analysis.ipynb` (Fig. 6: `plot_fidelity_by_threshold`; Fig. 7: `plot_fidelity_by_threshold_per_target`) |
| Fig. 8, 9 | Effect of k on median F-Fid+ (real-world) | `RW/results_combined.ipynb` → "Fidelity by threshold" (Fig. 8: `plot_fidelity_by_threshold`; Fig. 9: `plot_fidelity_by_threshold_per_target`) |
| Fig. 10–12 | ΔFid+ vs k, compared to k=20 (benzene, conjugated system, H-bond donor) | `SYN/fid_general.ipynb` → "Parameter k analysis" (`plot_negative_deviations`) |
| Fig. 13–16 | ΔFid+ vs k, compared to k=20 (ABL1, FXa logS, FXa logD, logP) | `RW/check_<task>_results.ipynb` → "Parameter k analysis" → "Negative deviation" (`plot_negative_deviations`) |
| Fig. 17–19, 24–26 | Binning analysis of ΔFid+ (benzene, conjugated system, H-bond donor) | `SYN/fid_general.ipynb` → "Parameter k analysis" (`plot_k_effect_bars`) |
| Fig. 20–23, 27–30 | Binning analysis of ΔFid+ (ABL1, FXa logD, FXa logS, logP) | `RW/check_<task>_results.ipynb` → "Parameter k analysis" (`plot_k_effect_bars`) |
| Fig. 31 | Spearman correlation of Fid+ between FXa logD and logS | `RW/compare_logd_logs_results.ipynb` → "Correlation per model and xAI method" |
| Fig. 32 | Spearman correlation of Fid+ between IG and SHAP | `RW/compare_logd_logs_results.ipynb` → "IG vs SHAP agreement within each task" |
| **Fig. 33–35** | **Masking degradation in absolute error, pretrained vs finetuned (synthetic, A.12)** | **`model_evaluation/dnn_evaluation.py` → `DNNEvaluator.evaluate_masking_degradation_analysis()` and `plot_masking_degradation_analysis()`** |
| **Fig. 36–39** | **Masking degradation in absolute error, pretrained vs finetuned (real-world, A.13)** | **Same as Fig. 33–35** |
| Table 3 | Fid+ summary statistics (synthetic) | `SYN/fid_general.ipynb` → "Summary table" |
| Table 4 | Fid+ summary statistics (real-world) | `RW/results_combined.ipynb` → "Summary table" |
| Tables 5, 6 | Sign disagreement frequency and ranking without it (synthetic) | `SYN/fid_general.ipynb` → "Sign normalization effect" |
| Tables 7, 8 | Sign disagreement frequency and ranking without it (real-world) | `RW/results_combined.ipynb` → "Sign normalization effect" |
| Table 9 | Std. of Fid+ with and without sign disagreement (real-world) | `RW/results_combined.ipynb` → "Std variation" |
| Fig. 40 | GT accuracy boxplots by model and xAI method | `SYN/gt_general.ipynb` |
| Tables 10–14 | Hyperparameter search spaces (CNN, Transformer, GAT, GCN, GIN) | Defined in `examples/hp_tuning_configs/*.yaml` (not a notebook) |
| Tables 15, 16 | OLS and effect sizes, shared xAI methods (synthetic) | `SYN/fid_general.ipynb` → "OLS analysis" (`supplementary_fid_effect_size_table`) |
| Tables 17, 18 | OLS and effect sizes, shared xAI methods (real-world) | `RW/results_combined.ipynb` → "OLS analysis" (`supplementary_fid_effect_size_table`) |
| Tables 19, 20 | OLS and effect sizes, graph models (synthetic) | `SYN/fid_general.ipynb` → "OLS analysis" (`supplementary_fid_effect_size_table_graph`) |
| Tables 21, 22 | OLS and effect sizes, graph models (real-world) | `RW/results_combined.ipynb` → "OLS analysis" (`supplementary_fid_effect_size_table_graph`) |
| Tables 23, 24 | OLS and effect sizes, GCN and CNN (synthetic) | `SYN/fid_general.ipynb` → "OLS analysis" (`supplementary_fid_effect_size_table_conv`) |
| Tables 25, 26 | OLS and effect sizes, GCN and CNN (real-world) | `RW/results_combined.ipynb` → "OLS analysis" (`supplementary_fid_effect_size_table_conv`) |
