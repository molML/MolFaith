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
│   │   ├── logp_dataset_exploration.ipynb
│   │   └── target_size_analysis.ipynb
│   ├── results_analysis/
│   │   ├── fid_general.ipynb
│   │   ├── fid_topk_thres_analysis.ipynb
│   │   ├── fid_vs_gt.ipynb
│   │   ├── gt_general.ipynb
│   │   ├── model_evaluation.ipynb
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
