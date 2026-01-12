import glob
import math
import os
from pathlib import Path
from typing import List, Dict

import optuna
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from optuna import Study
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from pandas import DataFrame
from plotly.io import show, kaleido


def load_study(study_path: str) -> Study:
    storage = JournalStorage(JournalFileBackend(study_path))
    loaded_study = optuna.load_study(study_name=None, storage=storage)
    return loaded_study


def plot_slices(study: Study, params: List[str], plot_title: str, target_name: str, plot_output_path: str):
    fig = optuna.visualization.plot_slice(study, params=params)
    fig.update_layout(
        title=dict(text=plot_title),
        yaxis=dict(title=dict(text=target_name))
    )
    # show(fig)
    kaleido.scope.mathjax = None
    fig.write_image(os.path.join(plot_output_path, "slice_plot.pdf"))


def plot_importance(study: Study, plot_output_path: str):
    fig = optuna.visualization.plot_param_importances(study)
    # show(fig)
    kaleido.scope.mathjax = None
    fig.write_image(os.path.join(plot_output_path, "param_importance.pdf"))


def plot_history(study: Study, plot_output_path: str, plot_title: str):
    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(
        title=dict(text=plot_title),
    )
    fig.write_image(os.path.join(plot_output_path, "hp_tuning_history.pdf"))


def get_metrics_from_splits(output_folder_path: str, train: bool, metric: str) -> DataFrame:
    all_data = []
    csv_file_name = 'train_metrics.csv' if train else 'val_metrics.csv'
    data_splits_paths = [f for f in glob.glob(os.path.join(output_folder_path, 'data_split_*'))]
    for data_split_idx, target_folder_paths in enumerate(data_splits_paths):
        df_data = pd.read_csv(os.path.join(target_folder_paths, csv_file_name))

        if metric not in df_data.columns:
            raise ValueError(f"Metric '{metric}' not found in the data. Available metrics are: {list(df_data.columns)}")

        if data_split_idx == len(data_splits_paths) - 1:
            all_data.append(df_data[[metric, 'epoch_index']].rename(columns={metric: f'split_{data_split_idx}'}))
        else:
            all_data.append(df_data[[metric]].rename(columns={metric: f'split_{data_split_idx}'}))

    df = pd.concat(all_data, axis=1)

    dataset_label = 'train' if train else 'val'
    df['dataset'] = dataset_label

    split_columns = [col for col in df.columns if col not in ['epoch_index', 'dataset']]

    df_long = df.melt(id_vars=['epoch_index', 'dataset'],
                      value_vars=split_columns,
                      var_name='split',
                      value_name='metric')

    return df_long


def plot_metric_for_trial(plot_output_path: str, output_path: str, trial_number: int, metric: str, title_name: str):
    trial_output_folder_path = os.path.join(output_path, f'trial_{trial_number}')

    train_data = get_metrics_from_splits(output_folder_path=trial_output_folder_path, train=True, metric=metric)

    val_data = get_metrics_from_splits(output_folder_path=trial_output_folder_path, train=False, metric=metric)

    combined_df = pd.concat([train_data, val_data], ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_df,
                 x='epoch_index',
                 y='metric',
                 hue='dataset',
                 style='dataset',
                 estimator='mean',
                 errorbar='sd',
                 err_style='band')

    plt.title(title_name)
    plt.xlabel('Epoch')
    plt.ylabel(f'{metric.upper()}')
    plt.legend(title='Dataset')
    plt.tight_layout()

    plt.savefig(os.path.join(plot_output_path, f'trial_{trial_number}_model_performance.pdf'))


def get_param_names(param_dict: Dict[str, float]):
    return list(param_dict.keys())


def plot_trials_model_performance(output_path: str, study: Study, best_n_trials: int, plot_output_path: str):
    trials = study.get_trials()
    sorted_trials = sorted(trials, key=lambda x: x.values, reverse=False)
    for trial in sorted_trials[:best_n_trials]:
        plot_metric_for_trial(output_path=output_path, trial_number=trial.number, metric='rmse',
                              title_name=f'Trial {trial.number}', plot_output_path=plot_output_path)


def get_params_n_best_trials(study: Study, best_n_trials: int):
    trials = study.get_trials()
    sorted_trials = sorted(trials, key=lambda x: x.values, reverse=False)

    for trial in sorted_trials[:best_n_trials]:
        print(f'Trial {trial.number} params with value {trial.value}: {trial.params}')


def plot_params_for_best_n_trials(study: Study, best_n_trials: int, plot_output_path: str):
    trials = study.get_trials()
    sorted_trials = sorted(trials, key=lambda x: x.values, reverse=False)
    sorted_trials = sorted_trials[:best_n_trials]
    param_dicts = [trial.params for trial in sorted_trials]

    metric_column = 'metric'
    df = pd.DataFrame(param_dicts)
    df[metric_column] = [trial.value for trial in sorted_trials]
    hyperparameter_columns = [col for col in df.columns if col != metric_column]

    num_params = len(hyperparameter_columns)
    cols = 2
    rows = math.ceil(num_params / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 4))
    fig.suptitle(f'Hyperparameter Tuning Results - Best {best_n_trials}', fontsize=16)
    sns.set_theme(style="whitegrid")

    axes = axes.flatten()

    # Loop through each hyperparameter and plot it in the corresponding subplot
    for i, param in enumerate(hyperparameter_columns):
        # Continuous: Use a scatter plot
        sns.scatterplot(data=df, x=param, y=metric_column, ax=axes[i])
        axes[i].set_title(f'{param}')
        axes[i].set_xlabel(param)
        axes[i].set_ylabel(metric_column)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_output_path, f'best_hp_performance.pdf'))



def main():
    study_path = '/home/marcelhiltscher/Documents/Projects/Faithfulness_Test/experiments/GCN_HP_Norm_LR_Scheduler/hp_results.log'
    model_name = 'GCN'
    plot_title = f'{model_name} Hyperparameter Tuning'
    target_name = 'MSE'
    result_output_path = str(Path(study_path).parent.absolute())
    plot_output_path = os.path.join('/home/marcelhiltscher/Documents/Experiments/Hyperparameter_Optimization', plot_title.replace(' ', '_'))
    os.makedirs(plot_output_path, exist_ok=True)
    study = load_study(study_path=study_path)

    plot_slices(study=study, params=get_param_names(study.best_params), plot_output_path=plot_output_path,
                plot_title=plot_title, target_name=target_name)

    plot_importance(study=study, plot_output_path=plot_output_path)

    plot_history(study=study, plot_title=plot_title, plot_output_path=plot_output_path)

    plot_metric_for_trial(output_path=result_output_path, trial_number=study.best_trial.number, metric='rmse',
                          title_name=f'{model_name}: Best Trial', plot_output_path=plot_output_path)

    plot_trials_model_performance(output_path=result_output_path, study=study, best_n_trials=5,
                                  plot_output_path=plot_output_path)

    get_params_n_best_trials(study=study, best_n_trials=10)

    plot_params_for_best_n_trials(study=study, best_n_trials=10, plot_output_path=plot_output_path)


if __name__ == '__main__':
    main()
