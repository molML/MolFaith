import argparse
import os
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit

from mol_faith.data.types import DataCSVFileNames

ALPHA_VALUES = [0, 0.25, 0.50, 0.75, 1]


def create_stratified_train_val_test_data(data: DataFrame, seed: int, bin_method: str, target_column_index: int) \
        -> Tuple[DataFrame, DataFrame, DataFrame]:
    bin_column_name = 'bins'
    data_copy = data.copy()

    # First split: Training (70%) and Val-/Test-Set (30%)
    add_bins_to_data(data=data_copy, bin_method=bin_method, bin_column_name=bin_column_name,
                     target_column_index=target_column_index)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, temp_idx = next(sss1.split(data_copy, data_copy[bin_column_name]))

    train_df = data_copy.iloc[train_idx].drop(columns=[bin_column_name])
    temp_df = data_copy.iloc[temp_idx].drop(columns=[bin_column_name])

    # Second split: Validation (20%) and Test (10%)
    add_bins_to_data(data=temp_df, bin_method=bin_method, bin_column_name=bin_column_name,
                     target_column_index=target_column_index)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 3, random_state=seed)
    val_idx, test_idx = next(sss2.split(temp_df, temp_df[bin_column_name]))

    val_df = temp_df.iloc[val_idx].drop(columns=[bin_column_name])
    test_df = temp_df.iloc[test_idx].drop(columns=[bin_column_name])

    return train_df, val_df, test_df


def merge_small_bins(df: DataFrame, bin_column_name: str, min_samples: int = 2) -> None:
    while True:
        bin_counts = df[bin_column_name].value_counts().sort_index()
        small_bins = bin_counts[bin_counts < min_samples].index.to_list()

        if not small_bins:  # Exit when all bins have enough samples
            break

        for bin_id in small_bins:
            # Recalculate bin counts before each merge step
            bin_counts = df[bin_column_name].value_counts().sort_index()
            first_filled_bin = bin_counts.index[0]  # First non-empty bin
            last_filled_bin = bin_counts.index[-1]  # Last non-empty bin

            # Find the closest bin with enough samples
            possible_bins = bin_counts[bin_counts >= min_samples].index.to_list()
            if not possible_bins:
                raise ValueError("No valid bins found to merge small bins into.")

            if bin_id == first_filled_bin:  # Merge forward
                merge_target = possible_bins[0]  # First available bin
            elif bin_id == last_filled_bin:  # Merge backward
                merge_target = possible_bins[-1]  # Last available bin
            else:  # Merge with the closest bin
                larger_bins = [b for b in possible_bins if b > bin_id]
                smaller_bins = [b for b in possible_bins if b < bin_id]
                merge_target = min(larger_bins or [float('inf')]) if larger_bins else max(smaller_bins)

            # Merge small bin into selected target
            df.loc[df[bin_column_name] == bin_id, bin_column_name] = merge_target


def add_bins_to_data(data: DataFrame, bin_method: str, target_column_index: int, bin_column_name: str) -> None:
    bins = np.histogram_bin_edges(data.iloc[:, target_column_index], bins=bin_method)
    data[bin_column_name] = np.digitize(data.iloc[:, target_column_index], bins=bins, right=True)
    merge_small_bins(df=data, bin_column_name=bin_column_name)


def create_n_stratified_train_val_test_splits(base_output_path: str, num_splits: int, bin_method: str,
                                              target_column_index: int, input_column_index: int,
                                              data_path: Optional[str] = None,
                                              data: Optional[DataFrame] = None) -> None:
    if data_path is None and data is None:
        raise ValueError("Either data_path or data must be provided.")

    if data_path is not None and data is not None:
        raise ValueError("Only one of data_path or data must be provided.")

    if data_path:
        data = pd.read_csv(data_path)

    assert data is not None, 'Unexpected Error: data is None!'

    # Only select the input and target column -> Column 1: SMILES, Column 2: Target
    data = data.iloc[:, [input_column_index, target_column_index]].dropna()

    # Rename the columns
    column_names = ('SMILES', 'Target')
    data.rename(columns={old: new for old, new in zip(data.columns, column_names)}, inplace=True)

    for split_index in range(num_splits):
        path_to_data_split = os.path.join(base_output_path, f'data_split_{split_index}')
        os.makedirs(path_to_data_split, exist_ok=True)

        train_df, val_df, test_df = create_stratified_train_val_test_data(data=data, seed=split_index,
                                                                          target_column_index=1,
                                                                          bin_method=bin_method)

        train_df.to_csv(os.path.join(path_to_data_split, DataCSVFileNames.TRAIN.value), index=False)
        val_df.to_csv(os.path.join(path_to_data_split, DataCSVFileNames.VAL.value), index=False)
        test_df.to_csv(os.path.join(path_to_data_split, DataCSVFileNames.TEST.value), index=False)
