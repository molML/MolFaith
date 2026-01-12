import os
import pandas as pd
from pandas import Series
from typing import Callable

from mol_faith.data.create_dataset.create_stratified_data_splits import \
    create_n_stratified_train_val_test_splits


def min_max_scale(data: Series) -> pd.Series:
    return (data - data.min()) / (data.max() - data.min())


def generate_gt_data(data_path: str, output_path: str, count_substructure_func: Callable) -> None:
    os.makedirs(output_path, exist_ok=True)

    data = pd.read_csv(data_path)

    data['y_substructure'] = data['SMILES'].apply(count_substructure_func)
    data['y_substructure_scaled'] = min_max_scale(data['y_substructure'])

    new_data = data[['SMILES', 'y_substructure_scaled']]
    column_names = ('SMILES', 'Target')
    new_data = new_data.rename(
        columns={old: new for old, new in zip(new_data.columns, column_names)})

    create_n_stratified_train_val_test_splits(data=new_data, base_output_path=output_path, bin_method='fd',
                                              num_splits=5, input_column_index=0, target_column_index=1)
