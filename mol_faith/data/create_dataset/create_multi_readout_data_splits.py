import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit

from mol_faith.data.create_dataset.create_stratified_data_splits import merge_small_bins
from mol_faith.data.types import DataCSVFileNames

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Fractions of the two-stage split. Stage 1 holds out 30%, stage 2 splits that into
# 20% val / 10% test -> 70/20/10 overall (same as the single-target splitter).
TEST_AND_VAL_FRACTION = 0.3
TEST_FRACTION_OF_REMAINDER = 1 / 3

# The test set is the smallest of the three splits, so it dictates how many molecules a
# joint cell needs: sklearn requires at least MIN_SAMPLES_PER_CLASS members of every
# class to end up in a split.
MIN_SAMPLES_PER_CLASS = 2
TEST_FRACTION_OF_TOTAL = TEST_AND_VAL_FRACTION * TEST_FRACTION_OF_REMAINDER
DEFAULT_MIN_JOINT_CELL_COUNT = int(MIN_SAMPLES_PER_CLASS / TEST_FRACTION_OF_TOTAL)


def get_bin_column_name(readout_column_name: str) -> str:
    return f'_{readout_column_name}_bin'


def add_quantile_bins(data: DataFrame, readout_column_names: Sequence[str],
                      n_bins_per_readout: int) -> Tuple[List[str], Dict[str, List[float]]]:
    """Add one quantile bin column per readout and merge undersized bins per readout.

    Quantile (equal-count) bins are used rather than the equal-width bins of the
    single-target splitter: equal-width bins are lopsided for skewed readouts, and the
    cross-product of several lopsided binnings leaves cells with too few molecules.

    Returns:
        The names of the added bin columns, and the bin edges per readout. Fewer bins
        than requested may result when tied values force `duplicates='drop'`, which the
        returned edges make visible.
    """
    bin_column_names = []
    edges_per_readout = {}

    for readout_column_name in readout_column_names:
        bin_column_name = get_bin_column_name(readout_column_name)

        labels, edges = pd.qcut(data[readout_column_name], q=n_bins_per_readout, labels=False,
                                retbins=True, duplicates='drop')
        data[bin_column_name] = labels
        merge_small_bins(df=data, bin_column_name=bin_column_name)

        bin_column_names.append(bin_column_name)
        edges_per_readout[readout_column_name] = [float(edge) for edge in edges]

    return bin_column_names, edges_per_readout


def get_joint_cell_counts(data: DataFrame, bin_column_names: Sequence[str]) -> pd.Series:
    """Number of molecules per occupied joint cell.

    `groupby` omits combinations that no molecule falls into, which is exactly what we
    want: empty cells are harmless, only occupied-but-thin cells break the split.
    """
    return data.groupby(list(bin_column_names)).size()


def _find_largest_workable_bin_count(data: DataFrame, readout_column_names: Sequence[str],
                                     n_bins_per_readout: int,
                                     min_joint_cell_count: int) -> Optional[int]:
    """Largest bin count below `n_bins_per_readout` whose joint cells are all big enough."""
    for candidate_bin_count in range(n_bins_per_readout - 1, 1, -1):
        probe = data[list(readout_column_names)].copy()
        bin_column_names, _ = add_quantile_bins(data=probe, readout_column_names=readout_column_names,
                                                n_bins_per_readout=candidate_bin_count)
        if get_joint_cell_counts(data=probe, bin_column_names=bin_column_names).min() >= min_joint_cell_count:
            return candidate_bin_count
    return None


def validate_joint_strata(data: DataFrame, readout_column_names: Sequence[str],
                          bin_column_names: Sequence[str], n_bins_per_readout: int,
                          min_joint_cell_count: int = DEFAULT_MIN_JOINT_CELL_COUNT) -> None:
    """Raise if an occupied joint cell is too thin to survive the 70/20/10 split.

    Binning each readout on its own cannot guarantee usable joint cells: joint sparsity
    comes from correlation between the readouts, which per-readout binning cannot see
    (both marginals can look perfectly healthy while off-trend molecules land in cells
    holding a handful of entries).
    """
    joint_cell_counts = get_joint_cell_counts(data=data, bin_column_names=bin_column_names)
    thin_cells = joint_cell_counts[joint_cell_counts < min_joint_cell_count]

    if thin_cells.empty:
        logger.info(f'Stratifying on {len(joint_cell_counts)} occupied joint cells '
                    f'(smallest holds {joint_cell_counts.min()} molecules).')
        return

    workable_bin_count = _find_largest_workable_bin_count(
        data=data, readout_column_names=readout_column_names,
        n_bins_per_readout=n_bins_per_readout, min_joint_cell_count=min_joint_cell_count)

    if workable_bin_count is None:
        suggestion = ('No bin count down to 2 satisfies the threshold. Inspect the readouts for '
                      'outliers, or stratify on a single readout instead.')
    else:
        suggestion = f'Retry with n_bins_per_readout={workable_bin_count}.'

    raise ValueError(
        f'{len(thin_cells)} joint cell(s) hold fewer than {min_joint_cell_count} molecules, which is '
        f'too few for {MIN_SAMPLES_PER_CLASS} to reach the test set '
        f'({TEST_FRACTION_OF_TOTAL:.0%} of the data).\n'
        f'Bin columns {list(bin_column_names)} at n_bins_per_readout={n_bins_per_readout}.\n'
        f'Thin cells (bins -> molecules): {thin_cells.to_dict()}\n'
        f'{suggestion}'
    )


def _assert_splittable(strata: DataFrame, n_split_samples: int, description: str) -> None:
    """Pre-check sklearn's stratification requirements with a legible error message."""
    class_counts = strata.value_counts()

    too_small = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]
    if not too_small.empty:
        raise ValueError(f'Cannot {description}: {len(too_small)} stratum/strata hold fewer than '
                         f'{MIN_SAMPLES_PER_CLASS} molecules: {too_small.to_dict()}')

    if len(class_counts) > n_split_samples:
        raise ValueError(f'Cannot {description}: {len(class_counts)} strata do not fit into '
                         f'{n_split_samples} samples.')


def create_stratified_multi_readout_split(data: DataFrame, seed: int, bin_column_names: Sequence[str]
                                          ) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Split into train/val/test (70/20/10), stratified jointly on all bin columns.

    All bin columns are handed to `StratifiedShuffleSplit` together; it maps each distinct
    combination of bin values to one class, so the cross-product of the readouts is what
    gets balanced. The bin columns are removed from the returned frames.
    """
    bin_column_names = list(bin_column_names)

    # First split: Training (70%) and Val-/Test-Set (30%)
    _assert_splittable(strata=data[bin_column_names],
                       n_split_samples=round(len(data) * TEST_AND_VAL_FRACTION),
                       description='split off the validation/test set')
    first_split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_AND_VAL_FRACTION, random_state=seed)
    train_idx, val_and_test_idx = next(first_split.split(data, data[bin_column_names]))

    train_df = data.iloc[train_idx]
    val_and_test_df = data.iloc[val_and_test_idx]

    # Second split: Validation (20%) and Test (10%)
    _assert_splittable(strata=val_and_test_df[bin_column_names],
                       n_split_samples=round(len(val_and_test_df) * TEST_FRACTION_OF_REMAINDER),
                       description='split the held-out set into validation and test')
    second_split = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRACTION_OF_REMAINDER,
                                          random_state=seed)
    val_idx, test_idx = next(second_split.split(val_and_test_df, val_and_test_df[bin_column_names]))

    val_df = val_and_test_df.iloc[val_idx]
    test_df = val_and_test_df.iloc[test_idx]

    return tuple(split_df.drop(columns=bin_column_names) for split_df in (train_df, val_df, test_df))


def _prepare_data(data: DataFrame, smiles_column_name: str,
                  readout_column_names: Sequence[str]) -> DataFrame:
    """Keep only the SMILES and readout columns, drop incomplete rows, rename to 'SMILES'."""
    missing_columns = [column_name for column_name in (smiles_column_name, *readout_column_names)
                       if column_name not in data.columns]
    if missing_columns:
        raise ValueError(f'Columns {missing_columns} not found in the data. '
                         f'Available columns: {data.columns.tolist()}')

    # SMILES must stay the first column: GraphDataManager reads it positionally.
    prepared_data = data[[smiles_column_name, *readout_column_names]].dropna().reset_index(drop=True)

    return prepared_data.rename(columns={smiles_column_name: 'SMILES'})


def _write_readout_split(split_dfs: Tuple[DataFrame, DataFrame, DataFrame], output_path: str,
                         readout_column_name: str) -> None:
    """Write one readout's train/val/test CSVs as two 'SMILES,Target' columns."""
    os.makedirs(output_path, exist_ok=True)

    csv_file_names = (DataCSVFileNames.TRAIN.value, DataCSVFileNames.VAL.value,
                      DataCSVFileNames.TEST.value)

    for split_df, csv_file_name in zip(split_dfs, csv_file_names):
        readout_df = split_df[['SMILES', readout_column_name]].rename(
            columns={readout_column_name: 'Target'})
        readout_df.to_csv(os.path.join(output_path, csv_file_name), index=False)


def create_n_multi_readout_splits(base_output_path: str,
                                  num_splits: int,
                                  readout_column_names: Sequence[str],
                                  smiles_column_name: str = 'SMILES',
                                  n_bins_per_readout: int = 4,
                                  min_joint_cell_count: int = DEFAULT_MIN_JOINT_CELL_COUNT,
                                  output_folder_names: Optional[Dict[str, str]] = None,
                                  data_path: Optional[str] = None,
                                  data: Optional[DataFrame] = None) -> None:
    """Create `num_splits` train/val/test splits stratified on all readouts jointly.

    Every split is written once per readout, as two-column 'SMILES,Target' CSVs, with
    each readout in its own folder:

        <base_output_path>/<readout folder>/data_split_<i>/{train,val,test}.csv

    The molecules and their order are identical across the readout directories - only the
    Target values differ - so single-target models trained from different directories are
    trained on exactly the same split. Keeping the column named 'Target' also means
    `target_column_name: Target` in the training configs stays valid.

    Args:
        base_output_path: Directory the per-readout split directories are created in.
        num_splits: Number of splits; split i uses i as its random seed.
        readout_column_names: Continuous columns to stratify on jointly.
        smiles_column_name: SMILES column in the input data.
        n_bins_per_readout: Quantile bins per readout. Never lowered automatically; too
            few molecules per joint cell raises instead.
        min_joint_cell_count: Smallest acceptable number of molecules in an occupied
            joint cell.
        output_folder_names: Output folder name per readout, e.g.
            {'pKi': 'pki_data_splits', 'LogD (pH 7.40)': 'logd_data_splits'}. Useful to
            keep paths free of the spaces and parentheses in readout names. Defaults to
            the readout column name.
        data_path: Path to the input CSV. Mutually exclusive with `data`.
        data: Input dataframe. Mutually exclusive with `data_path`.

    Note:
        Rows missing ANY of the readouts are dropped, so the splits cover the molecules
        for which all readouts are available.
    """
    if data_path is None and data is None:
        raise ValueError("Either data_path or data must be provided.")

    if data_path is not None and data is not None:
        raise ValueError("Only one of data_path or data must be provided.")

    if data_path:
        data = pd.read_csv(data_path)

    assert data is not None, 'Unexpected Error: data is None!'

    output_folder_names = output_folder_names or {}

    data = _prepare_data(data=data, smiles_column_name=smiles_column_name,
                         readout_column_names=readout_column_names)
    logger.info(f'Splitting {len(data)} molecules stratified on {list(readout_column_names)} '
                f'with {n_bins_per_readout} quantile bins per readout.')

    # Bin and validate once on the full dataset so every split uses the same strata.
    bin_column_names, edges_per_readout = add_quantile_bins(
        data=data, readout_column_names=readout_column_names, n_bins_per_readout=n_bins_per_readout)
    validate_joint_strata(data=data, readout_column_names=readout_column_names,
                          bin_column_names=bin_column_names, n_bins_per_readout=n_bins_per_readout,
                          min_joint_cell_count=min_joint_cell_count)

    for readout_column_name, edges in edges_per_readout.items():
        logger.info(f'Bin edges for {readout_column_name}: {[round(edge, 4) for edge in edges]}')

    for split_index in range(num_splits):
        split_dfs = create_stratified_multi_readout_split(data=data, seed=split_index,
                                                          bin_column_names=bin_column_names)

        for readout_column_name in readout_column_names:
            output_folder_name = output_folder_names.get(readout_column_name, readout_column_name)
            output_path = os.path.join(base_output_path, output_folder_name,
                                       f'data_split_{split_index}')
            _write_readout_split(split_dfs=split_dfs, output_path=output_path,
                                 readout_column_name=readout_column_name)

        logger.info(f'Split {split_index}: {[len(split_df) for split_df in split_dfs]} '
                    f'molecules in train/val/test.')
