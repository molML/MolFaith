from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

from rdkit import Chem

from pathlib import Path
import pandas as pd

from mol_faith.target_extraction.benzene_substructure_extraction import BenzeneSubstructureExtractor
from mol_faith.target_extraction.hbond_donor_extraction import HBondDonorExtractor
from mol_faith.target_extraction.largest_conjugated_system_extraction import LargestConjugatedSystemExtractor

def load_raw_data(
        base_dir,
        target_name,
        models=['GCN', 'Transformer', 'CNN', 'GIN', 'GAT'],
        threshold='0.2',
        num_splits=5,
        exclude_methods=[],
):
    """
    Load fidelity data from all models, auto-discovering available XAI methods,
    and add a column with the target name.

    Path structure:
        base_dir/model/f_fid_eval/beta_0.1/top_percentage/xai_method/threshold_X.X/data_split_X/fidelity_results.csv

    Parameters
    ----------
    base_dir : str
        Base directory containing model folders.
    target : str
        Target name to tag the loaded data with.
    models : list
        Model names to load (default: ['GCN', 'Transformer', 'CNN', 'GIN', 'GAT']).
    threshold : str
        Threshold value (e.g., '0.2').
    num_splits : int
        Number of data splits (0 to num_splits-1).
    exclude_methods : list
        XAI methods to exclude (e.g., baselines).

    Returns
    -------
    pd.DataFrame
        Combined dataframe with fidelity data, including a 'target' column.
    """
    base_path = Path(base_dir)
    all_dataframes = []

    # Auto-discover available XAI methods
    all_xai_methods = set()
    for model_name in models:
        model_path = base_path / model_name / 'f_fid_eval' / 'beta_0.1' / 'top_percentage'
        if model_path.exists():
            methods = [d.name for d in model_path.iterdir()
                       if d.is_dir() and d.name not in exclude_methods]
            all_xai_methods.update(methods)

    xai_methods = sorted(list(all_xai_methods))
    print(f"Found XAI methods: {xai_methods}")
    print(f"Excluded methods: {exclude_methods}")

    # Load data for each model and XAI method
    for model_name in models:
        print(f"\nProcessing {model_name}...")
        model_dataframes = []

        for xai_method in xai_methods:
            method_dataframes = []

            for split_idx in range(num_splits):
                file_path = (
                    base_path / model_name / 'f_fid_eval' / 'beta_0.1' / 'top_percentage' /
                    xai_method / f"threshold_{threshold}" / f"data_split_{split_idx}" / "fidelity_results.csv"
                )

                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        df = df.dropna(subset=['fid_plus'])  # Remove invalid rows
                        
                        # Add metadata
                        df['model'] = model_name
                        df['explanation_method'] = xai_method
                        df['target_name'] = target_name 

                        method_dataframes.append(df)
                    except Exception as e:
                        print(f"  Error loading {xai_method} split {split_idx}: {e}")

            if method_dataframes:
                method_combined = pd.concat(method_dataframes, ignore_index=True)
                model_dataframes.append(method_combined)
                print(f"  {xai_method}: {len(method_combined)} rows")

        if model_dataframes:
            model_combined = pd.concat(model_dataframes, ignore_index=True)
            all_dataframes.append(model_combined)
            print(f"  {model_name} total: {len(model_combined)} rows")

    # Combine all models into one final DataFrame
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        final_df = final_df[['target_name', 'model', 'smiles', 'data_split_idx', 'target', 'explanation_method', 'fid_plus']].copy()

        print(f"\n✅ Final dataset for target '{target_name}': {len(final_df):,} rows")
        print(f"Models: {sorted(final_df['model'].unique())}")
        print(f"XAI methods: {sorted(final_df['explanation_method'].unique())}")
        return final_df
    else:
        print(f"No data loaded for target '{target_name}'!")
        return pd.DataFrame()


def load_raw_gt_data(
        base_dir,
        target_name,
        models=['GCN', 'Transformer', 'CNN', 'GIN', 'GAT'],
        num_splits=5,
        exclude_methods=[]):
    base_path = Path(base_dir)
    all_dataframes = []

    # Auto-discover XAI methods and targets
    all_xai_methods = set()
    all_targets = set()

    for model_name in models:
        model_gt_path = base_path / model_name / 'gt_eval'
        if model_gt_path.exists():
            # Discover target folders
            for target_dir in model_gt_path.iterdir():
                if target_dir.is_dir():
                    all_targets.add(target_dir.name)
                    # Discover XAI methods within each target
                    for method_dir in target_dir.iterdir():
                        if (method_dir.is_dir() and
                                method_dir.name not in exclude_methods):
                            all_xai_methods.add(method_dir.name)

    xai_methods = sorted(list(all_xai_methods))
    targets = sorted(list(all_targets))

    print(f"Found targets: {targets}")
    print(f"Found XAI methods: {xai_methods}")
    print(f"Excluded: {exclude_methods}")

    # Load data
    for model_name in models:
        print(f"\nProcessing {model_name}...")
        model_dataframes = []

        for target in targets:
            for xai_method in xai_methods:
                method_dataframes = []

                for split_idx in range(num_splits):
                    file_path = (base_path / model_name / 'gt_eval' / target /
                                 xai_method / f"data_split_{split_idx}" / "target_fraction_results.csv")

                    if file_path.exists():
                        try:
                            df = pd.read_csv(file_path)
                            # Filter out rows with invalid ground truth data
                            df = df.dropna(subset=['metadata_target_fraction', 'metadata_precision_at_k_gt'])

                            # Add metadata
                            df['model'] = model_name
                            df['target_name'] = target_name
                            method_dataframes.append(df)
                        except Exception as e:
                            print(f"  Error loading {target}/{xai_method} split {split_idx}: {e}")

                if method_dataframes:
                    method_combined = pd.concat(method_dataframes, ignore_index=True)
                    model_dataframes.append(method_combined)
                    print(f"  {target}/{xai_method}: {len(method_combined)} rows")

        if model_dataframes:
            model_combined = pd.concat(model_dataframes, ignore_index=True)
            all_dataframes.append(model_combined)
            print(f"  {model_name} total: {len(model_combined)} rows")

    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        # Select only the specified columns
        final_df = final_df[['target_name', 'model', 'smiles', 'data_split_idx', 'target', 'explanation_method',
                             'metadata_target_fraction', 'metadata_precision_at_k_gt']].copy()

        print(f"\nFinal dataset: {len(final_df):,} rows")
        print(f"Models: {sorted(final_df['model'].unique())}")
        print(f"XAI methods: {sorted(final_df['explanation_method'].unique())}")
        print(f"Targets processed: {len(targets)}")

        return final_df
    else:
        print("No data loaded!")
        return pd.DataFrame()


def get_model_colors():
    return {
        'Transformer': '#2D4976',
        'CNN': '#A0B1CF',
        'GAT': '#D6CAAE',
        'GCN': '#928463',
        'GIN': '#C09C5D'
    }


def get_xai_colors():
    return {
        'grad_cam': '#c03221',
        'guided_grad_cam': '#46579B',
        'integrated_gradients': '#f3a712',
        'kernel_shap': '#f0cea0',
        'lime': '#6d8549',
        'shap_sampling': '#E0EFBD',
        'gnn_explainer': '#ABA6B5',
        'graph_mask': '#8bbf9f'
    }


def get_target_colors():
    return {
        'Benzene': '#F0CEA0',
        'Conjugated System': '#9D8189',
        'Hbond Donor': '#569A9F'
    }


def map_explanation_method_names(df):
    name_mapping = {
        'grad_cam': 'GradCAM',
        'guided_grad_cam': 'Guided GradCAM',
        'integrated_gradients': 'IG',
        'kernel_shap': 'KernelSHAP',
        'lime': 'LIME',
        'shap_sampling': 'SHAP',
        'gnn_explainer': 'GNNExplainer',
        'graph_mask': 'GraphMask'
    }

    df_copy = df.copy()
    df_copy['explanation_method'] = df_copy['explanation_method'].map(name_mapping)
    return df_copy


def process_fidelity_df(df, epsilon=1e-10):
    """
    Process fidelity dataframe with unified faithfulness score.
    
    Uses two-step formula:
    1. R = (fid_xai - fid_random) / (|fid_xai| + |fid_random| + ε)
    2. S = R if fid_xai ≥ 0, else S = -|R|
    
    This ensures negative XAI F-Fid+ → negative faithfulness score.
    
    Args:
        df: Raw dataframe with columns [model, smiles, data_split_idx, 
                                        explanation_method, fid_plus, target]
        epsilon: Small constant for numerical stability
    
    Returns:
        Processed dataframe with normalized_fid (faithfulness score)
    """
    
    # Separate random baseline
    df_random = df[df['explanation_method'] == 'random_explanation'][
        ['model', 'smiles', 'data_split_idx', 'target', 'fid_plus']
    ].rename(columns={'fid_plus': 'fid_plus_random'})
    
    # XAI methods only
    df_xai = df[df['explanation_method'] != 'random_explanation'].copy()
    
    # Merge with random baseline
    df_result = df_xai.merge(
        df_random, 
        on=['model', 'smiles', 'data_split_idx', 'target'], 
        how='left'
    )
    
    # Step 1: Compute R with absolute denominator (vectorized)
    R = (df_result['fid_plus'] - df_result['fid_plus_random']) / \
        (np.abs(df_result['fid_plus']) + np.abs(df_result['fid_plus_random']) + epsilon)
    
    # Step 2: Apply sign correction based on XAI regime (vectorized)
    df_result['normalized_fid'] = np.where(
        df_result['fid_plus'] >= 0,
        R,           # Faithful regime: keep R
        -np.abs(R)   # Unfaithful regime: force negative
    )
    
    # Add regime categorization
    df_result['regime'] = np.where(
        df_result['fid_plus'] >= 0,
        'faithful',
        'unfaithful'
    )
    
    # Statistics
    print(f"Total samples: {len(df_result)}")
    print(f"\nRegime distribution:")
    print(df_result['regime'].value_counts())
    
    print(f"\nFaithfulness statistics by regime:")
    regime_stats = df_result.groupby('regime').agg({
        'normalized_fid': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'fid_plus': ['mean', 'min', 'max']
    }).round(4)
    print(regime_stats)
    
    print(f"\nOverall faithfulness score range: "
          f"[{df_result['normalized_fid'].min():.3f}, "
          f"{df_result['normalized_fid'].max():.3f}]")
    
    return df_result


def calculate_total_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol is not None else None

        
def extract_gt_atoms_number(smiles, target):
    extractors = {
        'benzene': BenzeneSubstructureExtractor().get_atom_and_bond_indices,
        'conjugated_system': LargestConjugatedSystemExtractor().get_atom_and_bond_indices,
        'hbond_donor': HBondDonorExtractor().get_atom_and_bond_indices
    }
    extractor = extractors[target]
    mol = Chem.MolFromSmiles(smiles)
    gt_atoms_num = extractor(mol)[0]
    return len(np.array(gt_atoms_num).flatten())


def load_raw_data_all_thres(
    base_dir,
    target_name,
    models=['GCN', 'Transformer', 'CNN', 'GIN', 'GAT'],
    thresholds=None,  # Changed: None means auto-discover all thresholds
    num_splits=5,
    exclude_methods=[],
    verbose=True,
):
    """
    Load fidelity data from all models, auto-discovering available XAI methods and thresholds,
    and add a column with the target name.

    Path structure:
        base_dir/model/f_fid_eval/beta_0.1/top_percentage/xai_method/threshold_X.X/data_split_X/fidelity_results.csv

    Parameters
    ----------
    base_dir : str
        Base directory containing model folders.
    target_name : str
        Target name to tag the loaded data with.
    models : list
        Model names to load (default: ['GCN', 'Transformer', 'CNN', 'GIN', 'GAT']).
    thresholds : list or None
        List of threshold values (e.g., ['0.1', '0.2']) or None to auto-discover all.
    num_splits : int
        Number of data splits (0 to num_splits-1).
    exclude_methods : list
        XAI methods to exclude (e.g., baselines).
    verbose : bool
        Whether to print detailed progress information.

    Returns
    -------
    pd.DataFrame
        Combined dataframe with fidelity data, including 'target_name' and 'threshold' columns.
    """
    base_path = Path(base_dir)
    all_dataframes = []

    # Auto-discover available XAI methods and thresholds
    all_xai_methods = set()
    all_thresholds = set()
    
    for model_name in models:
        model_path = base_path / model_name / 'f_fid_eval' / 'beta_0.1' / 'top_percentage'
        if model_path.exists():
            # Discover XAI methods
            methods = [d.name for d in model_path.iterdir()
                       if d.is_dir() and d.name not in exclude_methods]
            all_xai_methods.update(methods)
            
            # Discover thresholds for each method
            for method in methods:
                method_path = model_path / method
                if method_path.exists():
                    threshold_dirs = [d.name for d in method_path.iterdir() 
                                    if d.is_dir() and d.name.startswith('threshold_')]
                    # Extract threshold values (e.g., 'threshold_0.2' -> '0.2')
                    threshold_values = [d.replace('threshold_', '') for d in threshold_dirs]
                    all_thresholds.update(threshold_values)

    xai_methods = sorted(list(all_xai_methods))
    discovered_thresholds = sorted(list(all_thresholds), key=lambda x: float(x))
    
    # Use provided thresholds or auto-discovered ones
    if thresholds is None:
        thresholds = discovered_thresholds
        if verbose:
            print(f"Auto-discovered thresholds: {thresholds}")
    else:
        # Validate provided thresholds exist
        provided_set = set(thresholds)
        available_set = set(discovered_thresholds)
        missing = provided_set - available_set
        if missing:
            print(f"⚠️ Warning: Requested thresholds {missing} not found in data")
            print(f"   Available thresholds: {discovered_thresholds}")
        thresholds = [t for t in thresholds if t in available_set]
    
    if verbose:
        print(f"Loading thresholds: {thresholds}")
        print(f"Found XAI methods: {xai_methods}")
        print(f"Excluded methods: {exclude_methods}")

    # Load data for each model, XAI method, and threshold
    for model_name in models:
        if verbose:
            print(f"\nProcessing {model_name}...")
        model_dataframes = []

        for xai_method in xai_methods:
            method_dataframes = []

            for threshold in thresholds:
                threshold_dataframes = []
                
                for split_idx in range(num_splits):
                    file_path = (
                        base_path / model_name / 'f_fid_eval' / 'beta_0.1' / 'top_percentage' /
                        xai_method / f"threshold_{threshold}" / f"data_split_{split_idx}" / "fidelity_results.csv"
                    )

                    if file_path.exists():
                        try:
                            df = pd.read_csv(file_path)
                            df = df.dropna(subset=['fid_plus'])  # Remove invalid rows

                            # Add metadata
                            df['model'] = model_name
                            df['explanation_method'] = xai_method
                            df['target_name'] = target_name
                            df['threshold'] = float(threshold)  # Add threshold as numeric
                            df['data_split_idx'] = split_idx  # Ensure split index is recorded

                            threshold_dataframes.append(df)
                        except Exception as e:
                            if verbose:
                                print(f"  Error loading {xai_method} threshold {threshold} split {split_idx}: {e}")

                if threshold_dataframes:
                    threshold_combined = pd.concat(threshold_dataframes, ignore_index=True)
                    method_dataframes.append(threshold_combined)
                    if verbose:
                        print(f"  {xai_method} @ threshold {threshold}: {len(threshold_combined)} rows")

            if method_dataframes:
                method_combined = pd.concat(method_dataframes, ignore_index=True)
                model_dataframes.append(method_combined)
                if verbose:
                    print(f"  {xai_method} total: {len(method_combined)} rows across {len(thresholds)} thresholds")

        if model_dataframes:
            model_combined = pd.concat(model_dataframes, ignore_index=True)
            all_dataframes.append(model_combined)
            if verbose:
                print(f"  {model_name} total: {len(model_combined)} rows")

    # Combine all models into one final DataFrame
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Reorder columns for better readability
        column_order = ['target_name', 'model', 'smiles', 'data_split_idx', 
                       'threshold', 'target', 'explanation_method', 'fid_plus']
        
        # Add any additional columns that might exist (like fid_minus)
        existing_cols = final_df.columns.tolist()
        for col in existing_cols:
            if col not in column_order:
                column_order.append(col)
        
        # Only include columns that actually exist in the dataframe
        column_order = [col for col in column_order if col in existing_cols]
        final_df = final_df[column_order].copy()

        if verbose:
            print(f"\n✅ Final dataset for target '{target_name}':")
            print(f"   Total rows: {len(final_df):,}")
            print(f"   Models: {sorted(final_df['model'].unique())}")
            print(f"   XAI methods: {sorted(final_df['explanation_method'].unique())}")
            print(f"   Thresholds: {sorted(final_df['threshold'].unique())}")
            print(f"   Data splits: {sorted(final_df['data_split_idx'].unique())}")
            
            # Summary statistics
            print(f"\n📊 Data distribution:")
            print(f"   Rows per model: {final_df.groupby('model').size().to_dict()}")
            print(f"   Rows per threshold: {final_df.groupby('threshold').size().to_dict()}")
        
        return final_df
    else:
        print(f"❌ No data loaded for target '{target_name}'!")
        return pd.DataFrame()


def process_fidelity_df_all_thres(df, epsilon=1e-10):
    """
    Process fidelity dataframe with unified faithfulness score.
    
    Uses two-step formula:
    1. R = (fid_xai - fid_random) / (|fid_xai| + |fid_random| + ε)
    2. S = R if fid_xai ≥ 0, else S = -|R|
    
    This ensures negative XAI F-Fid+ → negative faithfulness score.
    
    Args:
        df: Raw dataframe with columns [model, smiles, data_split_idx, 
                                        explanation_method, fid_plus, target]
        epsilon: Small constant for numerical stability
    
    Returns:
        Processed dataframe with normalized_fid (faithfulness score)
    """
    
    # Separate random baseline
    df_random = df[df['explanation_method'] == 'random_explanation'][
        ['model', 'smiles', 'data_split_idx', 'target', 'threshold', 'fid_plus']
    ].rename(columns={'fid_plus': 'fid_plus_random'})
    
    # XAI methods only
    df_xai = df[df['explanation_method'] != 'random_explanation'].copy()
    
    # Merge with random baseline
    df_result = df_xai.merge(
        df_random, 
        on=['model', 'smiles', 'data_split_idx', 'target', 'threshold'], 
        how='left'
    )
    
    # Step 1: Compute R with absolute denominator (vectorized)
    R = (df_result['fid_plus'] - df_result['fid_plus_random']) / \
        (np.abs(df_result['fid_plus']) + np.abs(df_result['fid_plus_random']) + epsilon)
    
    # Step 2: Apply sign correction based on XAI regime (vectorized)
    df_result['normalized_fid'] = np.where(
        df_result['fid_plus'] >= 0,
        R,           # Faithful regime: keep R
        -np.abs(R)   # Unfaithful regime: force negative
    )
    
    # Add regime categorization
    df_result['regime'] = np.where(
        df_result['fid_plus'] >= 0,
        'faithful',
        'unfaithful'
    )
    
    # Statistics
    print(f"Total samples: {len(df_result)}")
    print(f"\nRegime distribution:")
    print(df_result['regime'].value_counts())
    
    print(f"\nFaithfulness statistics by regime:")
    regime_stats = df_result.groupby('regime').agg({
        'normalized_fid': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'fid_plus': ['mean', 'min', 'max']
    }).round(4)
    print(regime_stats)
    
    print(f"\nOverall faithfulness score range: "
          f"[{df_result['normalized_fid'].min():.3f}, "
          f"{df_result['normalized_fid'].max():.3f}]")
    
    return df_result