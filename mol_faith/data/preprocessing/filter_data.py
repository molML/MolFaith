import argparse
import logging
import os
import pandas as pd
from pandas import DataFrame
from rdkit import Chem
from tqdm import tqdm
from typing import Optional, Tuple

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def check_sequence_length_filter(smiles: str, tokenizer: SMILESTokenizer, max_sequence_length: int) -> \
Tuple[bool, int]:
    """
    Check if SMILES passes sequence length filter.
    
    Args:
        smiles: SMILES string to check
        tokenizer: Tokenizer instance with tokenize method
        max_sequence_length: Maximum allowed sequence length
        
    Returns:
        Tuple of (passes_filter, sequence_length)
    """
    try:
        tokens = tokenizer.tokenize(smiles)
        sequence_length = len(tokens)
        passes = sequence_length <= max_sequence_length
        return passes, sequence_length
    except Exception as e:
        logger.error(f'Error tokenizing SMILES {smiles}: {e}')
        return False, -1


def check_heavy_atom_filter(smiles: str, max_heavy_atoms: int) -> Tuple[bool, int]:
    """
    Check if SMILES passes heavy atom count filter.
    
    Args:
        smiles: SMILES string to check
        max_heavy_atoms: Maximum allowed heavy atom count
        
    Returns:
        Tuple of (passes_filter, heavy_atom_count)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, -1

        heavy_atom_count = mol.GetNumHeavyAtoms()
        passes = heavy_atom_count <= max_heavy_atoms
        return passes, heavy_atom_count
    except Exception as e:
        logger.error(f'Error calculating heavy atoms for SMILES {smiles}: {e}')
        return False, -1


def _apply_molecular_filters(df: DataFrame,
                             tokenizer: SMILESTokenizer,
                             max_sequence_length: Optional[int],
                             max_heavy_atoms: Optional[int]) -> Tuple[DataFrame, DataFrame]:
    """
    Apply molecular filters to a DataFrame with SMILES.
    
    Args:
        df: Input DataFrame with SMILES column
        tokenizer: Tokenizer instance (required if max_sequence_length is provided)
        max_sequence_length: Maximum allowed sequence length
        max_heavy_atoms: Maximum allowed heavy atom count

    Returns:
        Tuple of (filtered_df, rejected_df)
    """
    logger.info(f'Applying molecular filters to {len(df)} molecules...')
    logger.info(f'  - Max sequence length: {max_sequence_length}')
    logger.info(f'  - Max heavy atoms: {max_heavy_atoms}')

    passed_indices = []
    rejected_data = []

    for idx, row in tqdm(df.iterrows(), desc='Applying filters', total=len(df)):
        smiles = row['SMILES']
        passes_all_filters = True
        rejection_reasons = []

        # Check sequence length filter
        passed, sequence_length = check_sequence_length_filter(
            smiles, tokenizer, max_sequence_length
        )

        if not passed:
            passes_all_filters = False
            rejection_reasons.append(f'seq_len>{max_sequence_length}')

        # Check heavy atom filter
        passed, heavy_atom_count = check_heavy_atom_filter(
            smiles, max_heavy_atoms
        )

        if not passed:
            passes_all_filters = False
            rejection_reasons.append(f'heavy_atoms>{max_heavy_atoms}')

        if passes_all_filters:
            passed_indices.append(idx)
        else:
            rejected_row = row.to_dict()
            rejected_row['sequence_length'] = sequence_length
            rejected_row['heavy_atom_count'] = heavy_atom_count
            rejected_row['rejection_reason'] = '; '.join(rejection_reasons)
            rejected_data.append(rejected_row)

    # Create filtered and rejected DataFrames
    filtered_df = df.loc[passed_indices].reset_index(drop=True)
    rejected_df = pd.DataFrame(rejected_data) if rejected_data else pd.DataFrame()

    # Log filtering results
    original_count = len(df)
    filtered_count = len(filtered_df)
    rejected_count = len(rejected_df)

    logger.info(f'Filtering results:')
    logger.info(f'  - Input molecules: {original_count}')
    logger.info(f'  - Passed filters: {filtered_count} ({filtered_count / original_count * 100:.1f}%)')
    logger.info(f'  - Rejected: {rejected_count} ({rejected_count / original_count * 100:.1f}%)')

    return filtered_df, rejected_df


def filter_molecular_data(input_data_path: str,
                          output_dir: str,
                          max_sequence_length: Optional[int],
                          max_heavy_atoms: Optional[int]) -> None:
    """
    Filter molecular data based on sequence length and/or heavy atom count.
    
    Args:
        input_data_path: Path to input CSV file (typically cleaned_data.csv)
        output_dir: Directory to save filtered data and reports
        max_sequence_length: Maximum allowed sequence length
        max_heavy_atoms: Maximum allowed heavy atom count
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    logger.info(f'Loading data from {input_data_path}')
    df = pd.read_csv(input_data_path)

    if 'SMILES' not in df.columns:
        raise ValueError(f'Input file must contain SMILES column named SMILES')

    original_count = len(df)
    logger.info(f'Loaded {original_count} molecules')

    tokenizer = SMILESTokenizer()

    filtered_df, rejected_df = _apply_molecular_filters(
        df=df,
        tokenizer=tokenizer,
        max_sequence_length=max_sequence_length,
        max_heavy_atoms=max_heavy_atoms
    )

    filtered_output = os.path.join(output_dir, 'filtered_data.csv')
    filtered_df.to_csv(filtered_output, index=False)
    logger.info(f'Filtered data saved to: {filtered_output}')

    if len(rejected_df) > 0:
        rejected_output = os.path.join(output_dir, 'rejected_molecules.csv')
        rejected_df.to_csv(rejected_output, index=False)
        logger.info(f'Rejected molecules saved to: {rejected_output}')

    # Create filtering summary
    final_count = len(filtered_df)
    rejected_count = len(rejected_df)

    summary = {
        'input_molecules': original_count,
        'filtered_molecules': final_count,
        'rejected_molecules': rejected_count,
        'retention_rate': f'{(final_count / original_count) * 100:.2f}%',
        'rejection_rate': f'{(rejected_count / original_count) * 100:.2f}%',
        'max_sequence_length': max_sequence_length,
        'max_heavy_atoms': max_heavy_atoms
    }

    # Add filter-specific statistics
    if len(rejected_df) > 0:
        if 'sequence_length' in rejected_df.columns:
            seq_violations = rejected_df['rejection_reason'].str.contains('seq_len>', na=False).sum()
            summary['rejected_by_sequence_length'] = int(seq_violations)

        if 'heavy_atom_count' in rejected_df.columns:
            atom_violations = rejected_df['rejection_reason'].str.contains('heavy_atoms>', na=False).sum()
            summary['rejected_by_heavy_atoms'] = int(atom_violations)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, 'filtering_summary.csv'), index=False)

    logger.info('Filtering complete!')
    logger.info(f'Results: {original_count} → {final_count} molecules ({summary['retention_rate']} retained)')


def get_args():
    parser = argparse.ArgumentParser(description='Molecular Data Filtering')

    parser.add_argument('--input_data', type=str, required=True,
                        help='Path to input CSV file (e.g., cleaned_data.csv)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where filtered data will be saved')

    # Filtering parameters
    parser.add_argument('--max_sequence_length', type=int, default=None,
                        help='Maximum allowed sequence length for tokenized SMILES')
    parser.add_argument('--max_heavy_atoms', type=int, default=None,
                        help='Maximum allowed number of heavy atoms')

    return parser.parse_args()


def main():
    args = get_args()

    filter_molecular_data(
        input_data_path=args.input_data,
        output_dir=args.output_dir,
        max_sequence_length=args.max_sequence_length,
        max_heavy_atoms=args.max_heavy_atoms
    )


if __name__ == '__main__':
    main()
