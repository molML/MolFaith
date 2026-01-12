import argparse
import logging
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import pandas as pd
from pandas import DataFrame
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

type CleanedSmilesContainer = Dict[str, List[str]]


@dataclass
class CleaningResult:
    mol: Optional[Mol] = None
    processed_smiles: Optional[str] = None
    failed_step: Optional[str] = None


def use_rdkit_cleanup(mol: Mol) -> CleaningResult:
    # RemoveHs, Sanitize,  Metaldisconnector, normalizer, reionizer,  asssign stereochemistry
    step_name = 'rdkit Cleanup'
    try:
        mol = rdMolStandardize.Cleanup(mol)

        if mol is None:
            return CleaningResult(failed_step=step_name)

        return CleaningResult(mol=mol)
    except Exception as e:
        logger.error(f'There was an error while calling rdMolStandardize.Cleanup: {e} \n'
                     f'Skip molecule {Chem.MolToSmiles(mol)}')
        return CleaningResult(failed_step=step_name)


def remove_salts(mol: Mol) -> CleaningResult:
    # Use rdkit default parameters for removing salts
    step_name = 'rdkit SaltRemover'
    try:
        remover = SaltRemover()
        mol = remover.StripMol(mol, dontRemoveEverything=True)

        if mol is None:
            return CleaningResult(failed_step=step_name)

        return CleaningResult(mol=mol)
    except Exception as e:
        logger.error(f'There was an error while calling SaltRemover: {e} \n'
                     f'Skip molecule {Chem.MolToSmiles(mol)}')
        return CleaningResult(failed_step=step_name)


def remove_charge(mol: Mol) -> CleaningResult:
    # Neutralize the molecule
    step_name = 'rdkit Uncharger'
    try:
        u = rdMolStandardize.Uncharger()
        mol = u.uncharge(mol)

        if mol is None:
            return CleaningResult(failed_step=step_name)

        return CleaningResult(mol=mol)
    except Exception as e:
        logger.error(f'There was an error while calling Uncharger: {e} \n'
                     f'Skip molecule {Chem.MolToSmiles(mol)}')
        return CleaningResult(failed_step=step_name)


def canonicalize_tautomers(mol: Mol) -> CleaningResult:
    # Canonicalize tautomers
    step_name = 'rdkit TautomerEnumerator Canonicalize'
    try:
        # It will throw some warnings if some variations can not be kekuklized
        standardizer = rdMolStandardize.TautomerEnumerator()
        mol = standardizer.Canonicalize(mol)

        if mol is None:
            return CleaningResult(failed_step=step_name)

        return CleaningResult(mol=mol)
    except Exception as e:
        logger.error(f'There was an error while calling TautomerEnumerator: {e} \n'
                     f'Skip molecule {Chem.MolToSmiles(mol)}')
        return CleaningResult(failed_step=step_name)


def run_cleaning_steps(mol: Mol):
    cleaning_steps = [use_rdkit_cleanup, remove_salts, remove_charge, canonicalize_tautomers]
    for step in cleaning_steps:
        # Get the smiles being processed to save it later if something fails
        processed_smiles = Chem.MolToSmiles(mol)

        result = step(mol)

        # We still need to check if the rdkit method returns None or if we caught an exception. In these cases we do
        # not need to run the other steps
        if result.mol is None or result.failed_step is not None:
            result.processed_smiles = processed_smiles
            return result

        mol = result.mol
    return CleaningResult(mol=mol)


def _clean_smiles(smiles_list: List[str], output_path: str) -> CleanedSmilesContainer:
    cleaned_smiles = {'original': [], 'clean': []}
    failed_smiles = {'original': [], 'processed_smiles': [], 'reason': []}

    for smiles in tqdm(smiles_list, desc="Cleaning SMILES"):
        logger.info(f'Processing {smiles}')

        mol = Chem.MolFromSmiles(smiles)

        # If the SMILES is not valid, we need to skip it
        if mol is None:
            result = CleaningResult(failed_step='MolFromSmiles', processed_smiles=smiles)
            failed_smiles['original'].append(smiles)
            failed_smiles['processed_smiles'].append(result.processed_smiles)
            failed_smiles['reason'].append(result.failed_step)
            continue

        result = run_cleaning_steps(mol=mol)

        # Other error in one of the cleaning steps
        if result.mol is None or result.failed_step is not None:
            logger.info(f'Processing SMILES {smiles} was not successful in step {result.failed_step}!')
            failed_smiles['original'].append(smiles)
            failed_smiles['processed_smiles'].append(result.processed_smiles)
            failed_smiles['reason'].append(result.failed_step)
            continue

        # If the cleaned SMILES still contains multiple molecules, we need to skip it
        tmp_smiles = Chem.MolToSmiles(result.mol)
        if result.failed_step is None and '.' in tmp_smiles:
            logger.info(f'The cleaned SMILES {tmp_smiles} still contains multiple molecules. Skipping.')
            result = CleaningResult(failed_step='Multiple Molecules', processed_smiles=tmp_smiles)
            failed_smiles['original'].append(smiles)
            failed_smiles['processed_smiles'].append(result.processed_smiles)
            failed_smiles['reason'].append(result.failed_step)
            continue

        # Implicitly canonicalize SMILES
        canonical_smiles = Chem.MolToSmiles(result.mol)

        cleaned_smiles['original'].append(smiles)
        cleaned_smiles['clean'].append(canonical_smiles)

    if len(failed_smiles['original']) > 0:
        logger.info(f'There are {len(failed_smiles['original'])} failed data points!')
        df = pd.DataFrame.from_dict(failed_smiles)
        df.to_csv(os.path.join(output_path, 'failed_cleaning.csv'), index=False)

    return cleaned_smiles


def _remove_duplicates(cleaned_smiles_container: CleanedSmilesContainer, output_path: str) -> CleanedSmilesContainer:
    cleaned_and_unique_smiles_container = {'original': [], 'clean': []}
    duplicate_smiles_container = {'original': [], 'clean': []}
    seen_smiles_set = set()
    duplicate_smiles_set = set()

    for cleaned_smiles in cleaned_smiles_container['clean']:
        if cleaned_smiles in seen_smiles_set:
            duplicate_smiles_set.add(cleaned_smiles)
        else:
            seen_smiles_set.add(cleaned_smiles)

    # Remove all duplicates (Do not keep only one example). Go through the list again and filter
    for original_smiles, cleaned_smiles in zip(cleaned_smiles_container['original'], cleaned_smiles_container['clean']):
        if cleaned_smiles not in duplicate_smiles_set:
            cleaned_and_unique_smiles_container['original'].append(original_smiles)
            cleaned_and_unique_smiles_container['clean'].append(cleaned_smiles)
        else:
            duplicate_smiles_container['original'].append(original_smiles)
            duplicate_smiles_container['clean'].append(cleaned_smiles)

    assert (len(cleaned_smiles_container['clean']) - (len(cleaned_and_unique_smiles_container['clean'])) ==
            len(duplicate_smiles_container['clean']))

    logger.info(f'There exist {len(cleaned_and_unique_smiles_container['clean'])} unique and cleaned '
                f'smiles from a total of {len(cleaned_smiles_container['clean'])} cleaned smiles.')

    if len(duplicate_smiles_container['clean']) > 0:
        df = pd.DataFrame.from_dict(duplicate_smiles_container)
        df.to_csv(os.path.join(output_path, 'duplicates.csv'), index=False)

    return cleaned_and_unique_smiles_container


def _create_cleaned_dataframe(df: DataFrame, cleaned_and_unique_smiles: CleanedSmilesContainer) -> DataFrame:
    smiles_map = dict(zip(cleaned_and_unique_smiles['original'], cleaned_and_unique_smiles['clean']))

    df_filtered = df.loc[df['SMILES'].isin(cleaned_and_unique_smiles['original'])]

    cleaned_df = df_filtered.replace(smiles_map).reset_index(drop=True)

    return cleaned_df


def clean_molecular_data(dataset_path: str, output_path: str) -> None:
    """
    Clean molecular data by standardizing SMILES and removing duplicates.
    
    Args:
        dataset_path: Path to input CSV file with 'SMILES' column
        output_path: Directory to save cleaned data and reports
    """
    os.makedirs(output_path, exist_ok=True)

    logger.info(f'Loading dataset from {dataset_path}')
    original_df = pd.read_csv(dataset_path)
    
    if 'SMILES' not in original_df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column")
    
    smiles_list = original_df['SMILES'].tolist()
    original_count = len(smiles_list)

    logger.info(f'Starting cleaning process for {original_count} molecules')
    cleaned_smiles = _clean_smiles(smiles_list=smiles_list, output_path=output_path)

    logger.info('Removing duplicates')
    cleaned_and_unique_smiles = _remove_duplicates(cleaned_smiles_container=cleaned_smiles, output_path=output_path)

    cleaned_df = _create_cleaned_dataframe(df=original_df, cleaned_and_unique_smiles=cleaned_and_unique_smiles)
    
    # Save cleaned data
    output_file = os.path.join(output_path, 'cleaned_data.csv')
    cleaned_df.to_csv(output_file, index=False)
    
    # Create cleaning summary
    final_count = len(cleaned_df)
    failed_count = original_count - len(cleaned_smiles['clean'])
    duplicate_count = len(cleaned_smiles['clean']) - final_count
    
    summary = {
        'original_molecules': original_count,
        'failed_cleaning': failed_count,
        'duplicates_removed': duplicate_count,
        'final_cleaned_molecules': final_count,
        'cleaning_success_rate': f"{(len(cleaned_smiles['clean'])/original_count)*100:.2f}%",
        'final_retention_rate': f"{(final_count/original_count)*100:.2f}%"
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_path, 'cleaning_summary.csv'), index=False)
    
    logger.info("Cleaning complete!")
    logger.info(f"Results: {original_count} → {final_count} molecules ({summary['final_retention_rate']} retained)")
    logger.info(f"Failed cleaning: {failed_count}")
    logger.info(f"Duplicates removed: {duplicate_count}")
    logger.info(f"Output saved to: {output_file}")


def get_args():
    parser = argparse.ArgumentParser(description='Molecular Data Cleaning')

    parser.add_argument('--input_data', type=str, required=True,
                        help='Path to the input dataset (CSV with SMILES column).')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where cleaned data will be saved.')

    return parser.parse_args()


def main():
    args = get_args()
    clean_molecular_data(dataset_path=args.input_data, output_path=args.output_dir)


if __name__ == '__main__':
    main()