import logging
import os
import pandas as pd
import re
import torch
from pandas import DataFrame
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple

from mol_faith.data.data_managers.data_manager import DataManager
from mol_faith.data.types import DataCategoryType

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class SMILESTokenizationError(Exception):
    """Raised when SMILES tokenization fails."""
    pass


class InvalidSMILESError(Exception):
    """Raised when SMILES string is invalid or cannot be parsed."""
    pass


class VocabularyError(Exception):
    """Raised when there are issues with vocabulary building or usage."""
    pass


class MaskingError(Exception):
    """Raised when token masking operations fail."""
    pass


class SMILESTokenizer:
    """
    Tokenizer for SMILES strings.
    """

    # Elements found in the dataset - ordered by length (longest first) to avoid partial matches
    _ELEMENTS_STR = r"Br|Cl|As|Se|Si|se|B|C|F|I|N|O|P|S|H|b|c|n|o|s|p"

    # Single regex pattern that captures all SMILES features (excluding ring closures >99)
    _REGEX_PATTERN = rf"(\[|\]|{_ELEMENTS_STR}|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@@?|\?|\%[0-9]{{2}}|[0-9])"

    # Compile regex pattern
    _RE_PATTERN = re.compile(_REGEX_PATTERN)

    # Essential special tokens
    SPECIAL_TOKENS = {
        'PAD': '[PAD]',  # Padding token (ID: 0)
        'UNK': '[UNK]',  # Unknown token (ID: 1) - safety net
        'MASK': '[MASK]',  # Mask token (ID: 2) - for XAI
    }

    def __init__(self):
        """
        Initialize the chemical subword tokenizer.
        Always segments square brackets as separate tokens.
        """
        self.token_to_id = {}
        self.id_to_token = {}
        self._initialize_special_tokens()

    def _initialize_special_tokens(self):
        """Initialize vocabulary with special tokens at fixed positions."""
        for i, (name, token) in enumerate(self.SPECIAL_TOKENS.items()):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (including all special tokens)."""
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id[self.SPECIAL_TOKENS['PAD']]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id[self.SPECIAL_TOKENS['UNK']]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id[self.SPECIAL_TOKENS['MASK']]

    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenize SMILES string using chemical-aware regex patterns.

        Args:
            smiles: Clean, validated SMILES string
        Returns:
            List of chemical subword tokens
        Raises:
            SMILESTokenizationError: If tokenization fails
        """
        if not isinstance(smiles, str):
            raise InvalidSMILESError(f'SMILES must be a string, got {type(smiles)}')

        if not smiles.strip():
            raise InvalidSMILESError('SMILES string cannot be empty or whitespace-only')

        try:
            tokens = self._RE_PATTERN.findall(smiles.strip())

            if not tokens:
                raise SMILESTokenizationError(f'No valid tokens could be extracted from SMILES: {smiles}')

            return tokens

        except Exception as e:
            raise SMILESTokenizationError(f'Unexpected error tokenizing SMILES {smiles}: {e}') from e

    def build_vocab_from_corpus(self, smiles_list: List[str]) -> None:
        """
        Build vocabulary from a corpus of SMILES strings.

        Args:
            smiles_list: List of preprocessed SMILES strings
        Raises:
            VocabularyError: If vocabulary building fails
        """
        if not smiles_list:
            raise VocabularyError('Cannot build vocabulary from empty corpus')

        logger.info(f'Building vocabulary from {len(smiles_list)} SMILES strings')

        # Collect all unique tokens
        all_tokens = set()

        for smiles in smiles_list:
            tokens = self.tokenize(smiles)
            all_tokens.update(tokens)

        # Build vocabulary (special tokens already added)
        current_id = len(self.SPECIAL_TOKENS)

        # Sort tokens for reproducibility
        sorted_tokens = sorted(all_tokens)

        for token in sorted_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = current_id
                self.id_to_token[current_id] = token
                current_id += 1

        logger.info(f'Built vocabulary with {self.vocab_size} tokens')

    def encode(self,
               smiles: str,
               max_seq_len: int,
               padding: bool = True) -> torch.Tensor:
        """
        Encode SMILES string to token IDs

        Args:
            smiles: Input SMILES string
            max_seq_len: Maximum sequence length
            padding: Whether to pad sequence
        Returns:
            Tensor of token IDs
        """
        # Tokenize using chemical-aware segmentation
        tokens = self.tokenize(smiles)

        token_ids = []
        unknown_tokens = []

        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                token_ids.append(self.unk_token_id)
                unknown_tokens.append(token)

        if unknown_tokens:
            raise SMILESTokenizationError(f'Unknown tokens in {smiles}: {unknown_tokens}! \n'
                                          f'This may indicate vocabulary was not built on complete dataset')

        if len(token_ids) > max_seq_len:
            raise SMILESTokenizationError('Encounterd a tokenized sequence longer then the specifiec max sequence'
                                          f'length of {max_seq_len} with a length of {len(token_ids)}!')

        # Pad if too short and padding is enabled
        if len(token_ids) < max_seq_len and padding:
            token_ids.extend([self.pad_token_id] * (max_seq_len - len(token_ids)))

        return torch.tensor(token_ids, dtype=torch.long)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to SMILES string.

        Args:
            token_ids: Tensor of token IDs
            skip_special_tokens: Whether to skip special tokens in output
        Returns:
            Decoded SMILES string
        """
        tokens = []

        for token_id in token_ids.tolist():
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]

                # Skip special tokens if requested
                if skip_special_tokens and token in self.SPECIAL_TOKENS.values():
                    continue

                tokens.append(token)
            else:
                logger.warning(f'Unknown token ID encountered: {token_id}')

        return ''.join(tokens)

    @staticmethod
    def get_token_count(smiles: str, tokenizer: 'SMILESTokenizer') -> int:
        """
        Get the number of tokens in a SMILES string using the tokenizer.

        Args:
            smiles: SMILES string
            tokenizer: Tokenizer instance to use for tokenization
        Returns:
            Number of tokens in the tokenized SMILES
        Raises:
            InvalidSMILESError: If tokenization fails
        """
        try:
            tokens = tokenizer.tokenize(smiles)
            return len(tokens)
        except Exception as e:
            raise InvalidSMILESError(f'Could not tokenize SMILES: {smiles}. Error: {e}')

    def save_vocab(self, vocab_path: str):
        """Save vocabulary to JSON file."""
        import json

        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.SPECIAL_TOKENS
        }

        with open(vocab_path, 'w') as f:
            json.dump(vocab_data, f, indent=2)

        logger.info(f'Saved chemical subword vocabulary to {vocab_path}')

    @classmethod
    def load_vocab(cls, vocab_path: str) -> 'SMILESTokenizer':
        """Load vocabulary from JSON file."""
        import json

        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        tokenizer = cls()
        tokenizer.token_to_id = vocab_data['token_to_id']
        tokenizer.id_to_token = {int(v): k for k, v in vocab_data['token_to_id'].items()}

        return tokenizer


class SMILESSequenceDataset(Dataset):
    def __init__(self,
                 dataframe: DataFrame,
                 tokenizer: SMILESTokenizer,
                 max_seq_len: int,
                 target_column_name: str = 'Target',
                 smiles_column_name: str = 'SMILES'):
        super().__init__()
        self._validate_parameters(dataframe, target_column_name, smiles_column_name)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.smiles_data = dataframe[smiles_column_name].values
        self.targets = dataframe[target_column_name].values

    @staticmethod
    def _validate_parameters(dataframe: DataFrame, target_column: str, smiles_column: str):
        """Validate input parameters."""
        if dataframe.empty:
            raise ValueError('DataFrame cannot be empty')

        if smiles_column not in dataframe.columns:
            raise ValueError(f'SMILES column {smiles_column} not found in DataFrame')

        if target_column not in dataframe.columns:
            raise ValueError(f'Target column {target_column} not found in DataFrame')

        if dataframe[smiles_column].isna().any():
            raise ValueError('Found NaN values in SMILES column')

        if dataframe[target_column].isna().any():
            raise ValueError('Found NaN values in target column')

    def __len__(self) -> int:
        return len(self.smiles_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index
        Returns:
            Tuple of (token_ids, target) where token_ids uses tokenizer's max_length
        """
        if idx >= len(self):
            raise IndexError(f'Index {idx} out of range for dataset of size {len(self)}')

        smiles = self.smiles_data[idx]
        target = self.targets[idx]

        # Encode SMILES to token indices
        token_ids = self.tokenizer.encode(smiles=smiles, max_seq_len=self.max_seq_len)

        return token_ids, torch.tensor(target, dtype=torch.float32)

    def get_sequence_lengths(self) -> List[int]:
        """Get actual sequence lengths (before padding) for all samples."""
        lengths = []
        for smiles in self.smiles_data:
            tokens = self.tokenizer.tokenize(smiles)
            lengths.append(len(tokens))
        return lengths


class SMILESSequenceDataManager(DataManager):
    def __init__(self, data_path: str, batch_size: int, max_seq_len: int, tokenizer: SMILESTokenizer):
        super().__init__(data_path)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        assert len(self.tokenizer.token_to_id) > 0, 'Tokenizer was not loaded with vocabulary!'

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size for model initialization."""
        return self.tokenizer.vocab_size

    def load_data(self, target_column_name: str, data_category_type: str) -> DataLoader:
        """
        Load data for a specific split using pre-built tokenizer.

        Args:
            target_column_name: Name of target column
            data_category_type: Type of data split ('train', 'val', 'test')
        Returns:
            DataLoader for the specified split
        """
        if not target_column_name:
            raise ValueError('Target column name cannot be empty')

        file_path = os.path.join(self.data_path, f'{data_category_type}.csv')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'Data file not found: {file_path}')

        data = pd.read_csv(file_path)

        # Create dataset with pre-built tokenizer
        dataset = SMILESSequenceDataset(dataframe=data,
                                        tokenizer=self.tokenizer,
                                        max_seq_len=self.max_seq_len,
                                        target_column_name=target_column_name)

        # Create data loader
        shuffle = (data_category_type == DataCategoryType.TRAIN.value)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          shuffle=shuffle,
                          num_workers=0,
                          pin_memory=True)
