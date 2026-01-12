import logging
import os

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer, SMILESSequenceDataManager
from mol_faith.masking.mask_generator import SequenceFFidelityMasker
from mol_faith.model.CNN.cnn_model import BasicCNNRegressor
from mol_faith.model.transformer.transformer import BasicTransformerRegressor
from mol_faith.training.dnn_training.trainer import DNNTrainer

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class SequenceModelTrainer(DNNTrainer):
    def __init__(self, config_file_path: str):
        super().__init__(config_file_path)

    def _setup_components(self):
        self.tokenizer = self._get_tokenizer()
        self.mask_generator = self._get_mask_generator()
        self.data_manager = self._get_data_manager()
        self.model = self._get_model()
        self.checkpoint_handler = self._get_checkpoint_handler()

    def _get_tokenizer(self) -> SMILESTokenizer:
        if not os.path.exists(self.config.model.vocab_path):
            raise FileNotFoundError(f'Vocabulary file not found in path: {self.config.model.vocab_path}\n'
                                    f'Please run preprocessing to create vocabulary first.')

        tokenizer = SMILESTokenizer.load_vocab(self.config.model.vocab_path)
        logger.info(f'Loaded SMILES tokenizer with {tokenizer.vocab_size} vocabulary size '
                    f'from {self.config.model.vocab_path}')
        return tokenizer

    def _get_mask_generator(self):
        if self.config.training.finetune_beta:
            return SequenceFFidelityMasker(beta=self.config.training.finetune_beta, tokenizer=self.tokenizer)
        else:
            return None

    def _get_data_manager(self):
        return SMILESSequenceDataManager(data_path=self.config.training.dataset_folder_path,
                                         batch_size=self.config.training.batch_size,
                                         max_seq_len=self.config.model.max_seq_len,
                                         tokenizer=self.tokenizer)

    def _get_model(self):
        model_name = self.config.training.model_name.lower()

        if model_name == "cnn":
            logger.info("Creating BasicCNNRegressor model")
            return BasicCNNRegressor(config=self.config.model, tokenizer=self.tokenizer)
        elif model_name == "transformer":
            logger.info("Creating BasicTransformerRegressor model")
            return BasicTransformerRegressor(config=self.config.model, tokenizer=self.tokenizer)
        else:
            raise ValueError(f"Unknown model name: {model_name}. Supported models: ['cnn', 'transformer']")

    def _process_batch(self, batch):
        inputs, targets = batch
        return inputs.to(self.device), targets.to(self.device)
