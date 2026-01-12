import pytest
import torch
from torch_geometric.data import Data
from unittest.mock import Mock

from mol_faith.data.data_managers.smiles_data_manager import SMILESTokenizer
from mol_faith.masking.mask_generator import BaseFidelityMasker, SequenceFFidelityMasker, GraphFFidelityMasker, \
    EvaluationStrategy


class TestFidelityParameterCalculation:
    """Test F-Fidelity parameter calculation logic specifically"""

    @pytest.fixture
    def masker(self):
        BaseFidelityMasker.__abstractmethods__ = set()
        base_fidelity_masker = BaseFidelityMasker(beta=0.1, alpha_plus_orig=0.5, alpha_minus_orig=0.5)
        return base_fidelity_masker

    def test_constraint_activation_alpha_plus(self, masker):
        """Test when alpha_plus constraint becomes active"""
        # Case where constraint is NOT active: βtd/s > α+_orig
        s, td = 10, 100  # β*td/s = 0.1*100/10 = 1.0 > 0.5
        alpha_plus, alpha_minus, metadata = masker._calculate_fidelity_parameters(s, td, 0.8, EvaluationStrategy.MASS_THRESHOLD)

        assert alpha_plus == 0.5  # Original value
        assert metadata['constraint_activation']['alpha_plus_limited_by_beta'] is False

        # Case where constraint IS active: βtd/s < α+_orig
        s, td = 50, 100  # β*td/s = 0.1*100/50 = 0.2 < 0.5
        alpha_plus, alpha_minus, metadata = masker._calculate_fidelity_parameters(s, td, 0.8, EvaluationStrategy.MASS_THRESHOLD)

        assert alpha_plus == 0.2  # Constrained value
        assert metadata['constraint_activation']['alpha_plus_limited_by_beta'] is True

    def test_constraint_activation_alpha_minus(self, masker):
        """Test when alpha_minus constraint becomes active"""
        # Case where constraint is NOT active: βtd/(td-s) > α-_orig
        s, td = 10, 100  # β*td/(td-s) = 0.1*100/90 = 0.111 > 0.5? No, so constrained
        alpha_plus, alpha_minus, metadata = masker._calculate_fidelity_parameters(s, td, 0.8, EvaluationStrategy.MASS_THRESHOLD)

        expected_alpha_minus = 0.1 * 100 / 90  # ≈ 0.111
        assert abs(alpha_minus - expected_alpha_minus) < 1e-6
        assert metadata['constraint_activation']['alpha_minus_limited_by_beta'] is True

        # Case where constraint is NOT active: βtd/(td-s) > α-_orig
        s, td = 95, 100  # β*td/(td-s) = 0.1*100/5 = 2.0 > 0.5
        alpha_plus, alpha_minus, metadata = masker._calculate_fidelity_parameters(s, td, 0.8, EvaluationStrategy.MASS_THRESHOLD)

        assert alpha_minus == 0.5  # Original value
        assert metadata['constraint_activation']['alpha_minus_limited_by_beta'] is False

    def test_extreme_cases_rejection(self, masker):
        """Test that extreme cases are properly rejected"""
        # s = 0 case
        alpha_plus, alpha_minus, metadata = masker._calculate_fidelity_parameters(0, 100, 0.8, EvaluationStrategy.MASS_THRESHOLD)
        assert alpha_plus is None
        assert alpha_minus is None
        assert metadata is None

        # s = td case (all features selected)
        alpha_plus, alpha_minus, metadata = masker._calculate_fidelity_parameters(100, 100, 0.8, EvaluationStrategy.MASS_THRESHOLD)
        assert alpha_plus is None
        assert alpha_minus is None
        assert metadata is None

    def test_parameter_edge_case_detection(self, masker):
        # s = td-1 case (almost all features selected)
        alpha_plus, alpha_minus, metadata = masker._calculate_fidelity_parameters(99, 100, 0.8, EvaluationStrategy.MASS_THRESHOLD)
        assert metadata['edge_case_type'] == 'too_dense'
        assert metadata['is_edge_case']

        # s = 1 (almost no features selected)
        alpha_plus, alpha_minus, metadata = masker._calculate_fidelity_parameters(1, 100, 0.8, EvaluationStrategy.MASS_THRESHOLD)
        assert metadata['edge_case_type'] == 'too_sparse'
        assert metadata['is_edge_case']

    def test_beta_parameter_scaling(self):
        """Test how different beta values affect constraints"""
        # Low beta (tight budget)
        masker_low = BaseFidelityMasker(beta=0.05, alpha_plus_orig=0.5, alpha_minus_orig=0.5)

        # High beta (loose budget)
        masker_high = BaseFidelityMasker(beta=0.2, alpha_plus_orig=0.5, alpha_minus_orig=0.5)

        s, td = 30, 100

        alpha_plus_low, alpha_minus_low, metadata_low = masker_low._calculate_fidelity_parameters(s, td,0.8, EvaluationStrategy.MASS_THRESHOLD)
        alpha_plus_high, alpha_minus_high, metadata_high = masker_high._calculate_fidelity_parameters(s, td, 0.8,  EvaluationStrategy.MASS_THRESHOLD)

        # Low beta should have tighter constraints
        assert alpha_plus_low < alpha_plus_high
        assert alpha_minus_low < alpha_minus_high


class TestAttributionMassEdgeCases:
    """Test attribution mass selection with challenging distributions"""

    @pytest.fixture
    def masker(self):
        BaseFidelityMasker.__abstractmethods__ = set()
        base_fidelity_masker = BaseFidelityMasker(beta=0.1, alpha_plus_orig=0.5, alpha_minus_orig=0.5)
        return base_fidelity_masker

    def test_near_zero_attribution(self, masker):
        """Test with very small attribution values"""
        importance_scores = torch.tensor([1e-10, 1e-11, 1e-12, 1e-13])
        mask, s = masker._attribution_mass_selection(importance_scores, 0.8)

        # Should still work and select the highest values
        assert s == 1
        assert mask[0]  # Highest value should be selected
        assert not torch.any(mask[1:])  # All other values are False

    def test_negative_attribution_handling(self, masker):
        """Test handling of negative attribution scores"""
        # Attribution scores should be non-negative, but test robustness
        importance_scores = torch.tensor([0.5, -0.1, 0.3, 0.2])

        with pytest.raises(Exception, match='Attribution mass selection requires non-negative scores') as e_info:
            mask, s = masker._attribution_mass_selection(importance_scores, 0.8)

    def test_attribution_mass_precision(self, masker):
        """Test precision issues with attribution mass calculation"""
        # Create case where cumulative sum might have precision issues
        importance_scores = torch.tensor([0.333333, 0.333333, 0.333334])  # Sums to 1.0
        mask, s = masker._attribution_mass_selection(importance_scores, 0.8)

        # Should select all features to exceed 80%
        assert s == len(importance_scores)

    def test_identical_attribution_values(self, masker):
        """Test with all identical attribution values"""
        importance_scores = torch.tensor([0.50, 0.25, 0.25, 0.25])
        mask, s = masker._attribution_mass_selection(importance_scores, 0.8)

        # Should select 3 features to reach 80% mass
        assert s == 3

    def test_mass_threshold_edge_values(self, masker):
        """Test with extreme mass threshold values"""
        importance_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])

        # Very low threshold
        mask, s = masker._attribution_mass_selection(importance_scores, 0.01)
        assert s == 1  # Should select only the highest

        # Very high threshold
        mask, s = masker._attribution_mass_selection(importance_scores, 0.99)
        assert s == 4  # Should select all features

    def test_detect_edge_case(self, masker):
        """Test edge case detection"""

        # Too sparse
        assert masker._detect_edge_case(0.03) == 'too_sparse'

        # Too dense
        assert masker._detect_edge_case(0.97) == 'too_dense'

        # Normal case
        assert masker._detect_edge_case(0.5) is None

        # Boundary cases
        assert masker._detect_edge_case(0.05) is None
        assert masker._detect_edge_case(0.95) is None

    def test_attribution_mass_selection(self, masker):
        """Test attribution mass selection logic"""

        # Test case 1: Clear hierarchy
        importance_scores = torch.tensor([0.5, 0.3, 0.15, 0.05])  # sums to 1.0
        mask, s = masker._attribution_mass_selection(importance_scores, 0.8)

        expected_mask = torch.tensor([True, True, False, False]).to(mask)
        assert torch.equal(mask, expected_mask)
        assert s == 2

        # Test case 2: One dominant feature
        importance_scores = torch.tensor([0.9, 0.05, 0.03, 0.02])
        mask, s = masker._attribution_mass_selection(importance_scores, 0.8)

        expected_mask = torch.tensor([True, False, False, False]).to(mask)
        assert torch.equal(mask, expected_mask)
        assert s == 1

        # Test case 3: Uniform distribution
        importance_scores = torch.tensor([0.25, 0.25, 0.25, 0.25])
        mask, s = masker._attribution_mass_selection(importance_scores, 0.8)

        # Should select 4 features to reach 80% (need all for uniform)
        assert s == 4
        assert mask.sum().item() == 4


class TestSequenceFFidelityMasker:
    """Test sequence masker functionality"""

    @pytest.fixture
    def tokenizer(self):
        mock_object = Mock(spec=SMILESTokenizer)
        mock_object.pad_token_id = 0
        mock_object.vocab_size = 100
        return mock_object

    @pytest.fixture
    def masker(self, tokenizer):
        mask_generator = SequenceFFidelityMasker(tokenizer, beta=0.1, n_samples=10)
        mask_generator.device = 'cpu'
        return mask_generator

    def test_all_padding_sequence(self, masker):
        """Test sequence with only padding tokens"""
        data = torch.tensor([[0, 0, 0, 0]])  # All padding
        attribution_scores = torch.tensor([0.1, 0.2, 0.0, 0.0])

        # Should throw exception if no actual sequence is in input
        with pytest.raises(Exception, match='Encountered example with no attribution mass'):
            masker.create_evaluation_masks(data, attribution_scores)

    def test_single_real_token(self, masker):
        """Test sequence with only one real token"""
        data = torch.tensor([[5, 0, 0, 0]])  # One real token
        attribution_scores = torch.tensor([1.0, 0.0, 0.0, 0.0])

        result = masker.create_evaluation_masks(data, attribution_scores)
        # Should return None due to s=1, td=1 constraint (s >= td-1)
        assert result is None

    def test_padding_token_attribution_cleaning(self, masker):
        """Test setting attributions to padding token to zero"""
        data = torch.tensor([[1, 2, 3, 0]])  # 3 real tokens
        attribution_scores = torch.tensor([0.0, 0.0, 0.0, 0.5])  # Attribution only on padding

        # After cleaning, all attributions become 0
        cleaned_attr, td, mask = masker._process_attributions_and_get_td(data, attribution_scores)
        assert torch.equal(cleaned_attr, torch.tensor([0.0, 0.0, 0.0, 0.0]))
        assert td == 3

    def test_initialization(self, masker):
        """Test proper initialization"""
        assert masker.beta == 0.1
        assert masker.n_samples == 10
        assert masker.tokenizer.pad_token_id == 0

    def test_validate_sequence_inputs_valid(self, masker):
        """Test validation with valid inputs"""
        data = torch.tensor([[1, 2, 3, 4, 5]])
        attribution_scores = torch.tensor([0.1, 0.2, 0.3, 0.2, 0.2])

        # Should not raise any exception
        masker._validate_sequence_inputs(data, attribution_scores)

    def test_validate_sequence_inputs_invalid_shapes(self, masker):
        """Test validation with invalid shapes"""
        # Wrong data shape
        data = torch.tensor([[1, 2], [3, 4]])
        attribution_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="Expected batch dimension to be 1, got shape:"):
            masker._validate_sequence_inputs(data, attribution_scores)

        # Wrong attribution shape
        data = torch.tensor([[1, 2, 3, 4]])
        attribution_scores = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

        with pytest.raises(ValueError, match="Expected 1D attribution scores"):
            masker._validate_sequence_inputs(data, attribution_scores)

    def test_validate_sequence_inputs_length_mismatch(self, masker):
        """Test validation with length mismatch"""
        data = torch.tensor([[1, 2, 3]])
        attribution_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="Length mismatch"):
            masker._validate_sequence_inputs(data, attribution_scores)

    def test_validate_sequence_inputs_invalid_tokens(self, masker):
        """Test validation with invalid token indices"""
        data = torch.tensor([[1, 2, 150, 4]])  # 150 > vocab_size
        attribution_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])

        with pytest.raises(ValueError, match="Invalid token indices"):
            masker._validate_sequence_inputs(data, attribution_scores)

    def test_validate_batch_inputs_valid(self, masker):
        """Test batch validation with valid inputs"""
        batch_data = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])  # batch_size=2, seq_len=4

        # Should not raise any exception
        masker._validate_batch_inputs(batch_data)

    def test_validate_batch_inputs_invalid_shape(self, masker):
        """Test batch validation with invalid shape"""
        batch_data = torch.tensor([1, 2, 3, 4])  # 1D instead of 2D

        with pytest.raises(ValueError, match="Expected 2D batch"):
            masker._validate_batch_inputs(batch_data)

    def test_process_attributions_and_get_td(self, masker):
        """Test attribution processing and td calculation"""
        # Sequence with padding
        sequence = torch.tensor([1, 2, 3, 0, 0])  # 3 real tokens, 2 padding
        attribution_scores = torch.tensor([0.3, 0.4, 0.3, 0.1, 0.1])

        cleaned_attr, td, mask = masker._process_attributions_and_get_td(sequence, attribution_scores)

        assert td == 3  # Only non-padding tokens
        assert torch.equal(mask, torch.tensor([True, True, True, False, False]))

        expected_cleaned = torch.tensor([0.3, 0.4, 0.3, 0.0, 0.0])
        assert torch.equal(cleaned_attr, expected_cleaned)

    def test_create_finetuning_masks(self, masker):
        """Test fine-tuning mask creation"""
        batch_data = torch.tensor([
            [1, 2, 3, 0, 0],  # 3 real tokens
            [4, 5, 6, 7, 0]  # 4 real tokens
        ])

        masks = masker.create_finetuning_masks(batch_data)

        # Check shape
        assert masks.shape == (2, 5)

        # Check that padding tokens are never masked
        assert masks[0, 3] == 0  # padding position
        assert masks[0, 4] == 0  # padding position
        assert masks[1, 4] == 0  # padding position

        # Check mask values are 0 or 1
        assert torch.all((masks == 0) | (masks == 1))

    def test_finetuning_mask_edge_case(self, masker):
        # Should have at least 1 non-zero mask entry
        data = torch.tensor([
            [1, 2, 3, 0, 0]  # 3 real tokens
        ])

        masks = masker.create_finetuning_masks(data)

        assert masks.shape == (1, 5)  # same as node features
        assert masks.count_nonzero() == 1

    def test_create_evaluation_masks_normal_case(self, masker):
        """Test evaluation mask creation for normal case"""
        data = torch.tensor([[1, 2, 3, 4, 0]])  # 4 real tokens, 1 padding
        attribution_scores = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0])

        result = masker.create_evaluation_masks(data, attribution_scores, threshold_value=0.8)

        assert result is not None
        assert 'fid_plus_masks' in result
        assert 'fid_minus_masks' in result
        assert 'metadata' in result

        # Check mask shapes
        assert result['fid_plus_masks'].shape == (10, 5)  # n_samples=10, seq_len=5
        assert result['fid_minus_masks'].shape == (10, 5)

        # Check metadata
        metadata = result['metadata']
        assert metadata['modality'] == 'sequence'
        assert metadata['has_padding'] is True
        assert metadata['total_sequence_length'] == 5
        assert metadata['actual_molecule_length'] == 4

    def test_create_evaluation_masks_extreme_cases(self, masker):
        """Test evaluation mask creation for extreme cases"""
        data = torch.tensor([[1, 2, 3, 0]])

        # Case 1: s = 0 (no features selected) -> No attribution mass can be calculated
        attribution_scores = torch.tensor([0.0, 0.0, 0.0, 0.0])  # uniform
        with pytest.raises(Exception, match='Encountered example with no attribution mass') as e_info:
            result = masker.create_evaluation_masks(data, attribution_scores, threshold_value=0.8)

        # Case 2: s = td (all features selected)
        attribution_scores = torch.tensor([0.4, 0.3, 0.3, 0.0])
        result = masker.create_evaluation_masks(data, attribution_scores, threshold_value=0.99)
        assert result is None  # Should skip extreme case

    def test_create_evaluation_masks_empty_sequence(self, masker):
        """Test evaluation with sequence containing only padding"""
        data = torch.tensor([[0, 0, 0]])  # all padding
        attribution_scores = torch.tensor([0.0, 0.0, 0.0])

        # Should thrwo error as we do not have a sequence with non-zero attributions
        with pytest.raises(ValueError, match="Encountered example with no attribution mass"):
            result = masker.create_evaluation_masks(data, attribution_scores)

    def test_finetuning_mask_with_minimal_tokens(self, masker):
        """Test fine-tuning mask creation with very few real tokens"""
        batch_data = torch.tensor([
            [1, 0, 0, 0, 0],  # 1 real token
            [2, 3, 0, 0, 0],  # 2 real tokens
        ])

        masks = masker.create_finetuning_masks(batch_data)

        # Check that at least 1 token is masked per sequence (due to max(1, ...))
        for i in range(2):
            real_mask = (batch_data[i] != 0)
            masked_real_tokens = masks[i][real_mask].sum()
            assert masked_real_tokens >= 1

    def test_sequence_mask_shapes_consistency(self, masker):
        """Test that mask shapes are consistent across different sequence lengths"""
        for seq_len in [5, 10, 20]:
            data = torch.unsqueeze(torch.arange(1, seq_len + 1), 0)  # All real tokens
            data[0, -2:] = 0  # Add some padding

            attribution_scores = torch.rand(seq_len)
            attribution_scores[-2:] = 0  # Zero attribution on padding

            result = masker.create_evaluation_masks(data, attribution_scores, threshold_value=0.6)

            if result is not None:  # Skip extreme cases
                assert result['fid_plus_masks'].shape == (10, seq_len)
                assert result['fid_minus_masks'].shape == (10, seq_len)


class TestGraphFFidelityMasker:
    """Test graph masker functionality"""

    @pytest.fixture
    def masker(self):
        mask_generator = GraphFFidelityMasker(beta=0.1, n_samples=10)
        mask_generator.device = 'cpu'
        return mask_generator

    @pytest.fixture
    def sample_graph_data(self):
        """Create sample graph data"""
        x = torch.randn(4, 3)  # 4 nodes, 3 features each
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        batch = torch.zeros(x.shape[0])
        return Data(x=x, edge_index=edge_index, batch=batch)

    def test_initialization(self, masker):
        """Test proper initialization"""
        assert masker.beta == 0.1
        assert masker.n_samples == 10

    def test_validate_graph_inputs_valid(self, masker, sample_graph_data):
        """Test validation with valid inputs"""
        # Node-level attribution
        attribution_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
        masker._validate_graph_inputs(sample_graph_data, attribution_scores)

        # Node-feature level attribution
        attribution_scores = torch.randn(4, 3)
        masker._validate_graph_inputs(sample_graph_data, attribution_scores)

    def test_validate_graph_inputs_missing_features(self, masker):
        """Test validation with missing node features"""
        data = Data(edge_index=torch.tensor([[0, 1], [1, 0]]))
        attribution_scores = torch.tensor([0.5, 0.5])

        with pytest.raises(ValueError, match="missing node features"):
            masker._validate_graph_inputs(data, attribution_scores)

    def test_validate_graph_inputs_invalid_shape(self, masker, sample_graph_data):
        """Test validation with invalid attribution shape"""
        attribution_scores = torch.tensor([0.1, 0.2])  # Wrong size

        with pytest.raises(ValueError, match="not in"):
            masker._validate_graph_inputs(sample_graph_data, attribution_scores)

    def test_validate_graph_inputs_non_finite(self, masker, sample_graph_data):
        """Test validation with non-finite attribution scores"""
        attribution_scores = torch.tensor([0.5, float('nan'), 0.3, 0.2])

        with pytest.raises(ValueError, match="non-finite values"):
            masker._validate_graph_inputs(sample_graph_data, attribution_scores)

    def test_validate_batch_graph_inputs_valid(self, masker):
        """Test batch validation with valid inputs"""
        batch_data = Data(x=torch.randn(6, 3), batch=torch.zeros(6))  # 6 nodes, 3 features
        masker._validate_batch_graph_inputs(batch_data)

    def test_validate_batch_graph_inputs_invalid(self, masker):
        """Test batch validation with invalid inputs"""
        # Missing node features
        batch_data = Data()
        with pytest.raises(ValueError, match="missing node features"):
            masker._validate_batch_graph_inputs(batch_data)

        # Wrong shape
        batch_data = Data(x=torch.randn(5))  # 1D instead of 2D
        with pytest.raises(ValueError, match="Expected 2D node features"):
            masker._validate_batch_graph_inputs(batch_data)

        # No batch data
        batch_data = Data(x=torch.randn((2,4)))  # 1D instead of 2D
        with pytest.raises(ValueError, match='Expect to have a batch input in the finetuning stage!'):
            masks = masker.create_finetuning_masks(batch_data)

    def test_create_finetuning_masks(self, masker, sample_graph_data):
        """Test fine-tuning mask creation"""
        masks = masker.create_finetuning_masks(sample_graph_data)

        # Check shape
        assert masks.shape == (4, 3)  # same as node features

        # Check mask values are 0 or 1
        assert torch.all((masks == 0) | (masks == 1))

    def test_finetuning_mask_edge_case(self, masker):
        """Test fine-tuning mask creation with minimal features"""
        x = torch.randn(2, 2)  # 2 nodes, 2 features each
        edge_index = torch.tensor([[0, 1], [1, 0]])
        batch = torch.zeros(x.shape[0])
        data = Data(x=x, edge_index=edge_index, batch=batch)
        masks = masker.create_finetuning_masks(data)

        assert masks.shape == (2, 2)  # same as node features
        assert masks.count_nonzero() == 1

    def test_create_finetuning_masks_batched(self, masker):
        """Test fine-tuning mask creation with batched graph data"""
        # Create batched graph data: 2 graphs with 3 and 2 nodes respectively
        x = torch.randn(5, 3)  # Total 5 nodes, 3 features each
        batch = torch.tensor([0, 0, 0, 1, 1])
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 0, 4, 3]])

        batch_data = Data(x=x, batch=batch, edge_index=edge_index)

        masks = masker.create_finetuning_masks(batch_data)

        # Check shape
        assert masks.shape == (5, 3)  # same as node features

        # Check mask values are 0 or 1
        assert torch.all((masks == 0) | (masks == 1))

        # Check that each graph has some masked elements
        # Graph 0 (nodes 0, 1, 2): 3 nodes * 3 features = 9 total elements
        # With beta=0.1, should mask at least 1 element: max(int(0.1 * 9), 1) = 1
        graph_0_mask = masks[:3, :]  # First 3 nodes
        assert graph_0_mask.sum() >= 1

        # Graph 1 (nodes 3, 4): 2 nodes * 3 features = 6 total elements
        # With beta=0.1, should mask at least 1 element: max(int(0.1 * 6), 1) = 1
        graph_1_mask = masks[3:, :]  # Last 2 nodes
        assert graph_1_mask.sum() >= 1

    def test_create_finetuning_masks_single_graph_with_batch(self, masker):
        """Test fine-tuning mask creation with single graph that has batch attribute"""
        x = torch.randn(3, 2)  # 3 nodes, 2 features each
        batch = torch.tensor([0, 0, 0])  # All nodes belong to same graph
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])

        batch_data = Data(x=x, batch=batch, edge_index=edge_index)

        masks = masker.create_finetuning_masks(batch_data)

        # Check shape
        assert masks.shape == (3, 2)  # same as node features

        # Check mask values are 0 or 1
        assert torch.all((masks == 0) | (masks == 1))

        # Should mask at least 1 element: max(int(0.1 * 6), 1) = 1
        assert masks.sum() >= 1

    def test_create_finetuning_masks_no_batch_attribute(self, masker):
        """Test fine-tuning mask creation without batch attribute"""
        x = torch.randn(4, 3)  # 4 nodes, 3 features each
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

        # No batch attribute - treated as single graph
        data = Data(x=x, edge_index=edge_index)

        with pytest.raises(ValueError, match='Expect to have a batch input in the finetuning stage!'):
            masks = masker.create_finetuning_masks(data)

    def test_create_evaluation_masks_node_level(self, masker, sample_graph_data):
        """Test evaluation mask creation with node-level attributions"""
        attribution_scores = torch.tensor([0.5, 0.3, 0.15, 0.05])  # node-level

        result = masker.create_evaluation_masks(sample_graph_data, attribution_scores, threshold_value=0.8)

        assert result is not None
        assert 'fid_plus_masks' in result
        assert 'fid_minus_masks' in result
        assert 'metadata' in result

        # Check mask shapes
        assert result['fid_plus_masks'].shape == (10, 4, 3)  # n_samples, n_nodes, n_features
        assert result['fid_minus_masks'].shape == (10, 4, 3)

        # Check metadata
        metadata = result['metadata']
        assert metadata['modality'] == 'graph'
        assert metadata['has_padding'] is False
        assert metadata['n_nodes'] == 4
        assert metadata['n_features'] == 3
        assert metadata['granularity_type'] == 'node_level'
        assert metadata['importance_conservation_applied'] is True

    def test_create_evaluation_masks_node_feature_level(self, masker, sample_graph_data):
        """Test evaluation mask creation with node-feature level attributions"""
        attribution_scores = torch.abs(torch.randn(4, 3))  # node-feature level

        result = masker.create_evaluation_masks(sample_graph_data, attribution_scores, threshold_value=0.8)

        assert result is not None

        # Check metadata
        metadata = result['metadata']
        assert metadata['granularity_type'] == 'node_feature_level'
        assert metadata['importance_conservation_applied'] is False

    def test_create_evaluation_masks_extreme_cases(self, masker, sample_graph_data):
        """Test evaluation mask creation for extreme cases"""
        # Case 1: s = 0 (uniform distribution with high threshold) -> Nn -zero attribution scores
        attribution_scores = torch.zeros(4)
        with pytest.raises(Exception, match='Encountered example with no attribution mass') as e_info:
            result = masker.create_evaluation_masks(sample_graph_data, attribution_scores, threshold_value=0.8)

        # Case 2: s = td (concentrated distribution with low threshold)
        attribution_scores = torch.tensor([0.25, 0.25, 0.25, 0.25])
        result = masker.create_evaluation_masks(sample_graph_data, attribution_scores, threshold_value=0.99)
        assert result is None  # Should skip extreme case

    def test_importance_conservation(self, masker, sample_graph_data):
        """Test that importance is conserved during granularity conversion"""
        original_node_importance = torch.tensor([0.5, 0.3, 0.15, 0.05])

        # Simulate the granularity conversion manually
        n_features = 3
        converted = original_node_importance.unsqueeze(-1) / n_features
        converted = converted.repeat(1, n_features).view(-1)

        # Check conservation
        assert torch.allclose(original_node_importance.sum(), converted.sum(), atol=1e-6)

    def test_metadata_content(self, masker, sample_graph_data):
        """Test comprehensive metadata content"""
        attribution_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])

        result = masker.create_evaluation_masks(sample_graph_data, attribution_scores, threshold_value=0.8)
        metadata = result['metadata']

        # Check all expected metadata fields
        required_fields = [
            'alpha_plus', 'alpha_minus', 's_value', 'td_value', 's_ratio', 'alpha_plus_origin', 'alpha_minus_origin',
            'edge_case_type', 'is_edge_case', 'constraint_activation',
            'evaluation_feasibility', 'selection_strategy', 'threshold_value_used', 'total_sequence_length',
            'modality', 'n_nodes', 'n_features', 'granularity_type', 'has_padding', 'total_node_features',
            'importance_conservation_applied'
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

        # Check constraint activation structure
        assert 'alpha_plus_limited_by_beta' in metadata['constraint_activation']
        assert 'alpha_minus_limited_by_beta' in metadata['constraint_activation']

        # Check evaluation feasibility structure
        assert 'can_evaluate_fid_plus' in metadata['evaluation_feasibility']
        assert 'can_evaluate_fid_minus' in metadata['evaluation_feasibility']

    def test_single_node_graph(self, masker):
        """Test with single node graph"""
        data = Data(x=torch.randn(1, 3))  # 1 node, 3 features
        attribution_scores = torch.tensor([1.0])  # Node-level

        result = masker.create_evaluation_masks(data, attribution_scores)
        # Should return None due to s=td constraint (all features selected)
        assert result is None

    def test_single_feature_per_node(self, masker):
        """Test with single feature per node"""
        data = Data(x=torch.randn(4, 1))  # 4 nodes, 1 feature each
        attribution_scores = torch.tensor([0.5, 0.3, 0.15, 0.05])  # Node-level

        result = masker.create_evaluation_masks(data, attribution_scores, threshold_value=0.8)

        if result is not None:
            assert result['fid_plus_masks'].shape == (10, 4, 1)
            assert result['fid_minus_masks'].shape == (10, 4, 1)

    def test_importance_conservation_numerical(self, masker):
        """Test numerical accuracy of importance conservation"""
        data = Data(x=torch.randn(3, 5))  # 3 nodes, 5 features

        # Test with various node-level attributions
        test_cases = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.5, 0.3, 0.2]),
            torch.tensor([0.33333, 0.33334, 0.33333]),
        ]

        for node_attribution in test_cases:
            # Manually simulate the importance conservation
            n_features = 5
            feature_attribution = node_attribution.unsqueeze(-1) / n_features
            feature_attribution = feature_attribution.repeat(1, n_features).view(-1)

            # Check conservation to high precision
            original_sum = node_attribution.sum()
            conserved_sum = feature_attribution.sum()
            assert torch.allclose(original_sum, conserved_sum, atol=1e-10)

    def test_node_feature_level_no_conversion(self, masker):
        """Test that node-feature level attributions don't get converted"""
        data = Data(x=torch.randn(3, 4))
        attribution_scores = torch.abs(torch.randn(3, 4))  # Already node-feature level

        result = masker.create_evaluation_masks(data, attribution_scores, threshold_value=0.7)

        if result is not None:
            metadata = result['metadata']
            assert metadata['granularity_type'] == 'node_feature_level'
            assert metadata['importance_conservation_applied'] is False


class TestIntegration:
    """Integration tests across both maskers"""

    @pytest.fixture
    def tokenizer(self):
        mock_object = Mock(spec=SMILESTokenizer)
        mock_object.pad_token_id = 0
        mock_object.vocab_size = 100
        return mock_object

    @pytest.fixture
    def seq_masker(self, tokenizer):
        mask_generator = SequenceFFidelityMasker(tokenizer=tokenizer, n_samples=5, beta=0.1)
        mask_generator.device = 'cpu'
        return mask_generator

    @pytest.fixture
    def graph_masker(self):
        mask_generator = GraphFFidelityMasker(beta=0.1, n_samples=10)
        mask_generator.device = 'cpu'
        return mask_generator

    def test_consistent_metadata_structure(self, tokenizer, seq_masker, graph_masker):
        """Test that both maskers produce consistent metadata structure"""

        # Sequence data
        seq_data = torch.tensor([[1, 2, 3, 4, 0]])
        seq_attr = torch.tensor([0.4, 0.3, 0.2, 0.1, 0])
        seq_result = seq_masker.create_evaluation_masks(seq_data, seq_attr)

        # Graph data
        graph_data = Data(x=torch.randn(3, 2))
        graph_attr = torch.tensor([0.5, 0.3, 0.2])
        graph_result = graph_masker.create_evaluation_masks(graph_data, graph_attr)

        # Check that both have same core metadata fields
        core_fields = ['alpha_plus', 'alpha_minus', 's_value', 'td_value', 's_ratio',
                       'constraint_activation', 'evaluation_feasibility', 'modality']

        for field in core_fields:
            assert field in seq_result['metadata']
            assert field in graph_result['metadata']

    def test_mask_convention_consistency(self, tokenizer, seq_masker, graph_masker):
        """Test that mask convention (1=mask, 0=keep) is consistent across multiple runs"""

        # Run the test 10 times with different random seeds to ensure consistency
        for run_idx in range(10):
            # Set different seed for each run to test stochastic behavior
            torch.manual_seed(42 + run_idx)

            # Test sequence masks
            seq_data = torch.tensor([[1, 2, 3, 4]])
            seq_attr = torch.tensor([0.5, 0.3, 0.15, 0.05])
            seq_result = seq_masker.create_evaluation_masks(seq_data, seq_attr)

            assert seq_result is not None, f"Sequence result was None on run {run_idx}"

            fid_plus_masks = seq_result['fid_plus_masks']
            fid_minus_masks = seq_result['fid_minus_masks']

            # Check that mask values are either 0 or 1
            assert torch.all((fid_plus_masks == 0) | (fid_plus_masks == 1)), \
                f"Sequence Fid+ masks contain invalid values on run {run_idx}"
            assert torch.all((fid_minus_masks == 0) | (fid_minus_masks == 1)), \
                f"Sequence Fid- masks contain invalid values on run {run_idx}"

            # Check that mask values are not only 1 (some variability expected)
            assert not torch.all((fid_plus_masks == 1)), \
                f"Sequence Fid+ masks are all 1s on run {run_idx}"
            assert not torch.all((fid_minus_masks == 1)), \
                f"Sequence Fid- masks are all 1s on run {run_idx}"

            # Test graph masks with fresh random data each run
            graph_data = Data(x=torch.randn(2, 2))
            graph_attr = torch.tensor([0.6, 0.4])
            graph_result = graph_masker.create_evaluation_masks(graph_data, graph_attr)

            assert graph_result is not None, f"Graph result was None on run {run_idx}"

            fid_plus_masks = graph_result['fid_plus_masks']
            fid_minus_masks = graph_result['fid_minus_masks']

            # Check that mask values are either 0 or 1
            assert torch.all((fid_plus_masks == 0) | (fid_plus_masks == 1)), \
                f"Graph Fid+ masks contain invalid values on run {run_idx}"
            assert torch.all((fid_minus_masks == 0) | (fid_minus_masks == 1)), \
                f"Graph Fid- masks contain invalid values on run {run_idx}"

            # Check that mask values are not only 1 (some variability expected)
            assert not torch.all((fid_plus_masks == 1)), \
                f"Graph Fid+ masks are all 1s on run {run_idx}"
            assert not torch.all((fid_minus_masks == 1)), \
                f"Graph Fid- masks are all 1s on run {run_idx}"

            print(f"✓ Run {run_idx + 1}/10 passed")

        print("✓ All 10 runs completed successfully")


class TestStochasticBehavior:
    """Test stochastic sampling behavior and reproducibility"""

    @pytest.fixture
    def tokenizer(self):
        mock_object = Mock(spec=SMILESTokenizer)
        mock_object.pad_token_id = 0
        mock_object.vocab_size = 100
        return mock_object

    def test_stochastic_mask_variability(self, tokenizer):
        """Test that stochastic masks show appropriate variability"""
        masker = SequenceFFidelityMasker(tokenizer, n_samples=50, beta=0.1)
        masker.device = 'cpu'

        data = torch.tensor([[1, 2, 3, 4, 5]])
        attribution_scores = torch.tensor([0.4, 0.3, 0.2, 0.05, 0.05])

        result = masker.create_evaluation_masks(data, attribution_scores, threshold_value=0.8)

        if result is not None:
            fid_plus_masks = result['fid_plus_masks']

            # Check that not all samples are identical
            unique_masks = []
            for i in range(fid_plus_masks.shape[0]):
                mask_tuple = tuple(fid_plus_masks[i].tolist())
                if mask_tuple not in unique_masks:
                    unique_masks.append(mask_tuple)

            # Should have multiple unique masks due to stochastic sampling
            assert len(unique_masks) > 1, "All masks are identical - no stochastic variation"

    def test_seed_reproducibility(self, tokenizer):
        """Test that results are reproducible with same seed"""
        masker = SequenceFFidelityMasker(tokenizer, n_samples=10, beta=0.1)
        masker.device = 'cpu'

        data = torch.tensor([[1, 2, 3, 4]])
        attribution_scores = torch.tensor([0.4, 0.3, 0.2, 0.1])

        # Run 1
        torch.manual_seed(42)
        result1 = masker.create_evaluation_masks(data, attribution_scores)

        # Run 2 with same seed
        torch.manual_seed(42)
        result2 = masker.create_evaluation_masks(data, attribution_scores)

        if result1 is not None and result2 is not None:
            assert torch.equal(result1['fid_plus_masks'], result2['fid_plus_masks'])
            assert torch.equal(result1['fid_minus_masks'], result2['fid_minus_masks'])

    def test_different_seeds_different_results(self, tokenizer):
        """Test that different seeds produce different results"""
        masker = SequenceFFidelityMasker(tokenizer, n_samples=10, beta=0.1)
        masker.device = 'cpu'

        data = torch.tensor([[1, 2, 3, 4, 5]])
        attribution_scores = torch.tensor([0.4, 0.3, 0.2, 0.05, 0.05])

        # Run with seed 42
        torch.manual_seed(42)
        result1 = masker.create_evaluation_masks(data, attribution_scores)

        # Run with seed 123
        torch.manual_seed(123)
        result2 = masker.create_evaluation_masks(data, attribution_scores)

        # Results should be different (with high probability)
        assert not torch.equal(result1['fid_plus_masks'], result2['fid_plus_masks'])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
