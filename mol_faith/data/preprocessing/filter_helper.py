import numpy as np
from typing import Dict, List


def analyze_heavy_atom_distribution(smiles_list: List[str],
                                    plot: bool = True) -> Dict:
    """
    Comprehensive analysis of heavy atom count distribution for molecular filtering.

    Args:
        smiles_list: List of SMILES strings from your dataset
        plot: Whether to create distribution plots
    Returns:
        Dictionary with detailed statistics and recommendations
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
    except ImportError:
        raise ImportError('RDKit is required for heavy atom analysis. Install with: pip install rdkit')

    heavy_atom_counts = []
    failed_count = 0

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                heavy_atoms = mol.GetNumHeavyAtoms()
                heavy_atom_counts.append(heavy_atoms)
            else:
                failed_count += 1
        except:
            failed_count += 1
            continue

    if not heavy_atom_counts:
        raise ValueError('No valid SMILES found for heavy atom analysis')

    heavy_atom_counts = np.array(heavy_atom_counts)

    # Comprehensive statistics
    percentiles = [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9]
    percentile_values = {p: int(np.percentile(heavy_atom_counts, p)) for p in percentiles}

    stats = {
        'total_molecules': len(smiles_list),
        'valid_molecules': len(heavy_atom_counts),
        'failed_parsing': failed_count,
        'mean': float(np.mean(heavy_atom_counts)),
        'median': float(np.median(heavy_atom_counts)),
        'std': float(np.std(heavy_atom_counts)),
        'min': int(np.min(heavy_atom_counts)),
        'max': int(np.max(heavy_atom_counts)),
        'percentiles': percentile_values
    }

    # Coverage analysis for different cutoffs
    coverage_analysis = {}
    for p in percentiles:
        cutoff = percentile_values[p]
        coverage = np.sum(heavy_atom_counts <= cutoff) / len(heavy_atom_counts)
        excluded = np.sum(heavy_atom_counts > cutoff)
        avg_excess = np.mean(heavy_atom_counts[heavy_atom_counts > cutoff] - cutoff) if excluded > 0 else 0

        coverage_analysis[f'{p}th_percentile'] = {
            'cutoff': cutoff,
            'coverage': float(coverage),
            'molecules_excluded': int(excluded),
            'avg_excess_atoms': float(avg_excess)
        }

    stats['coverage_analysis'] = coverage_analysis

    # Size distribution analysis
    size_categories = _categorize_by_size(heavy_atom_counts)
    stats['size_categories'] = size_categories

    # Plotting
    if plot:
        _plot_heavy_atom_analysis(heavy_atom_counts, stats)

    return stats


def _categorize_by_size(heavy_atom_counts: np.ndarray) -> Dict:
    """Categorize molecules by size based on heavy atom counts."""
    total = len(heavy_atom_counts)

    categories = {
        'very_small': {'range': '1-10', 'count': 0, 'percentage': 0.0},
        'small': {'range': '11-20', 'count': 0, 'percentage': 0.0},
        'medium': {'range': '21-35', 'count': 0, 'percentage': 0.0},
        'large': {'range': '36-50', 'count': 0, 'percentage': 0.0},
        'very_large': {'range': '51-75', 'count': 0, 'percentage': 0.0},
        'extra_large': {'range': '76+', 'count': 0, 'percentage': 0.0}
    }

    # Count molecules in each category
    categories['very_small']['count'] = np.sum((heavy_atom_counts >= 1) & (heavy_atom_counts <= 10))
    categories['small']['count'] = np.sum((heavy_atom_counts >= 11) & (heavy_atom_counts <= 20))
    categories['medium']['count'] = np.sum((heavy_atom_counts >= 21) & (heavy_atom_counts <= 35))
    categories['large']['count'] = np.sum((heavy_atom_counts >= 36) & (heavy_atom_counts <= 50))
    categories['very_large']['count'] = np.sum((heavy_atom_counts >= 51) & (heavy_atom_counts <= 75))
    categories['extra_large']['count'] = np.sum(heavy_atom_counts >= 76)

    # Calculate percentages
    for category in categories.values():
        category['percentage'] = float(category['count'] / total * 100)

    return categories


def _plot_heavy_atom_analysis(heavy_atom_counts: np.ndarray, stats: Dict):
    """Create comprehensive plots for heavy atom distribution analysis."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Heavy Atom Count Distribution Analysis', fontsize=16)

        # 1. Histogram
        axes[0, 0].hist(heavy_atom_counts, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats['mean']:.1f}')
        axes[0, 0].axvline(stats['median'], color='blue', linestyle='--', label=f'Median: {stats['median']:.1f}')

        # Add drug-like cutoff
        axes[0, 0].axvline(30, color='green', linestyle=':', alpha=0.7, label='Drug-like cutoff (30)')

        axes[0, 0].set_xlabel('Heavy Atom Count')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Heavy Atom Counts')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cumulative distribution
        sorted_counts = np.sort(heavy_atom_counts)
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        axes[0, 1].plot(sorted_counts, cumulative)

        # Mark important percentiles
        for p in [95, 99, 99.5]:
            val = np.percentile(heavy_atom_counts, p)
            axes[0, 1].axvline(val, color='red', alpha=0.7, linestyle='--')
            axes[0, 1].text(val, p / 100, f'{p}%', rotation=90, verticalalignment='bottom')

        # Add drug-like cutoff
        axes[0, 1].axvline(30, color='green', linestyle=':', alpha=0.7, label='Drug-like')

        axes[0, 1].set_xlabel('Heavy Atom Count')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Size category distribution
        categories = stats['size_categories']
        cat_names = list(categories.keys())
        cat_counts = [categories[cat]['count'] for cat in cat_names]
        cat_labels = [f'{cat.replace('_', ' ').title()}\n({categories[cat]['range']})' for cat in cat_names]

        colors = plt.cm.viridis(np.linspace(0, 1, len(cat_names)))
        bars = axes[1, 0].bar(range(len(cat_names)), cat_counts, color=colors, alpha=0.7)
        axes[1, 0].set_xticks(range(len(cat_names)))
        axes[1, 0].set_xticklabels(cat_labels, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Number of Molecules')
        axes[1, 0].set_title('Molecular Size Distribution')
        axes[1, 0].grid(True, alpha=0.3)

        # Add percentage labels on bars
        for bar, cat in zip(bars, cat_names):
            height = bar.get_height()
            percentage = categories[cat]['percentage']
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 4. Data loss by cutoff
        percentiles = [90, 95, 97, 98, 99, 99.5, 99.9]
        excluded = [stats['coverage_analysis'][f'{p}th_percentile']['molecules_excluded'] for p in percentiles]
        cutoffs = [stats['coverage_analysis'][f'{p}th_percentile']['cutoff'] for p in percentiles]

        bars = axes[1, 1].bar(range(len(percentiles)), excluded, alpha=0.7)
        axes[1, 1].set_xticks(range(len(percentiles)))
        axes[1, 1].set_xticklabels([f'{p}%\n({cutoffs[i]} atoms)' for i, p in enumerate(percentiles)],
                                   rotation=45, ha='right')
        axes[1, 1].set_ylabel('Molecules Excluded')
        axes[1, 1].set_title('Data Loss by Heavy Atom Cutoff')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        _print_heavy_atom_summary(stats)

    except ImportError:
        print('Matplotlib not available. Skipping plots.')
        _print_heavy_atom_summary(stats)


def _print_heavy_atom_summary(stats: Dict):
    """Print text-based summary for heavy atom analysis."""
    print('\n' + '=' * 80)
    print('HEAVY ATOM COUNT ANALYSIS SUMMARY')
    print('=' * 80)
    print(f'Dataset: {stats['total_molecules']} molecules')
    print(
        f'Valid parsing: {stats['valid_molecules']} ({stats['valid_molecules'] / stats['total_molecules'] * 100:.1f}%)')
    print(f'Mean heavy atoms: {stats['mean']:.1f} ± {stats['std']:.1f}')
    print(f'Range: {stats['min']} - {stats['max']} heavy atoms')

    print(f'\nSIZE CATEGORY DISTRIBUTION:')
    print('-' * 40)
    for cat_name, cat_data in stats['size_categories'].items():
        print(f'{cat_name.replace('_', ' ').title():>12} ({cat_data['range']:>5}): '
              f'{cat_data['count']:>5} molecules ({cat_data['percentage']:>5.1f}%)')

    print(f'\nPERCENTILE CUTOFFS:')
    print('-' * 40)
    for p, val in stats['percentiles'].items():
        coverage = stats['coverage_analysis'][f'{p}th_percentile']['coverage']
        excluded = stats['coverage_analysis'][f'{p}th_percentile']['molecules_excluded']
        print(f'{p:>5}th percentile: {val:>2} atoms | '
              f'Coverage: {coverage * 100:>5.1f}% | '
              f'Excluded: {excluded:>4} molecules')


def analyze_sequence_length_distribution(tokenizer, smiles_list: List[str],
                                         plot: bool = True) -> Dict:
    """
    Comprehensive analysis of sequence length distribution for informed decision making.

    Args:
        tokenizer: Built tokenizer instance
        smiles_list: List of SMILES strings from your dataset
        plot: Whether to create distribution plots
    Returns:
        Dictionary with detailed statistics and recommendations
    """
    if not tokenizer.token_to_id:
        raise ValueError('Tokenizer vocabulary not built. Call build_vocab_from_corpus first.')

    lengths = []
    failed_count = 0

    for smiles in smiles_list:
        try:
            tokens = tokenizer.tokenize(smiles)
            lengths.append(len(tokens))
        except:
            failed_count += 1
            continue

    if not lengths:
        raise ValueError('No valid SMILES found for analysis')

    lengths = np.array(lengths)

    # Comprehensive statistics
    percentiles = [50, 75, 90, 95, 97, 98, 99, 99.5, 99.9]
    percentile_values = {p: int(np.percentile(lengths, p)) for p in percentiles}

    stats = {
        'total_molecules': len(smiles_list),
        'valid_molecules': len(lengths),
        'failed_tokenization': failed_count,
        'mean': float(np.mean(lengths)),
        'median': float(np.median(lengths)),
        'std': float(np.std(lengths)),
        'min': int(np.min(lengths)),
        'max': int(np.max(lengths)),
        'percentiles': percentile_values
    }

    # Coverage analysis for different max lengths
    coverage_analysis = {}
    for p in percentiles:
        max_len = percentile_values[p]
        coverage = np.sum(lengths <= max_len) / len(lengths)
        truncated = np.sum(lengths > max_len)
        avg_truncation = np.mean(lengths[lengths > max_len] - max_len) if truncated > 0 else 0

        coverage_analysis[f'{p}th_percentile'] = {
            'max_length': max_len,
            'coverage': float(coverage),
            'molecules_truncated': int(truncated),
            'avg_tokens_lost': float(avg_truncation)
        }

    stats['coverage_analysis'] = coverage_analysis

    # Efficiency analysis (padding overhead)
    efficiency_analysis = {}
    for p in percentiles:
        max_len = percentile_values[p]
        total_tokens_needed = np.sum(lengths)
        total_tokens_with_padding = len(lengths) * max_len
        padding_overhead = (total_tokens_with_padding - total_tokens_needed) / total_tokens_needed

        efficiency_analysis[f'{p}th_percentile'] = {
            'max_length': max_len,
            'padding_overhead': float(padding_overhead),
            'memory_ratio': float(total_tokens_with_padding / total_tokens_needed)
        }

    stats['efficiency_analysis'] = efficiency_analysis

    # Plotting
    if plot:
        _plot_distribution_analysis(lengths, stats)

    return stats


def _plot_distribution_analysis(lengths: np.ndarray, stats: Dict):
    """Create comprehensive plots for distribution analysis."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SMILES Sequence Length Distribution Analysis', fontsize=16)

        # 1. Histogram
        axes[0, 0].hist(lengths, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(stats['mean'], color='red', linestyle='--', label=f'Mean: {stats['mean']:.1f}')
        axes[0, 0].axvline(stats['median'], color='blue', linestyle='--', label=f'Median: {stats['median']:.1f}')
        axes[0, 0].set_xlabel('Sequence Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Sequence Lengths')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Cumulative distribution
        sorted_lengths = np.sort(lengths)
        cumulative = np.arange(1, len(sorted_lengths) + 1) / len(sorted_lengths)
        axes[0, 1].plot(sorted_lengths, cumulative)

        # Mark important percentiles
        for p in [95, 99, 99.5]:
            val = np.percentile(lengths, p)
            axes[0, 1].axvline(val, color='red', alpha=0.7, linestyle='--')
            axes[0, 1].text(val, p / 100, f'{p}%', rotation=90, verticalalignment='bottom')

        axes[0, 1].set_xlabel('Sequence Length')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Coverage vs Efficiency Trade-off
        percentiles = [90, 95, 97, 98, 99, 99.5, 99.9]
        coverages = [stats['coverage_analysis'][f'{p}th_percentile']['coverage'] for p in percentiles]
        overheads = [stats['efficiency_analysis'][f'{p}th_percentile']['padding_overhead'] for p in percentiles]

        axes[1, 0].plot(overheads, coverages, 'o-', markersize=8)
        for i, p in enumerate(percentiles):
            axes[1, 0].annotate(f'{p}%', (overheads[i], coverages[i]),
                                xytext=(5, 5), textcoords='offset points')

        axes[1, 0].set_xlabel('Padding Overhead')
        axes[1, 0].set_ylabel('Data Coverage')
        axes[1, 0].set_title('Coverage vs Efficiency Trade-off')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Molecules truncated by percentile
        truncated = [stats['coverage_analysis'][f'{p}th_percentile']['molecules_truncated'] for p in percentiles]
        axes[1, 1].bar(range(len(percentiles)), truncated, alpha=0.7)
        axes[1, 1].set_xticks(range(len(percentiles)))
        axes[1, 1].set_xticklabels([f'{p}%' for p in percentiles], rotation=45)
        axes[1, 1].set_ylabel('Molecules Truncated')
        axes[1, 1].set_title('Data Loss by Percentile Choice')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        _print_text_summary(stats)

    except ImportError:
        print('Matplotlib not available. Skipping plots.')
        _print_text_summary(stats)


def _print_text_summary(stats: Dict):
    """Print text-based summary."""
    print('\n' + '=' * 80)
    print('SEQUENCE LENGTH ANALYSIS SUMMARY')
    print('=' * 80)
    print(f'Dataset: {stats['total_molecules']} molecules')
    print(
        f'Valid tokenization: {stats['valid_molecules']} ({stats['valid_molecules'] / stats['total_molecules'] * 100:.1f}%)')
    print(f'Mean length: {stats['mean']:.1f} ± {stats['std']:.1f} tokens')
    print(f'Range: {stats['min']} - {stats['max']} tokens')

    print(f'\nPERCENTILE VALUES:')
    print('-' * 40)
    for p, val in stats['percentiles'].items():
        coverage = stats['coverage_analysis'][f'{p}th_percentile']['coverage']
        truncated = stats['coverage_analysis'][f'{p}th_percentile']['molecules_truncated']
        overhead = stats['efficiency_analysis'][f'{p}th_percentile']['padding_overhead']
        print(f'{p:>5}th percentile: {val:>3} tokens | '
              f'Coverage: {coverage * 100:>5.1f}% | '
              f'Truncated: {truncated:>4} | '
              f'Padding: {overhead * 100:>5.1f}%')
