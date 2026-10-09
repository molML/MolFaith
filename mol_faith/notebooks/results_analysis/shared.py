import math
from functools import reduce
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from rdkit import Chem
from scipy.stats import kruskal, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.anova import anova_lm

from mol_faith.target_extraction.benzene_substructure_extraction import (
    BenzeneSubstructureExtractor,
)
from mol_faith.target_extraction.hbond_donor_extraction import HBondDonorExtractor
from mol_faith.target_extraction.largest_conjugated_system_extraction import (
    LargestConjugatedSystemExtractor,
)

REPO_DIR = Path(__file__).resolve().parents[3]
REAL_WORLD_RESULTS = REPO_DIR / 'results' / 'real_world'
REAL_WORLD_DATA = REPO_DIR / 'data' / 'real_world'
SYNTHETIC_RESULTS = REPO_DIR / 'results' / 'synthetic'
REAL_WORLD_DATASETS = {'LogP': ('logp', 'logp'), 'FXA_logD': ('FXa_logd', 'logd'),
                       'FXA_logS': ('FXa_logs', 'logs'), 'ABL1': ('ABL1_pki', 'abl1')}
SYNTHETIC_TARGETS = {'Benzene': ('benzene_target', 'benzene'),
                     'Conjugated System': ('conjugated_system_target', 'conjugated_system'),
                     'Hbond Donor': ('hbond_donor', 'hbond_donor')}

FID_DIR = Path('f_fid_eval', 'beta_0.1', 'top_percentage')
MODELS = ['GCN', 'Transformer', 'CNN', 'GIN', 'GAT']
MODEL_ORDER = ['Transformer', 'CNN', 'GAT', 'GCN', 'GIN']
METHOD_NAMES = {
    'grad_cam': 'GradCAM',
    'guided_grad_cam': 'Guided GradCAM',
    'integrated_gradients': 'IG',
    'kernel_shap': 'KernelSHAP',
    'lime': 'LIME',
    'shap_sampling': 'SHAP',
    'gnn_explainer': 'GNNExplainer',
    'graph_mask': 'GraphMask',
}
METHOD_ORDER = ['IG', 'SHAP', 'KernelSHAP', 'LIME', 'GradCAM', 'Guided GradCAM', 'GNNExplainer', 'GraphMask']
K_EFFECT_COMPARISONS = {'GIN': 'graph_mask', 'GAT': 'graph_mask', 'GCN': 'graph_mask',
                        'CNN': 'kernel_shap', 'Transformer': 'kernel_shap'}

MODEL_COLORS = {'Transformer': '#2D4976', 'CNN': '#A0B1CF', 'GAT': '#D6CAAE', 'GCN': '#928463', 'GIN': '#C09C5D'}
SYNTHETIC_TARGET_COLORS = {'Benzene': '#F0CEA0', 'Conjugated System': '#9D8189', 'Hbond Donor': '#569A9F'}
REAL_WORLD_TARGET_COLORS = {'FXA_logD': '#E0B87A', 'FXA_logS': '#A85F6E', 'ABL1': '#4C8E96', 'LogP': '#496F97'}
NEGATIVE_DEVIATION_COLORS = {
    'shap_sampling': '#5A3A31', 'graph_mask': '#46579B', 'grad_cam': '#F3A712', 'guided_grad_cam': '#F0CEA0',
    'integrated_gradients': '#6D8549', 'kernel_shap': '#ABA6B5', 'lime': '#E0EFBD', 'gnn_explainer': '#8BBF9F',
}
GT_COLS = ['metadata_target_fraction', 'metadata_precision_at_k_gt']


def _set_style(font_scale=1):
    sns.set_style('whitegrid', {'axes.grid': False, 'xtick.bottom': True, 'ytick.left': True, 'font': 'Arial'})
    sns.set_context('paper', font_scale=font_scale)


def _save(fig, output_dir, filename):
    if output_dir is None:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(output_dir) / f'{filename}.png', dpi=450, bbox_inches='tight')
    fig.savefig(Path(output_dir) / f'{filename}.svg', format='svg', dpi=450, bbox_inches='tight')


def _set_spines(ax, linewidth):
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


def map_explanation_method_names(df):
    return df.assign(explanation_method=df['explanation_method'].map(METHOD_NAMES))


def _combine(target_dataframes):
    return pd.concat(
        [map_explanation_method_names(df.assign(target=target))
         for target, df in target_dataframes.items() if not df.empty],
        ignore_index=True,
    )


def _sign_effect_mask(df):
    return ((df['fid_plus'] > 0) & (df['fid_plus_random'] < 0)) | ((df['fid_plus'] < 0) & (df['fid_plus_random'] > 0))


def load_raw_data_all_thres(base_dir, target_name, models=MODELS, thresholds=None, num_splits=5,
                            exclude_methods=(), verbose=True):
    """Load fidelity_results.csv for every model / xAI method / threshold / split below base_dir."""
    base_dir = Path(base_dir)
    methods, found_thresholds = set(), set()
    for model in models:
        model_dir = base_dir / model / FID_DIR
        if not model_dir.exists():
            continue
        for method_dir in model_dir.iterdir():
            if method_dir.is_dir() and method_dir.name not in exclude_methods:
                methods.add(method_dir.name)
                found_thresholds.update(d.name.removeprefix('threshold_') for d in method_dir.iterdir()
                                        if d.is_dir() and d.name.startswith('threshold_'))
    found_thresholds = sorted(found_thresholds, key=float)
    thresholds = found_thresholds if thresholds is None else [t for t in thresholds if t in found_thresholds]

    frames = []
    for model in models:
        for method in sorted(methods):
            for threshold in thresholds:
                for split in range(num_splits):
                    path = (base_dir / model / FID_DIR / method / f'threshold_{threshold}'
                            / f'data_split_{split}' / 'fidelity_results.csv')
                    if path.exists():
                        frames.append(pd.read_csv(path).dropna(subset=['fid_plus']).assign(
                            model=model, explanation_method=method, target_name=target_name,
                            threshold=float(threshold), data_split_idx=split))
    if not frames:
        print(f'No data loaded for target {target_name!r}')
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    first_cols = ['target_name', 'model', 'smiles', 'data_split_idx', 'threshold', 'target', 'explanation_method', 'fid_plus']
    df = df[first_cols + [c for c in df.columns if c not in first_cols]]
    if verbose:
        print(f'{target_name}: {len(df):,} rows | models {sorted(df["model"].unique())} | '
              f'methods {sorted(df["explanation_method"].unique())} | thresholds {sorted(df["threshold"].unique())}')
    return df


def load_raw_data(base_dir, target_name, models=MODELS, threshold='0.2', num_splits=5, exclude_methods=()):
    df = load_raw_data_all_thres(base_dir, target_name, models, [threshold], num_splits, exclude_methods)
    return df[['target_name', 'model', 'smiles', 'data_split_idx', 'target', 'explanation_method', 'fid_plus']]


def load_raw_gt_data(base_dir, target_name, models=MODELS, num_splits=5, exclude_methods=()):
    base_dir = Path(base_dir)
    target_dirs = [d for model in models if (base_dir / model / 'gt_eval').exists()
                   for d in (base_dir / model / 'gt_eval').iterdir() if d.is_dir()]
    targets = sorted({d.name for d in target_dirs})
    methods = sorted({m.name for d in target_dirs for m in d.iterdir()
                      if m.is_dir() and m.name not in exclude_methods})

    frames = []
    for model in models:
        for target in targets:
            for method in methods:
                for split in range(num_splits):
                    path = (base_dir / model / 'gt_eval' / target / method / f'data_split_{split}'
                            / 'target_fraction_results.csv')
                    if path.exists():
                        frames.append(pd.read_csv(path).dropna(subset=GT_COLS).assign(model=model, target_name=target_name))
    df = pd.concat(frames, ignore_index=True)
    print(f'{target_name}: {len(df):,} rows | targets {targets} | methods {methods}')
    return df[['target_name', 'model', 'smiles', 'data_split_idx', 'target', 'explanation_method'] + GT_COLS]


def process_fidelity_df(df, epsilon=1e-10):
    """Faithfulness score S per XAI row against the random baseline of the same molecule (and threshold):
    R = (fid - fid_random) / (|fid| + |fid_random| + eps), S = R if fid >= 0 else -|R|."""
    keys = ['model', 'smiles', 'data_split_idx', 'target'] + (['threshold'] if 'threshold' in df else [])
    is_random = df['explanation_method'] == 'random_explanation'
    random = df.loc[is_random, keys + ['fid_plus']].rename(columns={'fid_plus': 'fid_plus_random'})
    result = df[~is_random].merge(random, on=keys, how='left')

    fid, fid_random = result['fid_plus'], result['fid_plus_random']
    r = (fid - fid_random) / (np.abs(fid) + np.abs(fid_random) + epsilon)
    result['normalized_fid'] = np.where(fid >= 0, r, -np.abs(r))
    result['regime'] = np.where(fid >= 0, 'faithful', 'unfaithful')
    print(f'{len(result):,} rows | {result["regime"].value_counts().to_dict()} | normalized_fid in '
          f'[{result["normalized_fid"].min():.3f}, {result["normalized_fid"].max():.3f}]')
    return result


def _power10(x, sigfigs, latex=False):
    if x == 0:
        return '0'
    exp = math.floor(math.log10(abs(x)))
    coeff = f'{x / 10 ** exp:.{sigfigs - 1}f}'
    return f'{coeff} \\times 10^{{{exp}}}' if latex else f'{coeff} × 10^{exp}'


def format_multi_model_comparison_table(csv_path, model_order=MODEL_ORDER,
                                        columns=('rmse_mean', 'rmse_std', 'mae_mean', 'mae_std', 'f3_mean', 'f3_std'),
                                        exact_format_substrings=('pearson_r', 'r2', 'f3'), power10_sigfigs=2):
    df = pd.read_csv(csv_path).set_index('model')
    for col in df.columns:
        if any(s in col for s in exact_format_substrings):
            df[col] = df[col].map(lambda x: f'{x:.2}')
        else:
            df[col] = df[col].map(lambda x: _power10(x, power10_sigfigs))
    return df.reindex(list(model_order))[list(columns)]


def format_latex_model_comparison_table(csv_path, caption, label, model_order=MODEL_ORDER,
                                        sigfigs=2, q2_col='f3', q2_decimals=2):
    df = pd.read_csv(csv_path).set_index('model')

    def pow10_pair(metric, row):
        return f"${_power10(row[f'{metric}_mean'], sigfigs, True)} \\pm {_power10(row[f'{metric}_std'], sigfigs, True)}$"

    rows = []
    for model in model_order:
        row = df.loc[model]
        q2 = f"${row[f'{q2_col}_mean']:.{q2_decimals}f} \\pm {row[f'{q2_col}_std']:.{q2_decimals}f}$"
        rows.append(f"{model} & {pow10_pair('rmse', row)} & {pow10_pair('mae', row)} & {q2} \\\\")
    return '\n'.join([
        r'\begin{table}[H]', r'\centering', rf'\caption{{{caption}}}', rf'\label{{{label}}}',
        r'\begin{tabular}{lccc}', r'\toprule', r' & \textbf{RMSE} & \textbf{MAE} & \textbf{Q2} \\', r'\midrule',
        *rows,
        r'\bottomrule', r'\end{tabular}', r'\end{table}',
    ])


def combine_and_calculate_variance(target_dataframes):
    summary = (
        _combine(target_dataframes)
        .groupby(['model', 'explanation_method', 'target'])['normalized_fid']
        .agg(n='count', mean='mean', median='median', std='std', var='var')
        .reset_index()
        .sort_values(['target', 'model', 'explanation_method'])
    )
    cols = ['mean', 'median', 'std', 'var']
    summary[cols] = summary[cols].round(3)
    return summary.sort_values('median', ascending=False)


def calculate_stats_by_method_target(target_dataframes, value_col='normalized_fid',
                                     group_cols=('explanation_method', 'target')):
    group_cols = list(group_cols)
    df = _combine(target_dataframes)
    return (
        df[np.isfinite(df[value_col])]
        .groupby(group_cols)[value_col]
        .agg(n='count', median='median', std='std', var='var')
        .reset_index()
        .sort_values(group_cols[::-1])
    )


def plot_std_by_method_target(variance_df, figsize=(3, 2.5), output_dir=None, plot_linewidth=0.5, labelsize=7,
                              tick_label_size=7, tick_width=0.5, tick_length=2, point_size=5, alpha=0.7):
    _set_style()
    figures = {}
    for target in variance_df['target'].unique():
        df_target = variance_df[variance_df['target'] == target]
        fig, ax = plt.subplots(figsize=figsize)
        sns.stripplot(data=df_target, x='explanation_method', y='std', hue='model', palette=MODEL_COLORS,
                      order=METHOD_ORDER, dodge=False, jitter=False, alpha=alpha, linewidth=plot_linewidth,
                      size=point_size, ax=ax, legend=False)
        ax.set_xlabel('xAI Method', fontsize=labelsize)
        ax.set_ylabel('Standard Deviation (Fid⁺)', fontsize=labelsize)
        ax.tick_params(axis='y', labelsize=tick_label_size, width=tick_width, length=tick_length)
        ax.tick_params(axis='x', labelsize=tick_label_size, width=tick_width, length=tick_length, rotation=45)
        ax.set_xticklabels([])
        _set_spines(ax, plot_linewidth)
        ax.axhline(y=0, color='grey', linestyle='-', linewidth=plot_linewidth, alpha=0.5, zorder=1)
        ax.axhline(y=df_target['std'].median(), color='black', linestyle='dashed', linewidth=plot_linewidth,
                   alpha=0.5, zorder=1)
        plt.tight_layout()
        _save(fig, output_dir, f"std_by_method_{target.replace(' ', '_').lower()}")
        plt.show()
        figures[target] = fig
    return figures


def compute_k_effect(df, use_normalized_fid=True, default_k=0.2):
    """Per (model, method, molecule, split): fid summary across thresholds and delta_<k> relative to default_k."""
    fid_col = 'normalized_fid' if use_normalized_fid else 'fid_plus'
    keys = ['model', 'explanation_method', 'smiles', 'data_split_idx']
    grouped = df.groupby(keys)[fid_col]
    by_k = df.pivot_table(index=keys, columns='threshold', values=fid_col)
    summary = pd.DataFrame({
        'fid_at_20': by_k[default_k],
        'fid_min': grouped.min(),
        'fid_max': grouped.max(),
        'fid_std': grouped.std(ddof=0),
        'fid_mean': grouped.mean(),
    })
    deltas = by_k.sub(by_k[default_k], axis=0).add_prefix('delta_')
    deltas.columns.name = None
    return pd.concat([summary, deltas], axis=1).reset_index()


def bin_by_fid_percentile(df, method1='shap_sampling', model='GIN'):
    """Low/Medium/High FID bins from method1's 33rd/66th fid_at_20 percentiles per model, applied to every
    row of that model (so rows of the compared method are binned by method1's thresholds)."""
    if model is not None:
        df = df[df['model'] == model]
    binned = []
    for m in df['model'].unique():
        base = pd.to_numeric(df.loc[(df['model'] == m) & (df['explanation_method'] == method1), 'fid_at_20'],
                             errors='coerce').dropna()
        if base.empty:
            continue
        q33, q66 = base.quantile([0.33, 0.66])
        subset = df[df['model'] == m].copy()
        fid = pd.to_numeric(subset['fid_at_20'], errors='coerce')
        subset['fid_bin'] = np.select([fid <= q33, fid >= q66], ['Low FID', 'High FID'], default='Medium FID')
        binned.append(subset)
    if not binned:
        raise ValueError('No data to bin; check that method1 rows exist for the given model.')
    return pd.concat(binned, ignore_index=True)


def plot_delta_mean_bars_by_fid_bin(df, method1='shap_sampling', method2='graph_mask', model='GIN', k_cols=None,
                                    figsize=(6.5, 3.2), linewidth=0.5, plot_linewidth=0.5, labelsize=8,
                                    tick_label_size=8, tick_width=0.5, tick_length=2, show_legend=True,
                                    bin_tick_labels=('≤33', '34-65', '≥66'), ax=None):
    k_cols = k_cols or [f'delta_{i / 10:.1f}' for i in range(1, 10)]
    df = df[df['explanation_method'].isin([method1, method2])].assign(delta_mean_all_k=df[k_cols].mean(axis=1))
    df_plot = bin_by_fid_percentile(df, method1=method1, model=model)
    df_plot['method_display'] = df_plot['explanation_method'].map(lambda x: METHOD_NAMES.get(x, x))

    _set_style()
    own_fig = ax is None
    fig, ax = plt.subplots(figsize=figsize) if own_fig else (ax.figure, ax)
    sns.barplot(data=df_plot, x='fid_bin', y='delta_mean_all_k', hue='method_display',
                order=['Low FID', 'Medium FID', 'High FID'], palette=['#ABA6B5', '#8BBF9F'], ax=ax,
                linewidth=linewidth, errorbar='se', capsize=0.15, err_kws={'linewidth': 0.5})
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=plot_linewidth, alpha=0.5, zorder=1)
    ax.set_xlabel('Fid+ Percentiles', fontsize=labelsize)
    ax.set_ylabel('Delta Fid+ Mean (All k)', fontsize=labelsize)
    ax.set_xticks(range(len(bin_tick_labels)))
    ax.set_xticklabels(bin_tick_labels)
    ax.tick_params(axis='both', labelsize=tick_label_size, width=tick_width, length=tick_length)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    _set_spines(ax, plot_linewidth)
    if show_legend:
        ax.legend(fontsize=tick_label_size, title=None)
    elif ax.get_legend():
        ax.get_legend().remove()
    if own_fig:
        plt.tight_layout()
    return fig, ax


def _model_grid(models, nrows, ncols, figsize_per_panel, draw, labelsize, output_dir, filename):
    _set_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows))
    axes_flat = np.atleast_1d(axes).flatten()
    for ax, model in zip(axes_flat, models):
        draw(model, ax)
        ax.set_title(model, fontsize=labelsize)
    for ax in axes_flat[len(models):]:
        ax.set_visible(False)
    fig.tight_layout()
    _save(fig, output_dir, filename)
    return fig, axes


def plot_delta_bars_grid_by_model(df, models=('GIN', 'GAT', 'GCN', 'CNN', 'Transformer'), method1='shap_sampling',
                                  comparisons=K_EFFECT_COMPARISONS, reverse=False, nrows=3, ncols=2,
                                  figsize_per_panel=(3, 2.5), labelsize=8, output_dir=None,
                                  filename='k_effect_bar_grid', **bar_kwargs):
    """One plot_delta_mean_bars_by_fid_bin panel per model; reverse=True bins by the comparison method instead."""
    def draw(model, ax):
        m1, m2 = (comparisons[model], method1) if reverse else (method1, comparisons[model])
        plot_delta_mean_bars_by_fid_bin(df, method1=m1, method2=m2, model=model, labelsize=labelsize, ax=ax,
                                        **bar_kwargs)

    return _model_grid(models, nrows, ncols, figsize_per_panel, draw, labelsize, output_dir, filename)


def analyze_negative_deviation_k(df, model='GIN', k_range=None):
    df = df[df['model'] == model]
    if df.empty:
        raise ValueError(f'No data found for model={model}')
    if k_range is None:
        k_range = sorted(float(c.split('_')[1]) for c in df.columns if c.startswith('delta_'))

    results = {}
    for method in df['explanation_method'].unique():
        method_df = df[df['explanation_method'] == method]
        rows = []
        for k in k_range:
            col = f'delta_{k:.1f}'
            if col not in method_df:
                continue
            values = method_df[col].dropna()
            if values.empty:
                continue
            rows.append({
                'k': k, 'mean': values.mean(), 'median': values.median(), 'std': values.std(),
                'q25': values.quantile(0.25), 'q75': values.quantile(0.75), 'n_samples': len(values),
                'n_negative': (values < 0).sum(), 'pct_negative': (values < 0).sum() / len(values) * 100,
            })
        results[f'{method}_stats'] = pd.DataFrame(rows)
    return results


def plot_negative_deviation_k_analysis(results, statistic='mean', figsize=(7, 4), linewidth=1, plot_linewidth=0.5,
                                       labelsize=9, tick_label_size=8, tick_width=0.5, tick_length=2,
                                       show_legend=True, ax=None):
    _set_style()
    own_fig = ax is None
    fig, ax = plt.subplots(figsize=figsize) if own_fig else (ax.figure, ax)
    for method, color in NEGATIVE_DEVIATION_COLORS.items():
        stats = results.get(f'{method}_stats')
        if stats is None or stats.empty:
            continue
        sem = stats['std'] / np.sqrt(stats['n_samples'])
        sns.lineplot(data=stats, x='k', y=statistic, ax=ax, color=color, linewidth=linewidth, marker='o',
                     markersize=5, label=METHOD_NAMES[method])
        ax.fill_between(stats['k'], stats[statistic] - sem, stats[statistic] + sem, color=color, alpha=0.18)

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=plot_linewidth, alpha=0.6, zorder=1)
    ax.set_xlabel('k (Threshold)', fontsize=labelsize)
    ax.set_ylabel(f'Delta Fid+ {statistic.capitalize()}', fontsize=labelsize)
    ax.tick_params(axis='both', labelsize=tick_label_size, width=tick_width, length=tick_length)
    _set_spines(ax, plot_linewidth)
    if show_legend:
        ax.legend(fontsize=tick_label_size)
    elif ax.get_legend():
        ax.get_legend().remove()
    if own_fig:
        plt.tight_layout()
    return fig, ax


def plot_negative_deviation_grid_by_model(df, models=('GIN', 'GAT', 'GCN', 'CNN', 'Transformer'), statistic='mean',
                                          k_range=None, nrows=3, ncols=2, figsize_per_panel=(4, 3), labelsize=9,
                                          output_dir=None, filename='negative_deviation_grid', **line_kwargs):
    def draw(model, ax):
        results = analyze_negative_deviation_k(df, model=model, k_range=k_range)
        plot_negative_deviation_k_analysis(results, statistic=statistic, labelsize=labelsize, ax=ax, **line_kwargs)

    return _model_grid(models, nrows, ncols, figsize_per_panel, draw, labelsize, output_dir, filename)


def plot_k_effect_bars(k_effect_df, output_dir):
    """Delta Fid+ bars per model (SHAP vs K_EFFECT_COMPARISONS) and the combined grids in both binning directions."""
    for model, method2 in K_EFFECT_COMPARISONS.items():
        fig, _ = plot_delta_mean_bars_by_fid_bin(k_effect_df, method2=method2, model=model, figsize=(3, 2.5),
                                                 show_legend=False)
        fig.savefig(Path(output_dir) / f'k_effect_bar_{model}.svg', dpi=300)
        plt.show()
    for reverse, filename in ((False, 'k_effect_bar_grid'), (True, 'k_effect_bar_grid_reverse')):
        plot_delta_bars_grid_by_model(k_effect_df, reverse=reverse, output_dir=output_dir, filename=filename)
        plt.show()


def plot_negative_deviations(k_effect_df, output_dir):
    """Delta Fid+ vs k curves per model and as a combined grid."""
    for model in K_EFFECT_COMPARISONS:
        fig, _ = plot_negative_deviation_k_analysis(analyze_negative_deviation_k(k_effect_df, model=model),
                                                    figsize=(5, 3))
        fig.savefig(Path(output_dir) / f'negative_deviation_analysis_{model}.svg', dpi=300)
        plt.show()
    plot_negative_deviation_grid_by_model(k_effect_df, output_dir=output_dir)
    plt.show()


def _draw_minmax(ax, df, group_col, value_col, label_order, targets, target_colors, gap, marker, size):
    n = len(targets)
    group_width = 0.8 / n
    offsets = np.linspace(-0.461 + group_width / 2, 0.461 - group_width / 2, n) if n > 1 else [0]
    for i, label in enumerate(label_order):
        for j, target in enumerate(targets):
            vals = df.loc[(df[group_col] == label) & (df['target'] == target), value_col].dropna()
            if vals.empty:
                continue
            color = target_colors[target] if target_colors else 'grey'
            for y in (vals.min(), vals.max()):
                ax.scatter(i + offsets[j] * (1 - gap), y, marker=marker, s=size, color=color,
                           edgecolor='black', linewidth=1, alpha=0.8, zorder=10)


def _grouped_finish(ax, group_col, targets, target_colors, labelsize, tick_label_size, tick_width, tick_length,
                    tick_rotation, ylim, show_xtick_labels, show_ytick_labels, enable_reference_yticks,
                    reference_fig, default_yticks, plot_linewidth, show_legend, legend_fontsize, legend_title,
                    minmax_marker=None):
    ax.set_xlabel('Model' if group_col == 'model' else 'xAI Method', fontsize=labelsize)
    ax.set_ylabel('Fid+', fontsize=labelsize)
    ax.tick_params(axis='y', labelsize=tick_label_size, width=tick_width, length=tick_length)
    ax.tick_params(axis='x', labelsize=tick_label_size, width=tick_width, length=tick_length,
                   labelrotation=tick_rotation)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    if ylim is not None:
        ax.set_ylim(ylim)
    if not show_xtick_labels:
        ax.set_xticklabels([])
    if not show_ytick_labels:
        ax.set_yticklabels([])
    if enable_reference_yticks and group_col != 'model':
        if reference_fig is not None:
            ref_ax = reference_fig.get_axes()[0]
            ref_ylim = ref_ax.get_ylim()
            ax.set_yticks([t for t in ref_ax.get_yticks() if ref_ylim[0] <= t <= ref_ylim[1]])
            ax.set_ylim(ref_ylim)
        else:
            ax.set_yticks(list(default_yticks))
    _set_spines(ax, plot_linewidth)

    if show_legend:
        handles = [Patch(facecolor=target_colors.get(t, 'grey') if target_colors else 'grey', label=t) for t in targets]
        labels = list(targets)
        if minmax_marker:
            handles.append(Line2D([0], [0], marker=minmax_marker, color='gray', markerfacecolor='gray',
                                  markeredgecolor='black', markersize=8, linewidth=0, alpha=0.8))
            labels.append('Min/Max')
        ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=legend_fontsize, title=legend_title)
    plt.tight_layout()


def create_grouped_boxplot(target_dataframes, value_col, group_col, target_colors=None, hue_order=None,
                           figsize=(7, 3), output_dir=None, linewidth=0.5, box_width=0.8, gap=0.09,
                           show_minmax=False, minmax_marker='D', minmax_size=12, plot_linewidth=0.5, labelsize=7,
                           tick_label_size=7, tick_width=0.5, tick_length=2, tick_rotation=45,
                           show_xtick_labels=True, show_legend=False, legend_fontsize=11, legend_title='Target',
                           enable_reference_yticks=False, reference_fig=None,
                           default_yticks=(0.12, 0.08, 0.05, 0.02, 0.00, -0.02, -0.05, -0.08)):
    """One box per (group_col, target), colored by target."""
    full_df = _combine(target_dataframes)
    label_order = MODEL_ORDER if group_col == 'model' else METHOD_ORDER
    targets = list(hue_order or target_dataframes)

    fig, ax = plt.subplots(figsize=figsize)
    _set_style()
    sns.boxplot(data=full_df, x=group_col, y=value_col, hue='target', hue_order=targets, order=label_order,
                linewidth=linewidth, ax=ax, palette=target_colors, width=box_width, showfliers=False, gap=gap,
                dodge=True, legend=False)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=plot_linewidth, alpha=0.5, zorder=1)
    if show_minmax:
        _draw_minmax(ax, full_df, group_col, value_col, label_order, targets, target_colors, gap, minmax_marker,
                     minmax_size)
    _grouped_finish(ax, group_col, targets, target_colors, labelsize, tick_label_size, tick_width, tick_length,
                    tick_rotation, None, show_xtick_labels, True, enable_reference_yticks, reference_fig,
                    default_yticks, plot_linewidth, show_legend, legend_fontsize, legend_title,
                    minmax_marker if show_minmax else None)
    prefix = 'model' if group_col == 'model' else 'method'
    _save(fig, output_dir, f'{prefix}_grouped_boxplot')
    plt.show()
    return fig


def create_grouped_boxplot_with_violin(target_dataframes, value_col, group_col, target_colors=None, hue_order=None,
                                       figsize=(7, 3), output_dir=None, linewidth=0.5, box_width=0.8, gap=0.09,
                                       show_minmax=False, minmax_marker='D', minmax_size=12, plot_linewidth=0.5,
                                       labelsize=7, tick_label_size=7, tick_width=0.5, tick_length=2,
                                       tick_rotation=45, show_xtick_labels=False, show_ytick_labels=False,
                                       show_legend=False, legend_fontsize=11, legend_title='Target',
                                       enable_reference_yticks=False, reference_fig=None,
                                       default_yticks=(0.12, 0.08, 0.05, 0.02, 0.00, -0.02, -0.05, -0.08),
                                       violin_alpha=1, violin_gap=1.9, y_clip=(-1, 1)):
    """Violins per (group_col, target) under unfilled black boxes. y_clip cuts the KDE at the observed range and
    fixes the y-limits (with a small margin so caps at +-1 stay visible); y_clip=None disables both."""
    full_df = _combine(target_dataframes)
    label_order = MODEL_ORDER if group_col == 'model' else METHOD_ORDER
    targets = list(hue_order or target_dataframes)

    fig, ax = plt.subplots(figsize=figsize)
    _set_style()
    palette = {t: mcolors.to_rgba(target_colors.get(t, 'grey') if target_colors else 'grey', alpha=violin_alpha)
               for t in targets}
    sns.violinplot(data=full_df, x=group_col, y=value_col, hue='target', hue_order=targets, order=label_order,
                   ax=ax, palette=palette, dodge=True, gap=violin_gap, legend=False, linewidth=0,
                   cut=0 if y_clip is not None else 2, zorder=1)
    edge = {'zorder': 3}
    sns.boxplot(data=full_df, x=group_col, y=value_col, hue='target', hue_order=targets, order=label_order,
                linewidth=linewidth, ax=ax, palette=['black'] * len(targets), width=box_width, showfliers=False,
                gap=gap, dodge=True, legend=False, fill=False, boxprops=edge, whiskerprops=edge, capprops=edge,
                medianprops=edge)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=plot_linewidth, alpha=0.5, zorder=1)
    if show_minmax:
        _draw_minmax(ax, full_df, group_col, value_col, label_order, targets, target_colors, gap, minmax_marker,
                     minmax_size)
    ylim = None
    if y_clip is not None:
        margin = 0.03 * (y_clip[1] - y_clip[0])
        ylim = (y_clip[0] - margin, y_clip[1] + margin)
    _grouped_finish(ax, group_col, targets, target_colors, labelsize, tick_label_size, tick_width, tick_length,
                    tick_rotation, ylim, show_xtick_labels, show_ytick_labels, enable_reference_yticks,
                    reference_fig, default_yticks, plot_linewidth, show_legend, legend_fontsize, legend_title,
                    minmax_marker if show_minmax else None)
    prefix = 'model' if group_col == 'model' else 'method'
    _save(fig, output_dir, f'{prefix}_grouped_boxplot_with_violin')
    plt.show()
    return fig


def create_grouped_violinplot(target_dataframes, value_col, group_col, target_colors=None, hue_order=None,
                              figsize=(7, 3), output_dir=None, max_points_per_group=300, random_state=0,
                              plot_linewidth=0.5, labelsize=9, tick_label_size=8, tick_width=0.5, tick_length=2,
                              tick_rotation=45, show_xtick_labels=True, show_legend=False, legend_fontsize=11,
                              legend_title='Target', enable_reference_yticks=False, reference_fig=None,
                              default_yticks=(0.12, 0.08, 0.05, 0.02, 0.00, -0.02, -0.05, -0.08)):
    """Violins per (group_col, target) on a random subsample of max_points_per_group points per cell
    (None plots everything)."""
    full_df = _combine(target_dataframes)
    label_order = MODEL_ORDER if group_col == 'model' else METHOD_ORDER
    targets = list(hue_order or target_dataframes)
    if max_points_per_group is not None:
        full_df = full_df.groupby([group_col, 'target'], group_keys=False)[list(full_df.columns)].apply(
            lambda g: g.sample(n=min(len(g), max_points_per_group), random_state=random_state))

    fig, ax = plt.subplots(figsize=figsize)
    _set_style()
    sns.violinplot(data=full_df, x=group_col, y=value_col, hue='target', hue_order=targets, order=label_order,
                   palette=target_colors, dodge=True, linewidth=0, ax=ax, legend=False)
    ax.axhline(y=0, color='grey', linestyle='-', linewidth=plot_linewidth, alpha=0.5, zorder=1)
    if not show_xtick_labels:
        ax.set_xticks(range(len(label_order)))
    _grouped_finish(ax, group_col, targets, target_colors, labelsize, tick_label_size, tick_width, tick_length,
                    tick_rotation, (-1, 1), show_xtick_labels, True, enable_reference_yticks, reference_fig,
                    default_yticks, plot_linewidth, show_legend, legend_fontsize, legend_title)
    prefix = 'model' if group_col == 'model' else 'method'
    _save(fig, output_dir, f'{prefix}_grouped_violinplot')
    plt.show()
    return fig


def aggregate_and_merge_normalized_fid(target_dfs):
    """Median normalized_fid per (model, method) for every target as one wide table, plus MIN/MAX rows."""
    keys = ['model', 'explanation_method']
    medians = [df.groupby(keys)['normalized_fid'].median().rename(f'fid_normalized_median_{name}').reset_index()
               for name, df in target_dfs.items()]
    merged = reduce(lambda a, b: a.merge(b, on=keys, how='outer'), medians)
    fid_cols = [c for c in merged.columns if c.startswith('fid_normalized_median_')]
    extremes = pd.DataFrame([
        {'model': 'MIN', 'explanation_method': 'MIN', **merged[fid_cols].min().to_dict()},
        {'model': 'MAX', 'explanation_method': 'MAX', **merged[fid_cols].max().to_dict()},
    ])
    return pd.concat([merged, extremes], ignore_index=True)


def perform_pca(df_with_minmax):
    fid_columns = [c for c in df_with_minmax.columns if c.startswith('fid_')]
    df = df_with_minmax[df_with_minmax[fid_columns].notna().all(axis=1)]
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    scores = pca.fit_transform(scaler.fit_transform(df[fid_columns].values))
    pca_df = df[['model', 'explanation_method']].assign(PC1=scores[:, 0], PC2=scores[:, 1])
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    print(f'PCA on {fid_columns}, explained variance ratio {pca.explained_variance_ratio_}')
    return pca_df, pca, scaler, loadings, fid_columns


def plot_pca(pca_df, pca, loadings, fid_columns, show_loadings=False, figsize=(14, 10), font_scale=1,
             plot_linewidth=0.5, labelsize=9, tick_label_size=8, tick_width=0.5, tick_length=2, point_size=12,
             save_path=None):
    """PC1 vs PC2 colored by model; MIN/MAX rows as black points joined by a dashed line; square, symmetric axes."""
    _set_style(font_scale)
    fig, ax = plt.subplots(figsize=figsize)
    is_extreme = pca_df['model'].isin(['MIN', 'MAX'])
    sns.scatterplot(data=pca_df[~is_extreme], x='PC1', y='PC2', hue='model', edgecolor='black',
                    palette=MODEL_COLORS, s=point_size, alpha=1, linewidth=0.3, ax=ax)
    for _, row in pca_df[is_extreme].iterrows():
        ax.scatter(row['PC1'], row['PC2'], c='black', marker='o', s=point_size, edgecolor='black', alpha=1.0,
                   linewidth=0.3, zorder=3)
    if is_extreme.sum() >= 2:
        ends = pca_df.set_index('model').loc[['MIN', 'MAX']]
        ax.plot(ends['PC1'], ends['PC2'], 'k--', linewidth=plot_linewidth, alpha=0.7)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.3f} explained variance)', fontsize=labelsize)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.3f} explained variance)', fontsize=labelsize)
    _set_spines(ax, plot_linewidth)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=plot_linewidth, alpha=0.7, zorder=0)
    ax.axvline(x=0, color='lightgray', linestyle='-', linewidth=plot_linewidth, alpha=0.7, zorder=0)

    if show_loadings:
        scale = max(np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0]) / 3
        for (x, y), col in zip(loadings[:, :2], fid_columns):
            ax.arrow(0, 0, x * scale, y * scale, head_width=scale * 0.05, head_length=scale * 0.08,
                     fc='black', ec='black', alpha=0.8, linewidth=plot_linewidth)
            ax.text(x * scale * 1.15, y * scale * 1.15, col.replace('Fid_', '').replace('_target', ''),
                    fontsize=6, ha='center', va='center')

    max_abs = np.abs([*ax.get_xlim(), *ax.get_ylim()]).max()
    ax.set_xlim(-max_abs, max_abs)
    ax.set_ylim(-max_abs, max_abs)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', labelsize=tick_label_size, width=tick_width, length=tick_length)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, format='svg', dpi=450, facecolor='white')
    plt.show()
    return fig, ax


def plot_boxplot_grid_by_model(df, value_col='normalized_fid', figsize_per_panel=(3, 3), plot_linewidth=0.5,
                               labelsize=9, tick_label_size=8, output_dir=None,
                               filename='model_method_boxplot_grid'):
    """One panel per model with a boxplot of value_col across xAI methods."""
    df = map_explanation_method_names(df)
    models = [m for m in MODEL_ORDER if m in df['model'].unique()]
    _set_style()
    fig, axes = plt.subplots(1, len(models), figsize=(figsize_per_panel[0] * len(models), figsize_per_panel[1]),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, model in zip(axes, models):
        sub = df[df['model'] == model]
        sns.boxplot(data=sub, x='explanation_method', y=value_col, ax=ax, showfliers=False,
                    order=[m for m in METHOD_ORDER if m in sub['explanation_method'].unique()])
        ax.axhline(y=0, color='grey', linestyle='-', linewidth=plot_linewidth, alpha=0.5, zorder=1)
        ax.set_title(model, fontsize=labelsize)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=tick_label_size, labelrotation=45)
        ax.tick_params(axis='y', labelsize=tick_label_size)
        _set_spines(ax, plot_linewidth)
    axes[0].set_ylabel('Fid+', fontsize=labelsize)
    plt.tight_layout()
    _save(fig, output_dir, filename)
    return fig


def calculate_total_atoms(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumAtoms() if mol is not None else None


def extract_gt_atoms_number(smiles, target):
    extractors = {
        'benzene': BenzeneSubstructureExtractor,
        'conjugated_system': LargestConjugatedSystemExtractor,
        'hbond_donor': HBondDonorExtractor,
    }
    gt_atoms = extractors[target]().get_atom_and_bond_indices(Chem.MolFromSmiles(smiles))[0]
    return len(np.array(gt_atoms).flatten())


def summarize_molecule_stats(raw_data_paths):
    rows = []
    for name, path in raw_data_paths.items():
        smiles = pd.read_csv(path)['SMILES'].unique()
        atoms = pd.Series([calculate_total_atoms(s) for s in smiles])
        rows.append({'dataset': name, 'n_molecules': len(smiles), 'mean_num_atoms': atoms.mean(),
                     'std_num_atoms': atoms.std()})
    return pd.DataFrame(rows).round(2)


def pair_xai_methods(df, method_a='integrated_gradients', method_b='shap_sampling', label_a='ig', label_b='shap'):
    """Per-molecule normalized_fid of two xAI methods side by side (same model, smiles, split)."""
    keys = ['model', 'smiles', 'data_split_idx']
    a = df.loc[df['explanation_method'] == method_a, keys + ['target_name', 'normalized_fid']]
    b = df.loc[df['explanation_method'] == method_b, keys + ['normalized_fid']]
    return a.rename(columns={'normalized_fid': f'normalized_fid_{label_a}'}).merge(
        b.rename(columns={'normalized_fid': f'normalized_fid_{label_b}'}), on=keys)


def compute_correlation_summary(df, group_cols, value_col_a, value_col_b):
    rows = []
    for keys, group in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        pearson_r, pearson_p = pearsonr(group[value_col_a], group[value_col_b])
        spearman_r, spearman_p = spearmanr(group[value_col_a], group[value_col_b])
        rows.append({**dict(zip(group_cols, keys)), 'n': len(group), 'pearson_r': pearson_r,
                     'pearson_p': pearson_p, 'spearman_r': spearman_r, 'spearman_p': spearman_p})
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def plot_correlation_scatter_grid(df, corr_df, row, x, y, xlabel, ylabel, row_order=None, output_dir=None,
                                  filename='correlation_scatter_grid'):
    """Scatter of x vs y per (row, model) panel, annotated with the matching corr_df Spearman/Pearson r."""
    row_order = row_order or df[row].unique().tolist()
    model_order = [m for m in MODEL_ORDER if m in df['model'].unique()]
    corr_lookup = corr_df.set_index([row, 'model'])

    g = sns.FacetGrid(df, row=row, col='model', row_order=row_order, col_order=model_order, margin_titles=True,
                      height=2.6, aspect=1)
    for (row_value, model), ax in g.axes_dict.items():
        sub = df[(df[row] == row_value) & (df['model'] == model)]
        ax.scatter(sub[x], sub[y], s=10, alpha=0.5, color=MODEL_COLORS.get(model, '#555555'))
        ax.axline((0, 0), slope=1, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.axhline(0, color='grey', linewidth=0.5, alpha=0.4)
        ax.axvline(0, color='grey', linewidth=0.5, alpha=0.4)
        if (row_value, model) in corr_lookup.index:
            r = corr_lookup.loc[(row_value, model)]
            ax.text(0.05, 0.95, f"ρ={r['spearman_r']:.2f}\nr={r['pearson_r']:.2f}", transform=ax.transAxes,
                    ha='left', va='top', fontsize=8)
    g.set_axis_labels(xlabel, ylabel)
    g.set_titles(row_template='{row_name}', col_template='{col_name}')
    plt.tight_layout()
    _save(g.figure, output_dir, filename)
    return g


def plot_correlation_heatmap(corr_df, columns='task', index='model', values='spearman_r', column_order=None,
                             cbar_label='Spearman rho (IG vs SHAP)', xlabel='Task', ylabel='Model', figsize=(3, 4),
                             output_dir=None, filename='ig_vs_shap_correlation_heatmap'):
    pivot = corr_df.pivot(index=index, columns=columns, values=values)
    pivot = pivot.reindex([m for m in MODEL_ORDER if m in pivot.index])
    pivot = pivot[[c for c in (column_order or pivot.columns) if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', vmin=-1, vmax=1, cbar_kws={'label': cbar_label}, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    _save(fig, output_dir, filename)
    return fig


def plot_alpha_plus_times_s_value(proc_all_thres_df, model, explanation_method, data_split_idx):
    sub = proc_all_thres_df[(proc_all_thres_df['model'] == model)
                            & (proc_all_thres_df['explanation_method'] == explanation_method)
                            & (proc_all_thres_df['data_split_idx'] == data_split_idx)]
    mean_by_k = (sub['metadata_alpha_plus'] * sub['metadata_s_value']).groupby(sub['threshold']).mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(mean_by_k.index, mean_by_k.values, marker='o')
    ax.set_xlabel('k (threshold)')
    ax.set_ylabel('mean alpha_plus * s_value')
    ax.set_title(f'{model} / {explanation_method} / split {data_split_idx}')
    plt.tight_layout()
    return fig, ax


def plot_fidelity_by_threshold(dataframes, target_colors, label_mapping, figsize=(10, 6), save_path=None):
    """Median fid_plus vs threshold, one line per target_name (expects load_raw_data_all_thres output)."""
    merged = pd.concat(dataframes, ignore_index=True)
    fig, ax = plt.subplots(figsize=figsize)
    for target_name, target_df in merged.groupby('target_name', sort=False):
        medians = target_df.groupby('threshold')['fid_plus'].median()
        label = label_mapping.get(target_name, target_name)
        ax.plot(medians.index, medians.values, marker='o', markersize=7, linewidth=2, label=label,
                color=target_colors.get(label))
    ax.legend()
    ax.set_xlabel('Threshold', fontsize=7)
    ax.set_ylabel('Median Fid+', fontsize=7)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    return fig, ax


def plot_fidelity_by_threshold_per_target(dataframes, label_mapping, figsize=(12, 8), save_path=None):
    """One panel per target_name with median fid_plus vs threshold per model."""
    merged = pd.concat(dataframes, ignore_index=True)
    target_names = merged['target_name'].unique()
    fig, axes = plt.subplots(1, len(target_names), figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, target_name in zip(axes, target_names):
        for model, model_df in merged[merged['target_name'] == target_name].groupby('model'):
            medians = model_df.groupby('threshold')['fid_plus'].median()
            ax.plot(medians.index, medians.values, marker='o', markersize=5, linewidth=2, label=model,
                    color=MODEL_COLORS.get(model))
        ax.set_title(label_mapping.get(target_name, target_name), fontsize=10)
        ax.set_xlabel('Threshold', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Median Fid+', fontsize=8)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)), fontsize=8, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.show()
    return fig, axes


def _iter_sign_effect_groups(target_dataframes, by_model, by_method):
    group_cols = [col for col, on in (('model', by_model), ('explanation_method', by_method)) if on]
    for name, df in target_dataframes.items():
        if by_method:
            df = map_explanation_method_names(df)
        for key, group in (df.groupby(group_cols) if group_cols else [((), df)]):
            key = key if isinstance(key, tuple) else (key,)
            yield {'dataset': name, **dict(zip(group_cols, key))}, group, _sign_effect_mask(group)


def count_positive_fid_negative_random(target_dataframes, by_model=False, by_method=False):
    """Rows where fid_plus and fid_plus_random have opposite signs, per dataset (and model / method)."""
    rows = [{**meta, 'n_cases': mask.sum(), 'n_total': len(group), 'pct_cases': 100 * mask.sum() / len(group)}
            for meta, group, mask in _iter_sign_effect_groups(target_dataframes, by_model, by_method)]
    return pd.DataFrame(rows).round(2)


def analyze_std_by_sign_effect(target_dataframes, value_col='normalized_fid', by_model=False, by_method=False):
    """n/mean/std of value_col on all rows vs. with the opposite-sign (sign effect) rows removed."""
    rows = []
    for meta, group, mask in _iter_sign_effect_groups(target_dataframes, by_model, by_method):
        all_vals, excl_vals = group[value_col], group.loc[~mask, value_col]
        rows.append({**meta, 'n_all': len(all_vals), 'mean_all': all_vals.mean(), 'std_all': all_vals.std(),
                     'n_excluding': len(excl_vals), 'mean_excluding': excl_vals.mean(),
                     'std_excluding': excl_vals.std(),
                     'std_reduction_pct': 100 * (all_vals.std() - excl_vals.std()) / all_vals.std()})
    return pd.DataFrame(rows).round(4)


def calculate_ranking_excluding_sign_effect(target_dataframes, group_col, value_col='normalized_fid'):
    """Rank group_col by median value_col on all data and with the sign-effect rows excluded."""
    combined = map_explanation_method_names(
        pd.concat([df.assign(target=name) for name, df in target_dataframes.items()], ignore_index=True))
    mask = _sign_effect_mask(combined)
    print(f'Excluded {mask.sum():,} of {len(combined):,} rows ({100 * mask.sum() / len(combined):.2f}%) '
          f'with the sign effect (fid_plus and fid_plus_random have opposite signs).')

    def rank(df, label):
        summary = (df.groupby(group_col)[value_col].agg(n='count', mean='mean', median='median', std='std')
                   .round(3).sort_values('median', ascending=False).reset_index())
        summary.insert(0, 'rank', range(1, len(summary) + 1))
        summary.insert(0, 'data', label)
        return summary

    return pd.concat([rank(combined[~mask], 'excluding sign effect'), rank(combined, 'all data')], ignore_index=True)


def calculate_per_task_rank_panel(target_dataframes, group_col, value_col='normalized_fid'):
    """Per-task ranks of group_col by median value_col, on all data and without the sign-effect rows."""
    columns = {}
    for name, df in target_dataframes.items():
        df = map_explanation_method_names(df)
        for label, data in (('all', df), ('excluding', df[~_sign_effect_mask(df)])):
            columns[(name, label)] = data.groupby(group_col)[value_col].median().rank(method='first', ascending=False)
    panel = pd.DataFrame(columns)
    panel.columns = pd.MultiIndex.from_tuples(panel.columns, names=['task', 'data'])
    panel.index.name = group_col
    return panel.astype('Int64')


def fit_fid_ols(target_dataframes, model_order, method_order, target_order):
    """OLS normalized_fid ~ model * explanation_method + target_name with smiles-clustered SEs; the first entry
    of each order is the baseline and rows outside the orders are dropped."""
    df = pd.concat([df for df in target_dataframes.values() if not df.empty], ignore_index=True)
    df = df[df['model'].isin(model_order) & df['explanation_method'].isin(method_order)].copy()
    for col, order in (('model', model_order), ('explanation_method', method_order), ('target_name', target_order)):
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    print(f'{len(df):,} samples')
    return smf.ols('normalized_fid ~ model * explanation_method + target_name', data=df).fit(
        cov_type='cluster', cov_kwds={'groups': df['smiles']})


def compute_effect_sizes(ols_model, typ=2):
    anova = anova_lm(ols_model, typ=typ)
    anova['eta_sq'] = anova['sum_sq'] / (anova['sum_sq'].sum() + ols_model.ssr)
    anova['eta_sq_partial'] = anova['sum_sq'] / (anova['sum_sq'] + ols_model.ssr)
    return anova


def kruskal_by(target_dataframes, group_cols, value_col='normalized_fid'):
    """Kruskal-Wallis test of value_col across the groups formed by group_cols ('target' = dataset label)."""
    df = pd.concat([df.assign(target=name) for name, df in target_dataframes.items()], ignore_index=True)
    df = df[np.isfinite(df[value_col])]
    grouped = df.groupby(df[group_cols].astype(str).agg('_'.join, axis=1))[value_col]
    groups = [vals.values for _, vals in grouped]
    h, p = kruskal(*groups)
    k, n = len(groups), len(df)
    return pd.DataFrame({'test_name': [f"Kruskal–Wallis (by {' x '.join(group_cols)})"], 'H_statistic': [h],
                         'p_value': [p], 'n_groups': [k], 'N': [n], 'eta_squared': [(h - k + 1) / (n - k)],
                         'groups': [list(grouped.groups.keys())]})
