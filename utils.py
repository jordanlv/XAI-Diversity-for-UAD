import re
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from itertools import combinations
from sklearn.metrics import ndcg_score, jaccard_score, average_precision_score
from typing import Callable, Dict, Tuple, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from collections import defaultdict


def compute_pairwise_metric(
    results_dict: Dict[str, Dict[str, list]],
    key: str,
    metric_fn: Callable[[Any, Any], float],
) -> Tuple[np.ndarray, list]:
    """
    Generic function to compute pairwise similarity/distance matrices
    across models for a given metric and key (e.g., 'shapvalues', 'scores', 'predictions').
    """
    model_names = sorted(results_dict.keys() - {"ground_truth"})
    n_models = len(model_names)
    n_fold = len(results_dict[model_names[0]])

    data = {m: [results_dict[m][f][key] for f in range(n_fold)] for m in model_names}

    matrix = np.eye(n_models, dtype=float)

    for i, j in combinations(range(n_models), 2):
        scores = [
            metric_fn(data[model_names[i]][f], data[model_names[j]][f])
            for f in range(n_fold)
        ]
        matrix[i, j] = matrix[j, i] = np.nanmean(scores)

    return matrix, model_names


# --- Metric-specific wrappers --------------------------------------------------


def pearson_corr_vectorized(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute per-sample Pearson correlation between x and y.
    x, y shape: (n_samples, n_features)
    Returns: array of correlations (n_samples,)
    """
    x_mean = x.mean(axis=1, keepdims=True)
    y_mean = y.mean(axis=1, keepdims=True)
    num = np.sum((x - x_mean) * (y - y_mean), axis=1)
    den = np.sqrt(np.sum((x - x_mean) ** 2, axis=1) * np.sum((y - y_mean) ** 2, axis=1))
    return num / np.maximum(den, 1e-12)


def compute_shap_similarity_pearson(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute pairwise Pearson correlation similarity between SHAP values."""

    return compute_pairwise_metric(results_dict, "shapvalues", pearson_corr_vectorized)


def compute_shap_ndcg_similarity(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute pairwise NDCG similarity between SHAP value magnitudes."""

    def ndcg_metric(model_i_fold, model_j_fold) -> float:
        if len(model_i_fold.shape) == 1:
            model_i_fold = np.array([model_i_fold])
        if len(model_j_fold.shape) == 1:
            model_j_fold = np.array([model_j_fold])

        score_ij = ndcg_score(np.abs(model_i_fold), np.abs(model_j_fold))
        score_ji = ndcg_score(np.abs(model_j_fold), np.abs(model_i_fold))
        return np.mean([score_ij, score_ji])  # type: ignore

    return compute_pairwise_metric(results_dict, "shapvalues", ndcg_metric)


def compute_lime_similarity_pearson(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute pairwise Pearson correlation similarity between LIME values."""

    return compute_pairwise_metric(results_dict, "limevalues", pearson_corr_vectorized)


def compute_lime_ndcg_similarity(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute pairwise NDCG similarity between LIME value magnitudes."""

    def ndcg_metric(model_i_fold, model_j_fold) -> float:
        if len(model_i_fold.shape) == 1:
            model_i_fold = np.array([model_i_fold])
        if len(model_j_fold.shape) == 1:
            model_j_fold = np.array([model_j_fold])

        score_ij = ndcg_score(np.abs(model_i_fold), np.abs(model_j_fold))
        score_ji = ndcg_score(np.abs(model_j_fold), np.abs(model_i_fold))
        return np.mean([score_ij, score_ji])  # type: ignore

    return compute_pairwise_metric(results_dict, "limevalues", ndcg_metric)


def compute_pdp_similarity_pearson(
    results_dict,
):
    """Compute pairwise Pearson correlation similarity between PDP curves."""

    def pdp_corr(pdp_i, pdp_j) -> float:

        corrs = []
        for feat in pdp_i.keys():
            y_i = np.array(pdp_i[feat]["average"]).flatten()
            y_j = np.array(pdp_j[feat]["average"]).flatten()

            std_i = np.std(y_i)
            std_j = np.std(y_j)

            if std_i < 1e-12 and std_j < 1e-12:
                corrs.append(1.0)
            elif std_i < 1e-12 or std_j < 1e-12:
                corrs.append(0.0)
            else:
                corrs.append(np.corrcoef(y_i, y_j)[0, 1])

        return np.nanmean(corrs)

    return compute_pairwise_metric(results_dict, "pdpvalues", pdp_corr)


def compute_score_correlations(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute Pearson correlation between models' anomaly scores."""

    def score_corr(model_i_fold, model_j_fold) -> float:
        return np.corrcoef(model_i_fold, model_j_fold)[0, 1]

    return compute_pairwise_metric(results_dict, "scores", score_corr)


def compute_pred_jaccard(
    results_dict: Dict[str, Dict[str, list]],
) -> Tuple[np.ndarray, list]:
    """Compute pairwise Jaccard index between binary predictions."""

    def jaccard_metric(model_i_fold, model_j_fold) -> float:
        return jaccard_score(
            model_i_fold,
            model_j_fold,
            zero_division=0,
            average="binary",
            pos_label=1,
        )  # type: ignore

    return compute_pairwise_metric(results_dict, "predictions", jaccard_metric)


def compute_aucpr(results_dict):
    model_names = list(set((results_dict.keys())) - set(["ground_truth"]))
    n_models = len(model_names)
    n_fold = len(results_dict[model_names[0]])

    all_mccs = []

    for model in range(n_models):
        mccs = []
        for fold_idx in range(n_fold):
            y_true = results_dict["ground_truth"][fold_idx]
            y_test = results_dict[model_names[model]][fold_idx]["scores"]

            mccs.append(average_precision_score(y_true, y_test))

        all_mccs.append(np.nanmean(np.array(mccs)))

    return all_mccs, model_names


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as mc
import colorsys
from typing import Dict


def adjust_hue(color, amount=0.08):
    # Convertit RGB en HLS pour isoler la teinte (Hue)
    c = mc.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(*c)

    # Décale la teinte et utilise le modulo 1.0 pour rester sur la roue chromatique
    new_h = (h + amount) % 1.0

    return colorsys.hls_to_rgb(new_h, l, s)


def plot_heatmaps(
    matrices: Dict[str, pd.DataFrame],
    figsize=(20, 12),
    colormaps=["OrRd", "Blues", "YlGn", "PuBu"],
    n_colors=6,
    hue_shift=0.08,  # Paramètre pour contrôler la force du changement de couleur
):
    n_row = len(matrices) // 2
    n_col = 2
    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    matrix_items = list(matrices.items())

    for idx, (name, df) in enumerate(matrix_items):
        row = idx // n_col
        col = idx % n_col
        ax = axes[row, col]

        base_cmap = plt.get_cmap(colormaps[row])
        base_colors = base_cmap(np.linspace(0.15, 0.85, n_colors))

        # La colonne 0 garde la couleur originale, la colonne 1 reçoit le décalage de teinte
        if col == 0:
            adjusted_colors = base_colors
        else:
            adjusted_colors = [adjust_hue(c, amount=hue_shift) for c in base_colors]

        discrete_cmap = ListedColormap(adjusted_colors)

        vmin, vmax = np.min(df.values), np.max(df.values)
        bounds = np.linspace(vmin, vmax, n_colors + 1)
        norm = BoundaryNorm(bounds, discrete_cmap.N)

        sns.heatmap(
            df,
            cmap=discrete_cmap,
            annot=True,
            fmt=".0f",
            linewidths=0.25,
            linecolor="black",
            cbar=False,
            annot_kws={"size": 8},
            ax=ax,
            norm=norm,
        )
        ax.set_title(name, fontsize=13)

    plt.tight_layout()
    return fig


def load_nested_results(base_path) -> Dict[str, Any]:
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for dataset in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset)
        if not os.path.isdir(dataset_path) or dataset.startswith("."):
            continue

        for model in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model)
            if not os.path.isdir(model_path) or model.startswith("."):
                continue

            for fold_file in os.listdir(model_path):
                if not fold_file.endswith(".pkl") or fold_file.startswith("."):
                    continue

                try:
                    fold_number = int(fold_file[:-4])
                    with open(os.path.join(model_path, fold_file), "rb") as f:
                        all_results[dataset][model][fold_number] = pickle.load(f)
                except ValueError:
                    continue

    def natural_keys(text):
        return [
            int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(text))
        ]

    def to_regular_dict(d):
        if isinstance(d, dict):
            return {
                k: to_regular_dict(v)
                for k, v in sorted(d.items(), key=lambda x: natural_keys(x[0]))
            }
        return d

    return to_regular_dict(all_results)
