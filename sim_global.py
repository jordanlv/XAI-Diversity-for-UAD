import numpy as np
from scipy.interpolate import interp1d
from itertools import combinations


def _apply_feature_wise_metric(pdp_i, pdp_j, sub, metric_fn):
    """Helper to compute a metric over all features."""
    results = []
    # Ensure we are comparing the same set of features
    common_features = pdp_i.keys() & pdp_j.keys()
    if not common_features:
        return np.nan

    for feat in common_features:
        curve_i = np.asarray(pdp_i[feat][sub]).flatten()
        curve_j = np.asarray(pdp_j[feat][sub]).flatten()
        results.append(metric_fn(curve_i, curve_j))
    return np.nanmean(results)


def _mae_dist_on_feature(curve_i, curve_j):
    """Computes MAE-based distance for a single feature's curves."""
    # Center curves to isolate the marginal effect
    curve_i -= np.mean(curve_i)
    curve_j -= np.mean(curve_j)
    return np.mean(np.abs(curve_i - curve_j))


def _pdp_corr_on_feature(curve_i, curve_j):
    """Computes Pearson correlation for a single feature's curves."""
    # Handle cases with no variance
    std_i = np.std(curve_i)
    std_j = np.std(curve_j)
    if std_i < 1e-12 and std_j < 1e-12:
        return 1.0
    if std_i < 1e-12 or std_j < 1e-12:
        return 0.0
    return np.corrcoef(curve_i, curve_j)[0, 1]


def mae_dist(pdp_i, pdp_j, sub):
    """Computes Mean Absolute Error distance between curves."""
    return _apply_feature_wise_metric(pdp_i, pdp_j, sub, _mae_dist_on_feature)


def pdp_corr(pdp_i, pdp_j, sub) -> float:
    """Computes Pearson correlation between curves."""
    return _apply_feature_wise_metric(pdp_i, pdp_j, sub, _pdp_corr_on_feature)


def compute_similarity_pearson(results_dict, xai, sub):
    matrix, model_names = compute_pairwise_metric(results_dict, xai, sub, pdp_corr)
    # Scale Pearson correlation from [-1, 1] to [0, 1]
    return (matrix + 1) / 2, model_names


def compute_mae_sim(results_dict, xai, sub):
    """Computes similarity using exponential decay (RBF kernel)."""
    matrix, model_names = compute_pairwise_metric(results_dict, xai, sub, mae_dist)

    # Reset the diagonal to 0 because compute_pairwise_metric initializes it to 1.0
    np.fill_diagonal(matrix, 0.0)

    sim_matrix = np.exp(-1.0 * matrix)

    return sim_matrix, model_names


def compute_pairwise_metric(results_dict, key, sub, metric_fn):
    """
    Generic function to compute pairwise similarity/distance matrices
    across models for a given metric and key (e.g., 'shapvalues', 'scores', 'predictions').
    """
    model_names = sorted(results_dict.keys() - {"ground_truth"})
    n_models = len(model_names)
    n_fold = len(results_dict[model_names[0]])

    matrix = np.eye(n_models, dtype=float)

    for i, j in combinations(range(n_models), 2):
        scores = []
        for f in range(n_fold):
            scores.append(
                metric_fn(
                    results_dict[model_names[i]][f][key],
                    results_dict[model_names[j]][f][key],
                    sub,
                )
            )

        matrix[i, j] = matrix[j, i] = np.nanmean(scores)

    return matrix, model_names
