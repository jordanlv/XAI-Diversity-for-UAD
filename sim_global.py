import numpy as np
from scipy.interpolate import interp1d
from itertools import combinations


def mae_sim(pdp_i, pdp_j, sub):
    sims = []
    for feat in pdp_i.keys():
        curve_i = np.array(pdp_i[feat][sub])
        curve_j = np.array(pdp_j[feat][sub])

        # Center curves to isolate the marginal effect from global model bias differences
        curve_i -= np.mean(curve_i)
        curve_j -= np.mean(curve_j)

        dist = np.mean(np.abs(curve_i - curve_j))

        # Convert unbounded MAE distance to a [0, 1] similarity score
        sims.append(1 / (1 + dist))

    return np.nanmean(sims)


def pdp_corr(pdp_i, pdp_j, sub) -> float:
    corrs = []
    for feat in pdp_i.keys():
        tmp_pdp_i = pdp_i[feat][sub]
        tmp_pdp_j = pdp_j[feat][sub]

        if len(tmp_pdp_i.shape) >= 2:
            tmp_pdp_i = tmp_pdp_i.flatten()

        if len(tmp_pdp_j.shape) >= 2:
            tmp_pdp_j = tmp_pdp_j.flatten()

        corrs.append(np.corrcoef(tmp_pdp_i.flatten(), tmp_pdp_j.flatten())[0, 1])
    return np.nanmean(corrs)


def compute_similarity_pearson(results_dict, xai, sub):
    return compute_pairwise_metric(results_dict, xai, sub, pdp_corr)


def compute_mae_sim(results_dict, xai, sub):
    return compute_pairwise_metric(results_dict, xai, sub, mae_sim)


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
