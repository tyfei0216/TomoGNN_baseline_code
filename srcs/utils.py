import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def match_and_find_closest(pred_centers, label_centers):
    """
    Matches predicted centers to label centers using bipartite matching (Hungarian algorithm).
    Also finds the closest prediction for each label.

    Args:
        pred_centers: np.ndarray of shape (N_pred, 3) — predicted (z, y, x) coordinates
        label_centers: np.ndarray of shape (N_label, 3) — ground truth (z, y, x) coordinates

    Returns:
        match_pairs: list of (label_index, pred_index)
        match_distances: np.ndarray of distances for the matched pairs
        closest_preds: np.ndarray of shape (N_label, 2), where each row = (closest_pred_index, distance)
    """
    # Compute distance matrix between labels and predictions
    dist_matrix = cdist(label_centers, pred_centers, metric="euclidean")

    # --- Bipartite matching (Hungarian algorithm) ---
    label_idx, pred_idx = linear_sum_assignment(dist_matrix)
    match_pairs = list(zip(label_idx, pred_idx))
    match_distances = dist_matrix[label_idx, pred_idx]

    # --- Closest prediction for each label ---
    closest_pred_indices = np.argmin(dist_matrix, axis=1)
    closest_pred_distances = dist_matrix[
        np.arange(len(label_centers)), closest_pred_indices
    ]
    closest_preds = np.stack(
        [closest_pred_indices, closest_pred_distances], axis=1
    )  # (N_label, 2)

    return match_pairs, match_distances, closest_preds


def calculate_metrics(match_pairs, match_distances, y_pred_prob, threshold=20):
    """
    Calculate AUROC, precision, and recall for binary classification.

    Args:
        y_true: array-like of shape (n_samples,) — ground truth labels (0 or 1)
        y_pred_prob: array-like of shape (n_samples,) — predicted probabilities
        threshold: float — probability threshold for converting to binary predictions

    Returns:
        auc_score: float — AUROC score
        precision: float — Precision score
        recall: float — Recall score
    """
    matches = np.array(match_pairs)
    find_matches = np.array(match_distances) < threshold
    matched = matches[find_matches]
    score_label = np.zeros(len(y_pred_prob))
    score_label[matched[:, 1]] = 1

    precision, recall, _ = precision_recall_curve(score_label.astype(int), y_pred_prob)
    aupr = auc(recall, precision)

    aupr2 = average_precision_score(score_label.astype(int), y_pred_prob)

    return {
        "auroc": roc_auc_score(score_label.astype(int), y_pred_prob),
        "aupr": aupr,
        "aupr2": aupr2,
        "cnts": sum(find_matches),
        # "precision": precision_score(score_label.astype(int), y_pred_prob > 0.5),
        # "recall": recall_score(score_label.astype(int), y_pred_prob > 0.5),
    }


import mrcfile


def readTomogram(filename):
    with mrcfile.open(filename, permissive=True) as m:
        return m.data
