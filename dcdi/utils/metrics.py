import networkx as nx
import numpy as np
from cdt.metrics import SHD_CPDAG, SID, get_CPDAG, retrieve_adjacency_matrix


def edge_errors(pred, target):
    """
    Counts all types of edge errors (false negatives, false positives, reversed edges)

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    fn, fp, rev

    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(
        pred, target.nodes() if isinstance(target, nx.DiGraph) else None
    )

    diff = true_labels - predictions

    rev = (((diff + diff.transpose()) == 0) & (diff != 0)).sum() / 2
    # Each reversed edge necessarily leads to one fp and one fn so we need to subtract those
    fn = (diff == 1).sum() - rev
    fp = (diff == -1).sum() - rev

    return fn, fp, rev


def edge_accurate(pred, target):
    """
    Counts the number of edge in ground truth DAG, true positives and the true
    negatives

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    total_edges, tp, tn

    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(
        pred, target.nodes() if isinstance(target, nx.DiGraph) else None
    )

    total_edges = (true_labels).sum()

    tp = ((predictions == 1) & (predictions == true_labels)).sum()
    tn = ((predictions == 0) & (predictions == true_labels)).sum()

    return total_edges, tp, tn


def shd(pred, target):
    """
    Calculates the structural hamming distance

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix

    Returns:
    --------
    shd

    """
    return sum(edge_errors(pred, target))


def edge_errors_cpdag(pred, target, is_pred_cpdag: bool):
    """
    Counts all types of edge errors in CPDAG (more details in returns).

    Parameters:
    -----------
    pred: nx.DiGraph or ndarray
        The predicted adjacency matrix
    target: nx.DiGraph or ndarray
        The true adjacency matrix
    is_pred_cpdag: bool
        Indicator weather pred should be treated as cpdag

    Returns:
    --------
    fpd (False Positive Directed):
        Number of directed edges that are in pred but not in target
    fpu: (False Positive Undirected):
        Number of undirected edges that are in pred but not in target
    fud (False Undirected that should be Directed):
        Number of undirected edges that are in pred but are directed in target
    fud (False Directed that should be Undirected):
        Number of directed edges that are in pred but are undirected in target
    fnd (False Negative Directed):
        Number of directed edges that are in target but not in pred
    fnu (False Negative Undirected):
        Number of undirected edges that are in target but not in pred
    """
    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(
        pred, target.nodes() if isinstance(target, nx.DiGraph) else None
    )
    if not is_pred_cpdag:
        predictions = get_CPDAG(predictions)

    true_labels = get_CPDAG(true_labels)
    tl = true_labels + true_labels.T
    p = predictions + predictions.T
    fpd = np.sum((tl == 0) & (p == 1))
    fpu = np.sum((tl == 0) & (p == 2))
    fud = np.sum((tl == 1) & (p == 2))
    fdu = np.sum((tl == 2) & (p == 1))
    fnd = np.sum((tl == 1) & (p == 0))
    fnu = np.sum((tl == 2) & (p == 0))
    return fpd, fpu, fud, fdu, fnd, fnu


def compute_structure_metrics(current_adj, gt_adjacency, is_pred_cpdag: bool):
    fpd, fpu, fud, fdu, fnd, fnu = edge_errors_cpdag(
        pred=current_adj, target=gt_adjacency, is_pred_cpdag=is_pred_cpdag
    )
    fn, fp, rev = edge_errors(current_adj, gt_adjacency)
    sid = float(SID(target=gt_adjacency, pred=current_adj))
    shd_metric = float(shd(current_adj, gt_adjacency))
    shd_between_cpdags = float(SHD_CPDAG(target=gt_adjacency, pred=current_adj))
    shd_to_cpdag = float(shd(current_adj, get_CPDAG(gt_adjacency)))
    return {
        "shd": shd_metric,
        "sid": sid,
        "shd_between_cpdags": shd_between_cpdags,
        "shd_to_cpdag": shd_to_cpdag,
        "fn": fn,
        "fp": fp,
        "rev": rev,
        "fpd": fpd,
        "fpu": fpu,
        "fdu": fdu,
        "fud": fud,
        "fnd": fnd,
        "fnu": fnu,
    }
