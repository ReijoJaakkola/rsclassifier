import numpy as np
import pandas as pd
from discretization.information_theory import information
from typing import Tuple, List, Optional

FEATURE = 'feature'
TARGET = 'target'

def _minimum_information_gain(num_rows: int, entropy: float, entropy1: float, entropy2: float, unique_targets: int, unique_targets1: int, unique_targets2: int) -> float:
    """
    Calculate the minimum information gain.
    
    Args:
        num_rows (int): Number of rows in the dataset.
        entropy (float): Entropy of the target variable.
        entropy1 (float): Entropy of the first split.
        entropy2 (float): Entropy of the second split.
        unique_targets (int): Number of unique target values.
        unique_targets1 (int): Unique target values in the first split.
        unique_targets2 (int): Unique target values in the second split.
        
    Returns:
        float: Minimum information gain.
    """
    return (np.log2(num_rows - 1) / num_rows) + ((np.log2(3 ** unique_targets - 2) - unique_targets * entropy 
             + unique_targets1 * entropy1 + unique_targets2 * entropy2) / num_rows)

def _split_data_by_pivot(z: np.ndarray, pivot: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split dataframe into two subsets based on a pivot value for the feature.
    
    Args:
        z (pandas.DataFrame): The input dataframe.
        pivot (float): The pivot value to split the feature.

    Returns:
        tuple: Two subsets of the dataframe split by the pivot.
    """
    mask = z[:, 0] > pivot
    return z[mask], z[~mask]

def _find_best_pivot(z: np.ndarray, information_upper_bound: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Find the best pivot based on the smallest information value.

    Args:
        z (pandas.DataFrame): The input dataframe.
        information_upper_bound (float): Upper bound for information.

    Returns:
        tuple: The best pivot and its corresponding smallest information value.
    """
    unique_values = np.unique(z[:, 0])
    if len(unique_values) <= 1:
        return None, None
    
    unique_values.sort()
    pivot_candidates = (unique_values[:-1] + unique_values[1:]) / 2

    best_pivot = None
    smallest_information_value = information_upper_bound
    N = len(z)

    for pivot in pivot_candidates:
        z1, z2 = _split_data_by_pivot(z, pivot)
        n1, n2 = len(z1), len(z2)

        if n1 == 0 or n2 == 0:
            continue

        information_value = (n1 / N) * information(z1[:, 1]) + (n2 / N) * information(z2[:, 1])
        
        if information_value < smallest_information_value:
            best_pivot = pivot
            smallest_information_value = information_value

    return best_pivot, smallest_information_value

def find_pivots(x: pd.Series, y: pd.Series) -> List[float]:
    """
    Find optimal pivot points for splitting data based on information gain.

    Args:
        x (pandas.Series): Feature values for the pivot search.
        y (pandas.Series): Target variable values corresponding to feature `x`.

    Returns:
        list: List of pivot points that yield significant information gain.
    """
    x_np, y_np = x.to_numpy(copy = True), y.to_numpy(copy = True)
    z = np.column_stack((x_np, y_np))
    z = z[np.argsort(z[:, 0])]
    
    information_upper_bound = np.log2(len(np.unique(y_np))) + 1
    pivots = []
    stack = [z]
    
    while stack:
        current_subset = stack.pop()
        if len(current_subset) <= 1:
            continue

        best_pivot, smallest_information_value = _find_best_pivot(current_subset, information_upper_bound)
        
        if best_pivot is None:
            continue
        
        z1, z2 = _split_data_by_pivot(current_subset, best_pivot)
        
        w, v, u = current_subset[:, 1], z1[:, 1], z2[:, 1]
        E, E1, E2 = information(w), information(v), information(u)
        k, k1, k2 = len(np.unique(w)), len(np.unique(v)), len(np.unique(u))

        min_inf_gain = _minimum_information_gain(len(current_subset), E, E1, E2, k, k1, k2)

        if (E - min_inf_gain) > smallest_information_value:
            pivots.append(best_pivot)
            stack.append(z1)
            stack.append(z2)
    
    return sorted(pivots)