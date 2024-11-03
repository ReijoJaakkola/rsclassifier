import numpy as np
import pandas as pd
from discretization.information_theory import information
from typing import Tuple

def calculate_midpoints(numbers : np.ndarray) -> np.ndarray:
    """
    Calculate midpoints between sorted numbers.
    
    Args:
        numbers (numpy.ndarray): Array of numbers.

    Returns:
        numpy.ndarray: Array of midpoints.
    """
    sorted_numbers = np.sort(numbers)
    return (sorted_numbers[:-1] + sorted_numbers[1:]) / 2

def minimum_information_gain(num_rows : int, entropy : float, entropy1 : float, entropy2 : float, unique_targets : int, unique_targets1 : int, unique_targets2 : int) -> float:
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

def split_data_by_pivot(dataframe : pd.DataFrame, feature : str, pivot : float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into two subsets based on a pivot value for the feature.
    
    Args:
        dataframe (pandas.DataFrame): The input dataframe.
        feature (str): The feature to split on.
        pivot (float): The pivot value to split the feature.

    Returns:
        tuple: Two subsets of the dataframe split by the pivot.
    """
    df_greater = dataframe[dataframe[feature] > pivot]
    df_lesser_equal = dataframe[dataframe[feature] <= pivot]
    return df_greater, df_lesser_equal

def find_best_pivot(dataframe : pd.DataFrame, feature : str, target : str, pivot_candidates : np.ndarray, N : int, information_upper_bound : float) -> Tuple[float,float]:
    """
    Find the best pivot based on the smallest information value.

    Args:
        dataframe (pandas.DataFrame): The input dataframe.
        feature (str): The feature to split on.
        target (str): The target variable.
        pivot_candidates (numpy.ndarray): Array of pivot candidates.
        N (int): Number of rows in the dataframe.
        information_upper_bound (float): Upper bound for information.

    Returns:
        tuple: The best pivot and its corresponding smallest information value.
    """
    best_pivot = None
    smallest_information_value = information_upper_bound

    for pivot in pivot_candidates:
        z1, z2 = split_data_by_pivot(dataframe, feature, pivot)
        n1, n2 = len(z1), len(z2)

        if n1 == 0 or n2 == 0:
            continue  # Skip invalid splits

        information_value = (n1 / N) * information(z1[target]) + (n2 / N) * information(z2[target])
        if information_value < smallest_information_value:
            best_pivot = pivot
            smallest_information_value = information_value

    return best_pivot, smallest_information_value

def find_pivots(x : pd.Series, y : pd.Series) -> list:
    """
    Find optimal pivot points for splitting data based on information gain.

    Args:
        x (pandas.Series): Feature values for the pivot search.
        y (pandas.Series): Target variable values corresponding to feature `x`.

    Returns:
        list: List of pivot points that yield significant information gain.
    """
    z = pd.concat([x,y], axis = 1)
    feature = z.columns[0]
    target = z.columns[1]
    information_upper_bound = np.log2(len(z[target].unique())) + 1
    pivots = []
    stack = [z]

    while stack:
        z = stack.pop()
        num_rows = len(z)
        unique_values = z[feature].unique()

        if len(unique_values) <= 1:
            continue  # Skip if there are no pivot candidates

        pivot_candidates = calculate_midpoints(unique_values)
        best_pivot, smallest_information_value = find_best_pivot(z, feature, target, pivot_candidates, num_rows, information_upper_bound)

        if best_pivot is None:
            continue

        z1, z2 = split_data_by_pivot(z, feature, best_pivot)

        # Calculate information gain
        E = information(z[target])
        E1 = information(z1[target])
        E2 = information(z2[target])
        k = len(z[target].unique())
        k1 = len(z1[target].unique())
        k2 = len(z2[target].unique())

        min_inf_gain = minimum_information_gain(num_rows, E, E1, E2, k, k1, k2)

        # If significant information gain is found, store pivot and continue splitting
        if (E - min_inf_gain) > smallest_information_value:
            pivots.append(best_pivot)
            stack.extend([z1, z2])

    return pivots