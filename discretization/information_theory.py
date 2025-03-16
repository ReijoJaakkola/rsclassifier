import numpy as np

def entropy_logarithm(p : float) -> float:
    """
    Compute the logarithmic component of the entropy for a given probability value.

    Args:
        p (float): The probability value for which to compute the logarithmic term.

    Returns:
        float: The computed entropy logarithmic term. Returns 0 if `p = 0`.
    """
    return (-1) * p * np.log2(p) if p != 0 else 0

def entropy(p : list) -> float:
    """
    Calculate the entropy for a distribution of probabilities.

    Args:
        p list: A probability distribution.

    Returns:
        float: The computed entropy of the distribution.
    """
    return np.sum([entropy_logarithm(prob) for prob in p])

def information(y : np.ndarray) -> float:
    """
    Calculates the information content of the target variable `y`.

    Args:
        y (np.ndarray): A Pandas Series representing the target variable whose entropy is to be calculated.

    Returns:
        float: The computed entropy (information content) for the target variable `y`.
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities)