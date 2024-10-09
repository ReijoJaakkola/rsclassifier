import numpy as np

def entropy_logarithm(p):
    """
    Compute the logarithmic component of the entropy for a given probability value.

    Args:
        p (float): The probability value for which to compute the logarithmic term.

    Returns:
        float: The computed entropy logarithmic term. Returns 0 if `p = 0`.
    """
    return (-1) * p * np.log2(p) if p != 0 else 0

def entropy(p):
    """
    Calculate the entropy for a distribution of probabilities.

    Args:
        p (array-like): A list, NumPy array, or iterable of probability values (each between 0 and 1, inclusive).

    Returns:
        float: The computed entropy of the distribution.
    """
    return np.sum([entropy_logarithm(prob) for prob in p])

def information(y):
    """
    Calculates the information content of the target variable `y`.

    Args:
        y (pandas.Series): A Pandas Series representing the target variable whose entropy is to be calculated. Each unique 
                           value in the series corresponds to a class label, and the function calculates the proportion of 
                           each class to compute entropy.

    Returns:
        float: The computed entropy (information content) for the target variable `y`.
    """
    class_counts = y.value_counts(normalize=True)
    return entropy(class_counts)