import pandas as pd
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def feature_selection_using_decision_tree(X : pd.DataFrame, y : pd.Series, k : int) -> list:
    """
    Perform feature selection using a Decision Tree classifier.

    This function trains a Decision Tree classifier on the given dataset and ranks the features 
    based on their importance as determined by the classifier. It then returns the top `k` most 
    important features.

    Args:
        X (pd.DataFrame) : The feature data.
        y (pandas.Series): The target labels.
        k (int): The number of top-ranked features to return.

    Returns:
        list: A list of the top `k` most important feature names.
    """
    dt = DecisionTreeClassifier(random_state = 42)
    dt.fit(X, y)
    importances = dt.feature_importances_
    feature_names = X.columns
    feature_importances = {}
    for i in range(len(feature_names)):
        feature_importances[feature_names[i]] = importances[i]
    ranked_features = list((dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)).keys()))
    return ranked_features[:k]

def _generate_subsets(A: list, k: int):
    """
    Generate all non-empty subsets of a list `A` with size at most `k`.

    Args:
        A (list): A list of elements from which subsets will be generated.
        k (int): The maximum size of subsets to generate.

    Returns:
        list: A list containing all non-empty subsets of size at most `k`.
    """
    subsets = []
    
    def _generate_subset(start, current_subset):
        # Add the current subset to the list
        if len(current_subset) > 0:
            subsets.append(current_subset.copy())
        
        # If the current subset length reaches k, stop generating further subsets
        if len(current_subset) == k:
            return
        
        for i in range(start, len(A)):
            current_subset.append(A[i])
            _generate_subset(i + 1, current_subset)  # Move to the next element
            current_subset.pop()  # Backtrack

    _generate_subset(0, [])
    return subsets

def _calculate_ideal_empirical_accuracy(X : pd.DataFrame, y : pd.Series) -> float:
    """
    This function (over)fits a Decision Tree classifier on the provided feature data `X` 
    and target labels `y`, and computes the accuracy of the model on the same data. This
    is the highest achievable accuracy on this data.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target labels.

    Returns:
        float: The highest achievable accuracy on the given data (X,y).
    """
    dt = DecisionTreeClassifier(random_state = 42)
    dt.fit(X, y)
    y_pred = dt.predict(X)
    return accuracy_score(y_pred, y)

def feature_selection_using_brute_force(X : pd.DataFrame, y : pd.Series, k : int, silent : bool) -> list:
    """
    Perform brute-force feature selection by evaluating all subsets of features.

    Args:
        X (pd.DataFrame): The feature data.
        y (pd.Series): The target labels.
        k (int): The maximum number of features to select.
        silent (bool): Whether to suppress output during the process.

    Returns:
        list: A list of the `k` features that yielded the best accuracy.
    """
    best_features = None
    best_accuracy = 0.0
    feature_combinations = _generate_subsets(X.columns, k)
    for features in tqdm(feature_combinations, total = len(feature_combinations), desc='Going through feature combinations...', disable=silent):
        local_X = X[features]
        accuracy = _calculate_ideal_empirical_accuracy(local_X, y)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_features = features
    return best_features