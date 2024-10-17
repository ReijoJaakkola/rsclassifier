import pandas as pd
from sklearn.tree import DecisionTreeClassifier

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