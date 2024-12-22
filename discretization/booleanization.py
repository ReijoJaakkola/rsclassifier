import pandas as pd
from discretization.entropy_based_discretization import find_pivots
from tqdm import tqdm

def booleanize_categorical_features(X : pd.DataFrame, categorical_features : list) -> pd.DataFrame:
    """
    Convert categorical features into Boolean features.

    Args:
        X (pandas.DataFrame): The feature data.
        categorical_features (list): List of categorical features.

    Returns:
        pandas.DataFrame: Data with Booleanized categorical features.
    """
    bool_X = X.copy()
    for feature in categorical_features:
        unique_values = bool_X[feature].unique()
        new_columns = {}
        for value in unique_values:
            # Create a new column for each value (one-hot encoding style).
            new_columns[feature + ' = ' + str(value)] = (bool_X[feature] == value)
        # Concatenate the new Boolean columns with the original data.
        bool_X = pd.concat([bool_X, pd.DataFrame(new_columns)], axis=1)
    # Drop original categorical columns.
    bool_X.drop(columns=categorical_features, inplace=True)
    return bool_X

def booleanize_numerical_features(X : pd.DataFrame, y : pd.Series, numerical_features : list, silent : bool = False) -> pd.DataFrame:
    """
    Discretize numerical features using pivots and convert them into Boolean features.

    Args:
        X (pandas.DataFrame): The feature data.
        y (pandas.Series): The target labels.
        numerical_features (list): List of numerical features.
        silent (bool): Whether to suppress output.

    Returns:
        pandas.DataFrame: Data with Booleanized numerical features.
    """
    bool_X = X.copy()
    for feature in tqdm(numerical_features, total=len(numerical_features), desc='Discretizing numerical features...', disable = silent):
        # Find pivot points for discretization.
        pivots = find_pivots(bool_X[feature], y)
        if len(pivots) == 0:
            # Skip features with no suitable pivots.
            continue
        new_columns = {}
        for pivot in pivots:
            # Create a Boolean column for values greater than the pivot.
            new_columns[f'{feature} > {pivot:.2f}'] = bool_X[feature] > pivot
        # Concatenate new columns with the data.
        bool_X = pd.concat([bool_X, pd.DataFrame(new_columns)], axis=1)
    # Drop original numerical columns.
    bool_X.drop(columns=numerical_features, inplace=True)
    return bool_X