import pandas as pd
from discretization.entropy_based_discretization import find_pivots
from tqdm import tqdm

class Booleanizer:
    def __init__(self):
        self.classes_for_cat_features = {}
        self.pivots_for_num_features = {}

    def store_classes_for_cat_features(
            self,
            X : pd.DataFrame, 
            categorical_features : list
        ) -> None:
        """
        Learns the unique categories (values) for each categorical feature in the dataset. These are stored for Booleanization.

        Args:
            X (pandas.DataFrame): The feature data.
            categorical_features (list): List of categorical features.
        """
        for feature in categorical_features:
            self.classes_for_cat_features[feature] = list(X[feature].unique())

    def store_pivots_for_num_features(
            self,
            X : pd.DataFrame,
            y : pd.Series,
            numerical_features : list,
            silent : bool = False
        ) -> None:
        """
        Learns pivot points for numerical features using entropy-based discretization. These are stored for Booleanization.

        Args:
            X (pandas.DataFrame): The feature data.
            y (pandas.Series): The target labels.
            numerical_features (list): List of numerical features.
            silent (bool): Whether to suppress output.
        """
        for feature in tqdm(numerical_features, total=len(numerical_features), desc='Learning pivots for numerical features...', disable = silent):
            # Find pivot points for discretization.
            self.pivots_for_num_features[feature] = find_pivots(X[feature], y)

    def booleanize_dataframe(
            self,
            X : pd.DataFrame
        ) -> pd.DataFrame:
        """
        Converts the input dataset into a booleanized format.
        - Categorical features are one-hot encoded (for each unique value in the feature).
        - Numerical features are transformed into boolean columns based on whether the feature value is greater than a learned pivot.

        Args:
            X (pandas.DataFrame): The feature data.

        Returns:
            pandas.DataFrame: The Booleanized dataframe.
        """
        bool_X = X.copy()

        categorical_features = list(self.classes_for_cat_features.keys())
        for feature in self.classes_for_cat_features.keys():
            new_columns = {}
            for value in self.classes_for_cat_features[feature]:
                # Create a new column for each value (one-hot encoding style).
                new_columns[feature + ' = ' + str(value)] = (bool_X[feature] == value)
            # Concatenate the new Boolean columns with the original data.
            bool_X = pd.concat([bool_X, pd.DataFrame(new_columns)], axis=1)
        # Drop original categorical columns.
        bool_X.drop(columns = categorical_features, inplace=True)

        numerical_features = list(self.pivots_for_num_features.keys())
        for feature in numerical_features:
            new_columns = {}
            for pivot in self.pivots_for_num_features[feature]:
                # Create a Boolean column for values greater than the pivot.
                new_columns[f'{feature} > {pivot:.2f}'] = bool_X[feature] > pivot
            # Concatenate new columns with the data.
            bool_X = pd.concat([bool_X, pd.DataFrame(new_columns)], axis = 1)
        # Drop original numerical columns.
        bool_X.drop(columns=numerical_features, inplace=True)

        return bool_X