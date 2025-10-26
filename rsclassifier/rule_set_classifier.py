import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from tqdm import tqdm
from typing import Any, Tuple, Optional, List, Union, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from discretization.entropy_based_discretization import find_pivots
from rsclassifier.feature_selection import (
    feature_selection_using_decision_tree, 
    feature_selection_using_brute_force
)
from rsclassifier.quine_mccluskey import minimize_dnf


class RuleSetClassifier(BaseEstimator, ClassifierMixin):
    """
    A rule-based classifier that learns interpretable if-then rules from data.
    
    This classifier converts features to boolean representations, learns empirically
    optimal rules and simplifies them using various pruning techniques.
    
    The classifier supports serialization via pickle (joblib), making it compatible
    with standard sklearn model persistence workflows.
    
    Parameters
    ----------
    num_prop : int
        The number of features (properties) to use in rule generation.
    
    fs_algorithm : {'dt', 'brute'}, default='dt'
        Algorithm used to select which Boolean features to use.
        - 'dt': Decision tree-based feature selection
        - 'brute': Brute force feature selection
    
    growth_size : float, default=1.0
        Proportion of dataset to use for rule growth (training).
        Should be in range (0, 1]. If 1.0, no pruning split is created.
        The remaining data is used for rule pruning.
    
    random_state : int, default=42
        Controls shuffling for train/prune split.
    
    default_prediction : Any, optional
        Default prediction when no rule matches. If None, uses mode of training data.
    
    boolean_features : list, default=[]
        Names of boolean features in the input data.
    
    categorical_features : list, default=[]
        Names of categorical features in the input data.
    
    numerical_features : list, default=[]
        Names of numerical features in the input data.
    
    silent : bool, default=False
        If True, suppresses progress output during training.
    
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        The classes seen during fit.
    
    n_features_in_ : int
        Number of features seen during fit.
    
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.
    
    semantics_ : dict
        Maps propositional symbols to feature metadata.
    
    rules_ : list
        Learned rules as (output, terms) pairs.
    
    default_prediction_ : Any
        The default prediction value.
    
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> import joblib
    >>> from rsclassifier import RuleSetClassifier
    >>> X, y = load_iris(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>> clf = RuleSetClassifier(num_prop=3, numerical_features=X.columns.tolist())
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    >>> print(clf)  # Display learned rules
    >>> 
    >>> # Save the model
    >>> joblib.dump(clf, 'model.pkl')
    >>> # Load the model
    >>> clf_loaded = joblib.load('model.pkl')
    >>> y_pred_loaded = clf_loaded.predict(X_test)
    """
    
    def __init__(
        self,
        num_prop: int,
        fs_algorithm: str = 'dt',
        growth_size: float = 1.0,
        random_state: int = 42,
        default_prediction: Optional[Any] = None,
        boolean_features: List[str] = None,
        categorical_features: List[str] = None,
        numerical_features: List[str] = None,
        silent: bool = False
    ):
        self.num_prop = num_prop
        self.fs_algorithm = fs_algorithm
        self.growth_size = growth_size
        self.random_state = random_state
        self.default_prediction = default_prediction
        self.boolean_features = boolean_features or []
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.silent = silent

    def _validate_hyperparameters(self):
        """Validate hyperparameters before fitting."""
        if self.growth_size <= 0.0 or self.growth_size > 1.0:
            raise ValueError('growth_size must be in range (0, 1]')
        
        if self.fs_algorithm not in ['dt', 'brute']:
            raise ValueError("fs_algorithm must be 'dt' or 'brute'")
        
        if self.num_prop <= 0:
            raise ValueError('num_prop must be positive')

    def _booleanize_categorical_features(
        self, 
        X: pd.DataFrame, 
        categorical_features: List[str]
    ) -> pd.DataFrame:
        """Convert categorical features into Boolean features via one-hot encoding."""
        local_X = X.copy()
        for feature in categorical_features:
            unique_values = local_X[feature].unique()
            new_columns = {}
            for value in unique_values:
                col_name = f'{feature} = {value}'
                new_columns[col_name] = (local_X[feature] == value)
                self.semantics_[col_name] = ['categorical', feature, value]
            local_X = pd.concat([local_X, pd.DataFrame(new_columns)], axis=1)
        local_X.drop(columns=categorical_features, inplace=True)
        return local_X

    def _booleanize_numerical_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        numerical_features: List[str]
    ) -> pd.DataFrame:
        """Discretize numerical features using entropy-based pivots."""
        local_X = X.copy()
        desc = 'Discretizing numerical features...'
        for feature in tqdm(numerical_features, desc=desc, disable=self.silent):
            pivots = find_pivots(local_X[feature], y)
            if len(pivots) == 0:
                continue
            new_columns = {}
            for pivot in pivots:
                col_name = f'{feature} > {pivot:.2f}'
                new_columns[col_name] = local_X[feature] > pivot
                self.semantics_[col_name] = ['numerical', feature, pivot]
            local_X = pd.concat([local_X, pd.DataFrame(new_columns)], axis=1)
        local_X.drop(columns=numerical_features, inplace=True)
        return local_X

    def _preprocess_data(
        self, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> pd.DataFrame:
        """Convert all features to Boolean representation."""
        bool_X = X.copy()
        
        # Handle boolean features
        if len(self.boolean_features) > 0:
            for feature in self.boolean_features:
                bool_X[feature] = X[feature].astype(bool)
                self.semantics_[feature] = ['boolean', feature]
        
        # Handle categorical features
        if len(self.categorical_features) > 0:
            bool_X = self._booleanize_categorical_features(
                bool_X, self.categorical_features
            )
        
        # Handle numerical features
        if len(self.numerical_features) > 0:
            bool_X = self._booleanize_numerical_features(
                bool_X, y, self.numerical_features
            )

        # Store feature indices for numpy operations
        for key, value in self.semantics_.items():
            feature_idx = X.columns.get_loc(value[1])
            self.semantics_[key] = value + [feature_idx]

        if not self.silent:
            print(f'Total Boolean features: {len(bool_X.columns)}')

        return bool_X

    def _form_rule_set(
        self, 
        features: List[str], 
        default_prediction: Optional[Any]
    ) -> None:
        """Form initial rule set using covering algorithm."""
        self.rules_ = []
        
        # Convert to numpy for efficiency
        local_X_np = self.X_grow_[features].to_numpy()
        y_np = self.y_grow_.to_numpy()
        
        # Find unique feature combinations
        unique_types, inverse_indices = np.unique(
            local_X_np, axis=0, return_inverse=True
        )
        
        # Count occurrences of each (type, y) pair
        unique_y = np.unique(y_np)
        type_scores = np.zeros((len(unique_types), len(unique_y)))

        desc = 'Calculating probabilities...'
        for i in tqdm(range(len(y_np)), desc=desc, disable=self.silent):
            y_idx = np.where(unique_y == y_np[i])[0][0]
            type_scores[inverse_indices[i], y_idx] += 1

        # Determine default prediction
        if default_prediction is None:
            self.default_prediction_ = self.y_grow_.mode()[0]
        else:
            self.default_prediction_ = default_prediction

        # Generate rules
        rules = {y: [] for y in unique_y if y != self.default_prediction_}

        desc = 'Forming the classifier...'
        for i in tqdm(range(len(unique_types)), desc=desc, disable=self.silent):
            best_y = unique_y[np.argmax(type_scores[i])]
            if best_y != self.default_prediction_:
                rules[best_y].append(unique_types[i])

        # Convert to final format
        self.rules_ = [
            (key, [list(zip(rule, features)) for rule in value]) 
            for key, value in rules.items() if value
        ]

    def _prune_terms_using_domain_knowledge(
        self, 
        terms: List[List]
    ) -> List[List]:
        """Simplify terms using domain knowledge about feature types."""
        simplified_terms = []
        for term in terms:
            simplified_term = []
            positive_categories = []
            upper_bounds = {}
            lower_bounds = {}
            
            for literal in term:
                meaning = self.semantics_[literal[1]]
                
                if meaning[0] == 'categorical':
                    if literal[0] == 0:
                        simplified_term.append(literal)
                    if literal[0] == 1 and meaning[1] not in positive_categories:
                        positive_categories.append(meaning[1])
                        simplified_term.append(literal)
                
                elif meaning[0] == 'numerical':
                    if literal[0] == 0:
                        if meaning[1] not in upper_bounds or meaning[2] <= upper_bounds[meaning[1]]:
                            upper_bounds[meaning[1]] = meaning[2]
                    if literal[0] == 1:
                        if meaning[1] not in lower_bounds or meaning[2] > lower_bounds[meaning[1]]:
                            lower_bounds[meaning[1]] = meaning[2]
                
                else:  # Boolean
                    simplified_term.append(literal)
            
            # Add bounds
            for feature, bound in upper_bounds.items():
                simplified_term.append([0, f'{feature} > {bound:.2f}'])
            for feature, bound in lower_bounds.items():
                simplified_term.append([1, f'{feature} > {bound:.2f}'])
            
            simplified_terms.append(simplified_term)
        
        return simplified_terms
    
    def _evaluate_term_accuracy(
        self, 
        term: List, 
        prediction: Any, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> float:
        """Calculate accuracy of a term on given data."""
        term_mask = np.ones(len(X), dtype=bool)
        for literal in term:
            term_mask &= (X[literal[1]] == literal[0])
        
        covered = term_mask.sum()
        if covered == 0:
            return 0.0
        
        correct = np.sum(term_mask & (y == prediction))
        return correct / covered
    
    def _prune_terms_useless_literals(
        self, 
        terms: List[List], 
        prediction: Any
    ) -> List[List]:
        """Remove literals that don't improve accuracy on training data."""
        pruned_terms = []
        desc = f'Pruning terms for class {prediction}...'
        
        for term in tqdm(terms, desc=desc, disable=self.silent):
            local_term = term.copy()
            while True:
                old_score = self._evaluate_term_accuracy(
                    local_term, prediction, self.X_grow_, self.y_grow_
                )
                best_term = local_term
                
                for i in range(len(local_term)):
                    reduced_term = local_term[:i] + local_term[i + 1:]
                    new_score = self._evaluate_term_accuracy(
                        reduced_term, prediction, self.X_grow_, self.y_grow_
                    )
                    if old_score == new_score:
                        best_term = reduced_term
                        break
                
                if best_term == local_term:
                    break
                local_term = best_term
            
            pruned_terms.append(local_term)
        
        return pruned_terms
    
    def _prune_terms_cross_validation(
        self, 
        terms: List[List], 
        prediction: Any
    ) -> List[List]:
        """Prune terms using validation set."""
        pruned_terms = []
        desc = f'Cross-validation pruning for class {prediction}...'
        
        for term in tqdm(terms, desc=desc, disable=self.silent):
            local_term = term.copy()
            while True:
                best_score = self._evaluate_term_accuracy(
                    local_term, prediction, self.X_prune_, self.y_prune_
                )
                best_term = local_term
                
                for i in range(len(local_term)):
                    reduced_term = local_term[:i] + local_term[i + 1:]
                    score = self._evaluate_term_accuracy(
                        reduced_term, prediction, self.X_prune_, self.y_prune_
                    )
                    if score > best_score:
                        best_score = score
                        best_term = reduced_term
                
                if best_term == local_term:
                    break
                local_term = best_term
            
            pruned_terms.append(local_term)
        
        return pruned_terms

    def _entails(self, term1: List, term2: List) -> bool:
        """Check if term1 logically entails term2."""
        for l2 in term2:
            entailed = False
            for l1 in term1:
                if l1[0] == l2[0]:
                    meaning1 = self.semantics_[l1[1]]
                    meaning2 = self.semantics_[l2[1]]
                    
                    if meaning1[1] == meaning2[1]:
                        if meaning1[0] == 'boolean':
                            entailed = True
                            break
                        elif meaning1[0] == 'categorical' and meaning1[2] == meaning2[2]:
                            entailed = True
                            break
                        elif meaning1[0] == 'numerical':
                            if l1[0] == 0 and meaning2[2] >= meaning1[2]:
                                entailed = True
                                break
                            elif l1[0] == 1 and meaning2[2] <= meaning1[2]:
                                entailed = True
                                break
            
            if not entailed:
                return False
        return True
    
    def _simplify_rules(self) -> None:
        """Simplify rule set through multiple optimization steps."""
        simplified_rules = []
        
        for rule in self.rules_:
            prediction = rule[0]
            terms = rule[1]
            
            # Step 1: Boolean optimization
            simplified_terms = minimize_dnf(terms)
            
            # Step 2: Domain knowledge pruning
            simplified_terms = self._prune_terms_using_domain_knowledge(simplified_terms)
            
            # Step 3: Remove useless literals
            simplified_terms = self._prune_terms_useless_literals(
                simplified_terms, prediction
            )
            
            # Step 4: Cross-validation pruning
            if self.X_prune_ is not None:
                simplified_terms = self._prune_terms_cross_validation(
                    simplified_terms, prediction
                )
            
            # Step 5: Remove redundant terms
            necessary_terms = []
            for i in range(len(simplified_terms)):
                necessary = True
                for j in range(i + 1, len(simplified_terms)):
                    if self._entails(simplified_terms[i], simplified_terms[j]):
                        necessary = False
                        break
                if necessary:
                    necessary_terms.append(simplified_terms[i])
            
            simplified_rules.append([prediction, necessary_terms])
        
        self.rules_ = simplified_rules

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the RuleSetClassifier to training data.
        
        Parameters
        ----------
        X : pandas.DataFrame
        
        y : pandas.Series
        
        Returns
        -------
        self : object
            Fitted estimator.
        
        Raises
        ------
        TypeError
            If X is not a pandas DataFrame or y is not a pandas Series.
        """
        self._validate_hyperparameters()
        
        # Check that X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame, got {type(X).__name__}. "
                "RuleSetClassifier requires pandas DataFrames to properly handle "
                "feature names for categorical, numerical, and boolean features."
            )
        
        # Check that y is a pandas Series
        if not isinstance(y, pd.Series):
            raise TypeError(
                f"y must be a pandas Series, got {type(y).__name__}. "
                "RuleSetClassifier requires pandas Series for proper label handling."
            )
        
        # Validate input shapes and values
        if X.shape[0] != len(y):
            raise ValueError(
                f"X and y must have the same number of samples. "
                f"Got X.shape[0]={X.shape[0]} and len(y)={len(y)}"
            )
        
        if X.shape[0] == 0:
            raise ValueError("X and y must contain at least one sample")
        
        # Validate that specified feature names exist in X
        all_specified_features = set(self.boolean_features + 
                                     self.categorical_features + 
                                     self.numerical_features)
        existing_features = set(X.columns)
        missing_features = all_specified_features - existing_features
        
        if missing_features:
            raise ValueError(
                f"The following features were specified but not found in X: "
                f"{sorted(missing_features)}"
            )
        
        # Store classes and feature info
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns)
        
        # Initialize semantics
        self.semantics_ = {}
        
        # Preprocess features
        bool_X = self._preprocess_data(X, y)
        
        # Adjust num_prop if needed
        num_prop = min(self.num_prop, len(bool_X.columns))
        if num_prop < self.num_prop and not self.silent:
            print(f'WARNING: Using {num_prop} features (requested {self.num_prop})')
        
        # Feature selection
        if self.fs_algorithm == 'dt':
            used_props = feature_selection_using_decision_tree(bool_X, y, num_prop)
        else:  # brute
            used_props = feature_selection_using_brute_force(
                bool_X, y, num_prop, self.silent
            )
        
        # Split data for growth and pruning
        if self.growth_size == 1.0:
            self.X_grow_ = bool_X
            self.X_prune_ = None
            self.y_grow_ = y
            self.y_prune_ = None
        else:
            X_grow, X_prune, y_grow, y_prune = train_test_split(
                bool_X, y, train_size=self.growth_size, 
                random_state=self.random_state
            )
            self.X_grow_ = X_grow
            self.X_prune_ = X_prune
            self.y_grow_ = y_grow
            self.y_prune_ = y_prune
        
        # Form and simplify rules
        self._form_rule_set(used_props, self.default_prediction)
        self._simplify_rules()
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : pandas.DataFrame
        
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        
        Raises
        ------
        TypeError
            If X is not a pandas DataFrame.
        ValueError
            If X doesn't have the expected features.
        """
        check_is_fitted(self, ['rules_', 'default_prediction_', 'semantics_'])
        
        # Check that X is a pandas DataFrame
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pandas DataFrame, got {type(X).__name__}. "
                "RuleSetClassifier requires pandas DataFrames with named columns "
                "that match the training data."
            )
        
        # Validate that X has the expected features
        if hasattr(self, 'feature_names_in_'):
            missing_features = set(self.feature_names_in_) - set(X.columns)
            if missing_features:
                raise ValueError(
                    f"X is missing the following features that were present during fit: "
                    f"{sorted(missing_features)}"
                )
            
            extra_features = set(X.columns) - set(self.feature_names_in_)
            if extra_features and not self.silent:
                print(f"WARNING: X contains extra features that will be ignored: "
                      f"{sorted(extra_features)}")
            
            # Reorder columns to match training data
            X = X[self.feature_names_in_]
        
        # Validate shape
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} "
                f"was fitted with {self.n_features_in_} features"
            )
        
        X_np = X.to_numpy()
        num_samples = X_np.shape[0]
        predictions = np.full(num_samples, self.default_prediction_)
        
        for rule in self.rules_:
            output = rule[0]
            terms = rule[1]
            term_satisfied = np.zeros(num_samples, dtype=bool)
            
            for term in terms:
                term_mask = np.ones(num_samples, dtype=bool)
                
                for literal in term:
                    interpretation = self.semantics_[literal[1]]
                    
                    if interpretation[0] == 'boolean':
                        term_mask &= (X_np[:, interpretation[2]] == literal[0])
                    elif interpretation[0] == 'numerical':
                        term_mask &= (X_np[:, interpretation[3]] > interpretation[2]) == literal[0]
                    else:  # categorical
                        term_mask &= (X_np[:, interpretation[3]] == interpretation[2]) == literal[0]
                    
                    if not term_mask.any():
                        break
                
                term_satisfied |= term_mask
            
            predictions[term_satisfied] = output
        
        return predictions

    def _term_support_and_confidence(
        self, 
        term: List, 
        prediction: Any, 
        X: pd.DataFrame, 
        y: pd.Series
    ) -> Tuple[int, float]:
        """Calculate support and confidence for a term."""
        correct_mask = (y == prediction)
        term_mask = np.ones(len(X), dtype=bool)
        
        for literal in term:
            term_mask &= (X[literal[1]] == literal[0])
        
        support = term_mask.sum()
        if support == 0:
            return 0, 0.0
        
        correct = (term_mask & correct_mask).sum()
        confidence = correct / support
        
        return support, confidence

    def __str__(self) -> str:
        """Return human-readable rule set representation."""
        check_is_fitted(self, ['rules_', 'default_prediction_'])
        
        output = []
        for i, rule in enumerate(self.rules_):
            if i > 0:
                output.append('ELSE IF')
            else:
                output.append('IF')
            
            prediction = rule[0]
            terms = rule[1]
            
            for j, term in enumerate(terms):
                if j > 0:
                    output.append('OR ')
                
                term_parts = ['(']
                for k, literal in enumerate(term):
                    if k > 0:
                        term_parts.append(' AND ')
                    
                    if literal[0] == 0:
                        meaning = self.semantics_[literal[1]]
                        if meaning[0] == 'numerical':
                            term_parts.append(f'{meaning[1]} <= {meaning[2]}')
                        else:
                            term_parts.append(f'NOT {literal[1]}')
                    else:
                        term_parts.append(literal[1])
                
                # Use X_grow_ for support/confidence calculation
                support, confidence = self._term_support_and_confidence(
                    term, prediction, self.X_grow_, self.y_grow_
                )
                term_parts.append(f') {{support: {support}, confidence: {confidence:.2f}}}')
                output.append(''.join(term_parts))
            
            output.append(f'THEN {prediction}')
        
        output.append(f'ELSE {self.default_prediction_}')
        return '\n'.join(output)

    def __getstate__(self) -> Dict:
        """
        Get state for pickling. This method enables the classifier to be serialized.
        
        Returns
        -------
        state : dict
            Dictionary containing all necessary state for reconstruction.
        """
        state = self.__dict__.copy()
        # Convert DataFrames to dictionaries for better pickle compatibility
        if hasattr(self, 'X_grow_') and self.X_grow_ is not None:
            state['X_grow_'] = {
                'data': self.X_grow_.to_dict('list'),
                'columns': list(self.X_grow_.columns),
                'index': list(self.X_grow_.index)
            }
        if hasattr(self, 'X_prune_') and self.X_prune_ is not None:
            state['X_prune_'] = {
                'data': self.X_prune_.to_dict('list'),
                'columns': list(self.X_prune_.columns),
                'index': list(self.X_prune_.index)
            }
        if hasattr(self, 'y_grow_') and self.y_grow_ is not None:
            state['y_grow_'] = {
                'data': list(self.y_grow_),
                'index': list(self.y_grow_.index),
                'name': self.y_grow_.name
            }
        if hasattr(self, 'y_prune_') and self.y_prune_ is not None:
            state['y_prune_'] = {
                'data': list(self.y_prune_),
                'index': list(self.y_prune_.index),
                'name': self.y_prune_.name
            }
        return state

    def __setstate__(self, state: Dict) -> None:
        """
        Set state from unpickling. This method enables the classifier to be deserialized.
        
        Parameters
        ----------
        state : dict
            Dictionary containing the state to restore.
        """
        # Reconstruct DataFrames from dictionaries
        if 'X_grow_' in state and isinstance(state['X_grow_'], dict):
            X_grow_dict = state['X_grow_']
            state['X_grow_'] = pd.DataFrame(
                X_grow_dict['data'],
                columns=X_grow_dict['columns'],
                index=X_grow_dict['index']
            )
        
        if 'X_prune_' in state and isinstance(state['X_prune_'], dict):
            X_prune_dict = state['X_prune_']
            state['X_prune_'] = pd.DataFrame(
                X_prune_dict['data'],
                columns=X_prune_dict['columns'],
                index=X_prune_dict['index']
            )
        
        if 'y_grow_' in state and isinstance(state['y_grow_'], dict):
            y_grow_dict = state['y_grow_']
            state['y_grow_'] = pd.Series(
                y_grow_dict['data'],
                index=y_grow_dict['index'],
                name=y_grow_dict['name']
            )
        
        if 'y_prune_' in state and isinstance(state['y_prune_'], dict):
            y_prune_dict = state['y_prune_']
            state['y_prune_'] = pd.Series(
                y_prune_dict['data'],
                index=y_prune_dict['index'],
                name=y_prune_dict['name']
            )
        
        self.__dict__.update(state)

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted model to a file using pickle.
        
        This method provides a convenient way to save models. For production use,
        consider using joblib.dump() directly as it's more efficient for large models.
        
        Parameters
        ----------
        filepath : str or Path
            Path where the model should be saved. Supports .pkl, .pickle extensions.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        
        Examples
        --------
        >>> clf = RuleSetClassifier(num_prop=10)
        >>> clf.fit(X_train, y_train)
        >>> clf.save_model('my_classifier.pkl')
        """
        check_is_fitted(self, ['rules_', 'default_prediction_'])
        
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        if not self.silent:
            print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath: Union[str, Path]) -> 'RuleSetClassifier':
        """
        Load a fitted model from a file.
        
        This method provides a convenient way to load models. For production use,
        consider using joblib.load() directly as it's more efficient for large models.
        
        Parameters
        ----------
        filepath : str or Path
            Path to the saved model file.
        
        Returns
        -------
        model : RuleSetClassifier
            The loaded model.
        
        Examples
        --------
        >>> clf = RuleSetClassifier.load_model('my_classifier.pkl')
        >>> y_pred = clf.predict(X_test)
        """
        filepath = Path(filepath)
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        if not isinstance(model, RuleSetClassifier):
            raise ValueError(f"Loaded object is not a RuleSetClassifier instance")
        
        return model

    def save_rules_as_text(self, filepath: Union[str, Path]) -> None:
        """
        Save the learned rules to a human-readable text file.
        
        Parameters
        ----------
        filepath : str or Path
            Path where the rules should be saved.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        
        Examples
        --------
        >>> clf = RuleSetClassifier(num_prop=10)
        >>> clf.fit(X_train, y_train)
        >>> clf.save_rules_as_text('rules.txt')
        """
        check_is_fitted(self, ['rules_', 'default_prediction_'])
        
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            f.write(str(self))
        
        if not self.silent:
            print(f"Rules saved to {filepath}")

    def save_rules_as_json(self, filepath: Union[str, Path]) -> None:
        """
        Save the learned rules to a JSON file for programmatic access.
        
        The JSON structure includes rules, semantics, default prediction,
        and metadata about the model.
        
        Parameters
        ----------
        filepath : str or Path
            Path where the JSON should be saved.
        
        Raises
        ------
        ValueError
            If the model has not been fitted yet.
        
        Examples
        --------
        >>> clf = RuleSetClassifier(num_prop=10)
        >>> clf.fit(X_train, y_train)
        >>> clf.save_rules_as_json('rules.json')
        """
        check_is_fitted(self, ['rules_', 'default_prediction_'])
        
        # Convert rules to JSON-serializable format
        json_rules = []
        for rule in self.rules_:
            prediction = rule[0]
            terms = rule[1]
            
            json_terms = []
            for term in terms:
                json_term = {
                    'literals': [
                        {
                            'value': int(literal[0]),
                            'feature': literal[1],
                            'semantic': self.semantics_[literal[1]]
                        }
                        for literal in term
                    ]
                }
                
                # Add support and confidence
                support, confidence = self._term_support_and_confidence(
                    term, prediction, self.X_grow_, self.y_grow_
                )
                json_term['support'] = int(support)
                json_term['confidence'] = float(confidence)
                
                json_terms.append(json_term)
            
            json_rules.append({
                'prediction': str(prediction),
                'terms': json_terms
            })
        
        # Create full JSON structure
        json_data = {
            'model_type': 'RuleSetClassifier',
            'hyperparameters': {
                'num_prop': self.num_prop,
                'fs_algorithm': self.fs_algorithm,
                'growth_size': self.growth_size,
                'random_state': self.random_state
            },
            'classes': [str(c) for c in self.classes_],
            'n_features': int(self.n_features_in_),
            'feature_names': list(self.feature_names_in_) if hasattr(self, 'feature_names_in_') else None,
            'default_prediction': str(self.default_prediction_),
            'rules': json_rules,
            'semantics': {k: v for k, v in self.semantics_.items()}
        }
        
        filepath = Path(filepath)
        with open(filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        if not self.silent:
            print(f"Rules saved to {filepath}")

    def get_params(self, deep: bool = True) -> Dict:
        """
        Get parameters for this estimator.
        
        This method is required for sklearn compatibility and enables
        grid search and other hyperparameter tuning methods.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
        
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {
            'num_prop': self.num_prop,
            'fs_algorithm': self.fs_algorithm,
            'growth_size': self.growth_size,
            'random_state': self.random_state,
            'default_prediction': self.default_prediction,
            'boolean_features': self.boolean_features,
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'silent': self.silent
        }

    def set_params(self, **params) -> 'RuleSetClassifier':
        """
        Set the parameters of this estimator.
        
        This method is required for sklearn compatibility and enables
        grid search and other hyperparameter tuning methods.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        
        Returns
        -------
        self : object
            Estimator instance.
        """
        valid_params = self.get_params(deep=False).keys()
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(
                    f"Invalid parameter {key} for estimator {type(self).__name__}. "
                    f"Valid parameters are: {sorted(valid_params)}"
                )
            setattr(self, key, value)
        return self