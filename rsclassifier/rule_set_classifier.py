import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Any, Tuple
from scipy.special import betainc
from scipy.stats import binom
from rsclassifier.feature_selection import feature_selection_using_decision_tree, feature_selection_using_random_forest
from rsclassifier.quine_mccluskey import minimize_dnf
from rsclassifier.entropy_based_discretization import find_pivots

class Error(Exception):
    pass

class RuleSetClassifier:
    def __init__(self):
        """
        Initialize the RuleSetClassifier with default values.
        """
        self.semantics = {}  # Maps propositional symbols to pairs [type, feature, value].
        self.rules = []  # List of rules. Each rule is a pair [output, terms].
        self.default_prediction = None  # Output if no rule matches.
        
        self.is_initialized = False
        self.is_fitted = False

        self.X = None  # The feature data.
        self.y = None  # The target labels.

    def _booleanize_categorical_features(self, X : pd.DataFrame, categorical_features : list) -> pd.DataFrame:
        """
        Convert categorical features into Boolean features.

        Args:
            X (pandas.DataFrame): The feature data.
            categorical_features (list): List of categorical features.

        Returns:
            pandas.DataFrame: Data with Booleanized categorical features.
        """
        local_X = X.copy()
        for feature in categorical_features:
            unique_values = local_X[feature].unique()
            new_columns = {}
            for value in unique_values:
                # Create a new column for each value (one-hot encoding style).
                new_columns[feature + ' = ' + str(value)] = (local_X[feature] == value)
                # Store semantics for future use.
                self.semantics[feature + ' = ' + str(value)] = ['categorical', feature, value]
            # Concatenate the new Boolean columns with the original data.
            local_X = pd.concat([local_X, pd.DataFrame(new_columns)], axis=1)
        # Drop original categorical columns.
        local_X.drop(columns=categorical_features, inplace=True)
        return local_X

    def _booleanize_numerical_features(self, X : pd.DataFrame, y : pd.Series, numerical_features : list, silent : bool = False) -> pd.DataFrame:
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
        local_X = X.copy()
        for feature in tqdm(numerical_features, total=len(numerical_features), desc='Discretizing numerical features...', disable = silent):
            # Combine the feature and target for pivot finding.
            Z = pd.concat([local_X[feature], y], axis=1)
            # Find pivot points for discretization.
            pivots = find_pivots(Z)
            if len(pivots) == 0:
                # Skip features with no suitable pivots.
                continue
            new_columns = {}
            for pivot in pivots:
                # Create a Boolean column for values greater than the pivot.
                new_columns[f'{feature} > {pivot:.2f}'] = local_X[feature] > pivot
                # Store semantics for future use.
                self.semantics[f'{feature} > {pivot:.2f}'] = ['numerical', feature, pivot]
            # Concatenate new columns with the data.
            local_X = pd.concat([local_X, pd.DataFrame(new_columns)], axis=1)
        # Drop original numerical columns.
        local_X.drop(columns=numerical_features, inplace=True)
        return local_X

    def _get_type(self, row : pd.Series):
        """
        Convert a row into a list of literals representing its type.

        Args:
            row (pandas.Series): A row of feature data.

        Returns:
            list: A list of literals representing the type.
        """
        type = []
        for atom in row.keys():
            if row[atom] == True:
                type.append([1, atom])
            else:
                type.append([0, atom])
        return type

    # Loads and preprocesses data into the classifier.
    def load_data(self, X : pd.DataFrame, y : pd.Series, boolean : list = [], categorical : list = [], numerical : list = [], silent : bool = False) -> None:
        """
        Load and preprocess the data into the classifier by converting features to Boolean features.

        Args:
            X (pandas.DataFrame): The feature data.
            y (pandas.Series): The target labels.
            boolean (list): List of Boolean features.
            categorical (list): List of categorical features.
            numerical (list): List of numerical features.
            silent (bool): Whether to suppress output.
        """
        bool_X = X.copy()
        if len(boolean) > 0:
            for feature in boolean:
                self.semantics[feature] = ['boolean', feature]
        if len(categorical) > 0:
            bool_X = self._booleanize_categorical_features(bool_X, categorical)
        if len(numerical) > 0:
            bool_X = self._booleanize_numerical_features(bool_X, y, numerical, silent)
        self.X = bool_X
        self.y = y
        self.is_initialized = True
        if not silent:
            print(f'Total number of Boolean features: {len(self.X.columns)}')

    def _form_rule_list(self, features : list, default_prediction : Any, silent : bool) -> None:
        """
        Form a list of rules based on the data.

        Args:
            features (list): List of features to use.
            default_prediction (any): Default prediction if no rule matches.
            silent (bool): Whether to suppress output.
        """
        self.rules = []
        local_X = self.X[features]
        
        # Track unique types and scores for each type.
        types = []
        type_scores = {}
        unique_y = list(self.y.unique())

        for index, row in tqdm(local_X.iterrows(), total=len(local_X), desc='Calculating probabilities...', disable=silent):
            # Convert row into a "type".
            type = self._get_type(row)
            type_code = hash(str(type))
            if type_code not in type_scores:
                types.append(type)
                type_scores[type_code] = {v: 0 for v in unique_y}
            # Increment score for the correct label.
            type_scores[type_code][self.y.loc[index]] += 1

        # Determine the default prediction if not specified.
        if default_prediction is None:
            self.default_prediction = self.y.value_counts().idxmax()
        else:
            self.default_prediction = default_prediction
        unique_y.remove(self.default_prediction)

        rules = {v: [] for v in unique_y}

        # Form rules based on the types.
        for type in tqdm(types, total=len(types), desc='Forming the classifier...', disable=silent):
            type_code = hash(str(type))
            output = max(type_scores[type_code], key=type_scores[type_code].get)
            if output != self.default_prediction:
                rules[output].append(type)

        # Remove empty rules.
        rules = {k: v for k, v in rules.items() if v}
        
        self.rules = list(rules.items())  # Store the rules.

    def _prune_terms_using_domain_knowledge(self, terms : list) -> list:
        """
        Prune terms using domain knowledge to simplify them.

        Args:
            terms (list of lists): The list of terms to simplify.

        Returns:
            list: Simplified terms.
        """
        simplified_terms = []
        for term in terms:
            simplified_term = []
            positive_categories = []
            upper_bounds = {}
            lower_bounds = {}
            for literal in term:
                meaning = self.semantics[literal[1]]
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
                else:
                    simplified_term.append(literal)
            for feature in upper_bounds.keys():
                simplified_term.append([0, f'{feature} > {upper_bounds[feature]:.2f}'])
            for feature in lower_bounds.keys():
                simplified_term.append([1, f'{feature} > {lower_bounds[feature]:.2f}'])
            simplified_terms.append(simplified_term)
        return simplified_terms

    def _term_support_and_confidence(self, term : list, prediction : Any) -> Tuple[int, int]:
        """
        Calculate the support and confidence for a given term and prediction.

        Support: The number of rows (instances) where the term is true.
        Confidence: The number of rows where the term is true and the prediction matches the actual target value.

        Args:
            term (list): A list of literals.
            prediction (any): The predicted value associated with the term.

        Returns:
            tuple: (t, p)
                - t (int): The number of rows where the term is satisfied (support).
                - p (int): The number of rows where the term is satisfied, and the prediction matches the actual target value (confidence).
        """
        # Identify rows where the prediction is correct
        correct_prediction_mask = (self.y == prediction)

        # Check whether each row satisfies the term
        term_mask = np.ones(len(self.X), dtype = bool)
        for literal in term:
            term_mask &= (self.X[literal[1]] == literal[0])

        # Number of rows where term is true (t) and correct prediction (p)
        t = term_mask.sum()
        p = (term_mask & correct_prediction_mask).sum()

        return t, p

    def _evaluate_term(self, term : list, prediction : Any) -> Tuple[int, float]:
        """
        Evaluate the quality of a given term for a specified prediction by calculating its support and a probabilistic score.
        
        The score represents how likely it is that a random rule with the same coverage would achieve at least the same 
        accuracy as the term in question.

        Args:
            term (list): A list of literals.
            prediction (any): The predicted label associated with the term.

        Returns:
            tuple: (t, score)
                - t (int): The number of rows where the term is satisfied (support).
                - score (float): A score indicating how likely a random rule would match the performance of the term.
        """
        T = len(self.X)
        
        # Identify rows where the prediction is correct
        correct_prediction_mask = (self.y == prediction)
        P = correct_prediction_mask.sum()

        t, p = self._term_support_and_confidence(term, prediction)

        # Calculate the probability that a random rule with the same coverage as term would get at least as good accuracy.
        if t > 12:
            score = betainc(p, t - p + 1, P / T)
        else:
            score = 1 - binom.cdf(t, p - 1, P / T)
        return t, score
    
    def _prune_term(self, term : list, prediction : Any) -> list:
        """
        Prune a given term to remove unnecessary literals using probabilistic evaluation.

        Args:
            term (list): A list of literals.
            prediction (Any): The predicted class label associated with the term.

        Returns:
            list: The pruned term.
        """
        local_term = term.copy()
        while True:
            lowest_score = self._evaluate_term(local_term, prediction)
            best_term = local_term
            for i in range(len(local_term)):
                reduced_term = local_term[:i] + local_term[i + 1:]  # Avoid deep copy
                score = self._evaluate_term(reduced_term, prediction)
                if score < lowest_score:
                    lowest_score = score
                    best_term = reduced_term
            if best_term == local_term:
                break
            local_term = best_term
        return local_term

    def _entails(self, term1 : list, term2 : list) -> bool:
        """
        Check whether term1 entails term2, meaning every condition in term2 is satisfied by term1.

        Args:
            term1 (list): The first term.
            term2 (list): The second term.

        Returns:
            bool: True if term1 entails term2, False otherwise.
        """
        for l2 in term2:
            # We need to check that term1 contains a literal l1 which entails l2.
            entailed = False
            for l1 in term1:
                if l1[0] == l2[0]:
                    meaning1 = self.semantics[l1[1]]
                    meaning2 = self.semantics[l2[1]]
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
                            elif meaning2[2] <= meaning1[2]:
                                entailed = True
                                break
            if not entailed:
                return False
        return True
    
    def _simplify(self, silent = False) -> None:
        """
        Simplify the classifier's rule set by pruning and removing redundant terms.

        The simplification process involves three main steps:
        1. Boolean optimization using the Quine-McCluskey algorithm.
        2. Domain-specific pruning. E.g. (x > 7 AND x > 6) is equivalent with (x > 7).
        3. Further pruning based on probabilistic heuristics.
        4. Removing redundant terms by checking if any term entails another.

        Args:
            silent (bool): Whether to suppress output.
        """
        simplified_rules = []
        for rule in self.rules:
            prediction = rule[0]
            terms = rule[1]
            
            # Step 1. Boolean optimization + domain knowledge.
            simplified_terms = self._prune_terms_using_domain_knowledge(minimize_dnf(terms))

            # Step 2. Further pruning based on probabilistic heuristics.
            for i in tqdm(range(len(simplified_terms)), desc = f'Pruning terms for class {prediction}...', disable = silent):
                simplified_terms[i] = self._prune_term(simplified_terms[i], prediction)

            # Step 3. Pruning can cause some of the rules to become redundant. Remove them.
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

        # TODO: Pruning can also lead to conflicts between rules. Is this a problem?

        self.rules = simplified_rules

    def fit(self, num_prop : int, feature_selection : str = 'dt', default_prediction : Any = None, silent : bool = False) -> None:
        """
        Train the RuleSetClassifier by selecting features, forming rules, and simplifying them.

        Args:
            num_prop (int): The number of features (properties) to use.
            feature_selection (str): Either 'dt' or 'rf'.
            default_prediction (any): The default prediction if no rule matches.
            silent (bool): If True, suppress output during training.

        Returns:
            None
        """
        if not self.is_initialized:
            raise Error('Data has not been loaded.')
        
        if num_prop > len(self.X.columns):
            print('WARNING: num_prop more than the number of features. All of the features will be used.')
            num_prop = len(self.X.columns)
        
        if feature_selection == 'dt':
            used_props = feature_selection_using_decision_tree(self.X, self.y, num_prop)
        elif feature_selection == 'rf':
            used_props = feature_selection_using_random_forest(self.X, self.y, num_prop)
        else:
            raise Error('Invalid feature selection method.')
        self._form_rule_list(used_props, default_prediction, silent)
        self._simplify(silent)
        self.is_fitted = True

    def evaluate(self, assignment : dict) -> Any:
        """
        Evaluate an input assignment by checking which rule it satisfies.

        Args:
            assignment (dict): A dictionary representing the feature values of an instance.

        Returns:
            The predicted output class based on the rules.
        """
        for rule in self.rules:
            output = rule[0]
            terms = rule[1]
            for term in terms:
                truth_value = True
                for literal in term:
                    interpretation = self.semantics[literal[1]]
                    if interpretation[0] == 'boolean':
                        if literal[0] != assignment[interpretation[1]]:
                            truth_value = False
                            break
                    elif interpretation[0] == 'numerical':
                        if literal[0] != (assignment[interpretation[1]] > interpretation[2]):
                            truth_value = False
                            break
                    else:
                        if literal[0] != (assignment[interpretation[1]] == interpretation[2]):
                            truth_value = False
                            break
                if truth_value:
                    return output
        return self.default_prediction

    def predict(self, X : pd.DataFrame) -> pd.Series:
        """
        Predict the class labels for a dataset.

        Args:
            X (pandas.DataFrame): The feature data for which predictions are made.

        Returns:
            pandas.Series: The predicted class labels.
        """
        if not self.is_fitted:
            raise Error('Model has not been fitted.')
        def get_prediction(row):
            assignment = row.to_dict()
            return self.evaluate(assignment)
        return X.apply(get_prediction, axis=1)

    def __str__(self) -> str:
        """
        Return a string representation of the rule set.

        Each rule is displayed with its corresponding terms, support, and confidence.

        Returns:
            str: A human-readable string representation of the rule set.
        """
        output = str()
        for i in range(len(self.rules)):
            rule = self.rules[i]
            if i > 0:
                output += 'ELSE IF\n'
            else:
                output += 'IF\n'
            prediction = rule[0]
            terms = rule[1]
            for i in range(len(terms)):
                if i > 0:
                    output += 'OR '
                output += '('
                term = terms[i]
                for j in range(len(term)):
                    if j > 0:
                        output += ' AND '
                    if term[j][0] == 0:
                        meaning = self.semantics[term[j][1]]
                        if meaning[0] == 'numerical':
                            output += f'{meaning[1]} <= {meaning[2]}'
                        else:
                            output += f'NOT {term[j][1]}'
                    else:
                        output += term[j][1]
                support, confidence = self._term_support_and_confidence(term, prediction)
                confidence = confidence / support
                output += f') {{support: {support}, confidence: {confidence:.2f}}}\n'
            output += f'THEN {rule[0]}\n'
        output += f'ELSE {self.default_prediction}\n'
        return output