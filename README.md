# rsclassifier

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://static.pepy.tech/badge/rsclassifier)](https://pepy.tech/projects/rsclassifier)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue.svg)](https://github.com/ReijoJaakkola/rsclassifier)

# Overview

This package consists of two modules, `rsclassifier` and `discretization`. The first one implements a rule-based machine learning algorithm that is fully compatible with scikit-learn, while the second one implements an entropy-based supervised discretization algorithm and a class for booleanizing data.

# Installation

To install the package, you can simply use `pip`:

```bash
pip install rsclassifier
```

# First module: rsclassifier

This module contains the class `RuleSetClassifier`, which is a **scikit-learn compatible** non-parametric supervised learning method that can be used for classification and data mining. As the name suggests, `RuleSetClassifier` produces classifiers which consist of a set of rules which are learned from the given data. As a concrete example, the following classifier was produced from the well-known Iris data set.

**IF**  
**(petal_length_in_cm > 2.45 AND petal_width_in_cm > 1.75) {support: 33, confidence: 0.97}**  
**THEN virginica**  
**ELSE IF**  
**(petal_length_in_cm <= 2.45 AND petal_width_in_cm <= 1.75) {support: 31, confidence: 1.00}**  
**THEN setosa**  
**ELSE versicolor**  

Notice that each rule is accompanied by:
- **Support**: The number of data points that satisfy the rule.
- **Confidence**: The probability that a data point satisfying the rule is correctly classified.

As another concrete example, the following classifier was produced from the Breast Cancer Wisconsin data set.

**IF**  
**(bare_nuclei > 2.50 AND clump_thickness > 4.50) {support: 134, confidence: 0.94}**  
**OR (uniformity_of_cell_size > 3.50) {support: 150, confidence: 0.94}**  
**OR (bare_nuclei > 5.50) {support: 119, confidence: 0.97}**  
**THEN 4**  
**ELSE 2**  

This classifier classifies all tumors which satisfy one of the four rules listed above as malign (4) and all other tumors as benign (2).

### Advantages
- `RuleSetClassifier` produces extremely interpretable and transparent classifiers.
- **Fully compatible with scikit-learn**: Works seamlessly with pipelines, cross-validation, grid search, and all sklearn utilities.
- It is very easy to use, with intuitive hyperparameters.
- It can handle both categorical and numerical data.
- The learning process is very fast.
- **Works with pandas DataFrames and Series** for proper feature name handling.

### Requirements
- **Input format**: `RuleSetClassifier` requires pandas DataFrame for features (X) and pandas Series for labels (y). This ensures proper handling of feature names for categorical, numerical, and boolean feature processing.
- **No missing values**: The classifier does not support missing values. Please preprocess your data (e.g., imputation or removal) before fitting.

### How to use RuleSetClassifier

`RuleSetClassifier` follows the standard scikit-learn API, making it easy to integrate into existing ML workflows.

#### Basic Usage

```python
from rsclassifier import RuleSetClassifier

# Initialize the classifier with feature types
clf = RuleSetClassifier(
    num_prop=10,                              # Maximum number of features to use
    numerical_features=['feature1', 'feature2'],  # List of numerical features
    categorical_features=['feature3'],            # List of categorical features
    boolean_features=['feature4'],                # List of boolean features
    fs_algorithm='dt',                            # Feature selection: 'dt' or 'brute'
    growth_size=0.67,                             # Proportion for rule growth (training)
    random_state=42,                              # For reproducibility
    silent=False                                  # Show progress
)

# Fit the classifier (just like any sklearn estimator)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Display the learned rules
print(clf)
```

#### Hyperparameters

- **`num_prop`** (required): Maximum number of Boolean features to use in rules. Smaller values produce more interpretable models but may underfit. Balance interpretability with accuracy.

- **`fs_algorithm`** (default='dt'): Algorithm for feature selection:
  - `'dt'`: Uses decision trees (fast, recommended)
  - `'brute'`: Brute force search minimizing training error (can be slow for large `num_prop`)

- **`growth_size`** (default=1.0): Proportion of data used for rule learning, in range (0, 1]. The remaining data is used for rule pruning. 
  - `1.0`: No pruning (uses all data for learning)
  - `0.67`: Common choice, uses 2/3 for learning, 1/3 for pruning

- **`random_state`** (default=42): Controls shuffling for train/prune split.

- **`default_prediction`** (default=None): Prediction when no rule matches. If None, uses the mode of training data.

- **`numerical_features`** (default=[]): List of numerical feature names.

- **`categorical_features`** (default=[]): List of categorical feature names.

- **`boolean_features`** (default=[]): List of boolean feature names.

- **`silent`** (default=False): If True, suppresses progress output.

#### Scikit-learn Integration

`RuleSetClassifier` works with all standard scikit-learn tools:

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cross-validation
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Grid search for hyperparameter tuning
param_grid = {
    'num_prop': [5, 10, 15, 20],
    'growth_size': [0.67, 0.8, 1.0],
    'fs_algorithm': ['dt', 'brute']
}
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")

# Use in pipelines
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', clf)
])
pipeline.fit(X_train, y_train)
```

#### Model Persistence

`RuleSetClassifier` supports multiple methods for saving and loading models:

```python
import joblib

# Method 1: Using joblib (recommended for production)
joblib.dump(clf, 'model.pkl')
clf_loaded = joblib.load('model.pkl')

# Method 2: Using built-in save/load methods
clf.save_model('model.pkl')
clf_loaded = RuleSetClassifier.load_model('model.pkl')

# Method 3: Using standard pickle
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('model.pkl', 'rb') as f:
    clf_loaded = pickle.load(f)

# Export rules in human-readable format
clf.save_rules_as_text('rules.txt')

# Export rules as JSON for programmatic access
clf.save_rules_as_json('rules.json')
```

The JSON export includes complete rule information with support and confidence metrics, making it easy to integrate learned rules into other systems or for documentation purposes.

**Note**: At present, `RuleSetClassifier` does not support datasets with missing values. You will need to preprocess your data (e.g., imputing or removing missing values) before using the classifier.

### Background

The rule learning method implemented by `RuleSetClassifier` was inspired by and extends the approach taken in the [paper](https://arxiv.org/abs/2402.05680), which we refer to here as the **ideal DNF-method**. The ideal DNF-method goes as follows. First, the input data is Booleanized. Then, a small number of promising features is selected. Finally, a DNF-formula is computed for those promising features for which the number of misclassified points is as small as possible.

The way `RuleSetClassifier` extends and modifies the ideal DNF-method is primarily as follows:
- We use an entropy-based Booleanization for numerical features with minimum description length principle working as a stopping rule.
- `RuleSetClassifier` is not restricted to binary classification tasks.
- We use the Quine-McCluskey algorithm for finding near-optimal size DNF-formulas.
- We also implement rule pruning as a postprocessing step. This makes the rules shorter and more interpretable.

### Complete Example

```python
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from rsclassifier import RuleSetClassifier
import joblib

# Load the data set
iris = datasets.load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

# Initialize RuleSetClassifier
# All features in iris are numerical
clf = RuleSetClassifier(
    num_prop=2,                          # Use maximum 2 features
    numerical_features=X.columns.tolist(),  # All features are numerical
    growth_size=0.67,                    # Use 2/3 for learning, 1/3 for pruning
    random_state=42
)

# Fit the classifier
clf.fit(X_train, y_train)

# Make predictions
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

# Measure accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Display results
print("\n" + "="*60)
print("LEARNED RULE SET")
print("="*60)
print(clf)
print("\n" + "="*60)
print("PERFORMANCE METRICS")
print("="*60)
print(f'Training accuracy: {train_accuracy:.3f}')
print(f'Test accuracy: {test_accuracy:.3f}')
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

# Save the model for later use
clf.save_model('iris_classifier.pkl')
clf.save_rules_as_text('iris_rules.txt')
clf.save_rules_as_json('iris_rules.json')

# Load and use the saved model
clf_loaded = RuleSetClassifier.load_model('iris_classifier.pkl')
y_pred_loaded = clf_loaded.predict(X_test)
print(f"\nLoaded model accuracy: {accuracy_score(y_test, y_pred_loaded):.3f}")
```

### Migration from Previous Versions

If you're upgrading from an earlier version of `rsclassifier`, here's how to update your code:

**Old API:**
```python
rsc = RuleSetClassifier()
rsc.load_data(X=X_train, y=y_train, numerical=X.columns)
rsc.fit(num_prop=2)
```

**New API (sklearn-compatible):**
```python
rsc = RuleSetClassifier(
    num_prop=2,
    numerical_features=X.columns.tolist()
)
rsc.fit(X_train, y_train)
```

Key changes:
- Feature types are now specified in the constructor, not in a separate `load_data()` method
- `fit()` now takes `X` and `y` directly (standard sklearn pattern)
- All hyperparameters are set in `__init__()`
- The classifier now extends `BaseEstimator` and `ClassifierMixin` from sklearn

# Second module: discretization

This module contains the `find_pivots` function and the `Booleanizer` class.

## find_pivots

`find_pivots` can be used for entropy-based supervised discretization of numeric features. This is the function that `RuleSetClassifier` and `Booleanizer` use for Booleanizing numerical data.

### How does it work?

At a high level, the algorithm behind `find_pivots` works as follows:
1. **Sorting**: The feature column `x` is sorted to ensure that potential pivots represent transitions between distinct data values.
2. **Candidate Pivot Calculation**: Midpoints between consecutive unique values in the sorted list are calculated as candidate pivots.
3. **Split Evaluation**: Each candidate pivot is evaluated by splitting the dataset into two subsets:
   - One subset contains records with feature values ≤ the pivot.
   - The other subset contains records with feature values > the pivot.
4. **Information Gain Calculation**: Information gain is calculated to assess the quality of each split.
5. **Recursion**: If a split significantly increases information gain, the process is recursively applied to each subset until no further significant gains can be achieved.

For more details, see Section 7.2 of **Data Mining: Practical Machine Learning Tools and Techniques with Java Implementations** by Ian H. Witten and Eibe Frank.

### Example:
```python
import pandas as pd
from sklearn import datasets
from discretization import find_pivots

# Load the dataset
iris = datasets.load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Calculate pivots for the feature "petal length (cm)"
pivots = find_pivots(df['petal length (cm)'], df['target'])
print(pivots)  # Output: [2.45, 4.75]
```

## Booleanizer

The `Booleanizer` class is designed to transform a dataset into a booleanized format for classification purposes. The booleanized data is obtained by applying one-hot encoding to categorical features and splitting numerical features into boolean columns based on learned pivot values.

### How to use:
Let `booleanizer` be an instance of `Booleanizer`, `X` a pandas dataframe (input features) and `y` a pandas series (target labels).
- **Collect unique values for categorical features:** For categorical features we need to store the unique values (or classes) in each categorical feature. This step is done by calling the `store_classes_for_cat_features` method.
- **Learn pivots for numerical features:** For the numerical features, `Booleanizer` uses entropy-based discretization to learn pivot points which it will then store. This step is done using the `store_pivots_for_num_features` method.
- **Booleanize the data:** After storing the categories for categorical features and the pivot points for numerical features, the data can be booleanized using the `booleanize_dataframe` method.

Note that a single instance of `Booleanizer` can be used to booleanize several different datasets. In particular, a `Booleanizer` trained on the training data can be used to booleanize the test data.

### Example:
```python
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from discretization import Booleanizer

# Load the data
iris = datasets.load_iris()
X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the booleanizer
booleanizer = Booleanizer()

# Learn pivots from the training data
booleanizer.store_pivots_for_num_features(X_train, y_train, X.columns)

# Booleanize the training data and the test data using the same pivots
X_train_bool = booleanizer.booleanize_dataframe(X_train)
X_test_bool = booleanizer.booleanize_dataframe(X_test)
```

# Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

# License

This project is licensed under the MIT License - see the LICENSE file for details.

# Citation

If you use this package in your research, please cite:

```bibtex
@software{rsclassifier,
  author = {Jaakkola, Reijo},
  title = {rsclassifier: Rule-based Classification with Scikit-learn Compatibility},
  year = {2024},
  url = {https://github.com/ReijoJaakkola/rsclassifier}
}
```