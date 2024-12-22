# rsclassifier

[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![PyPI Downloads](https://static.pepy.tech/badge/rsclassifier)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue.svg)](https://github.com/ReijoJaakkola/rsclassifier)

# Overview

This package consist of two modules, `rsclassifier` and `discretization`. The first one implements a rule-based machine learning algorithm, while the second one implements an entropy-based supervised discretization algorithm.

# Installation

To install the package, you can simply use `pip`:

```bash
pip install rsclassifier
```

# First module: rsclassifier

This module contains the class `RuleSetClassifier`, which is a non-parametric supervised learning method that can be used for classification and data mining. As the name suggests, `RuleSetClassifier` produces classifiers which consist of a set of rules which are learned from the given data. As a concrete example, the following classifier was produced from the well-known Iris data set.

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

As an another concrete example, the following classifier was produced from the Breast Cancer Wisconsin data set.

**IF**  
**(bare_nuclei > 2.50 AND clump_thickness > 4.50) {support: 134, confidence: 0.94}**  
**OR (uniformity_of_cell_size > 3.50) {support: 150, confidence: 0.94}**  
**OR (bare_nuclei > 5.50) {support: 119, confidence: 0.97}**  
**THEN 4**  
**ELSE 2**  

This classifier classifiers all tumors which satisfy one of the four rules listed above as malign (4) and all other tumors as benign (2).

### Advantages
- `RuleSetClassifier` produces extremely interpretable and transparent classifiers.
- It is very easy to use, as it has only two hyperparameters.
- It can handle both categorical and numerical data.
- The learning process is very fast.

### How to use RuleSetClassifier

Let `rsc` be an instance of `RuleSetClassifier` and let `X` be a pandas dataframe (input features) and `y` a pandas series (target labels).
- **Load the data**: Use `rsc.load_data(X, y, boolean, categorical, numerical)` where `boolean`, `categorical` and `numerical` are (possibly empty) lists specifying which features in `X` are boolean, categorical or numerical, respectively. This function converts the data into a Boolean form for rule learning and store is to `rsc`.
- **Fit the classifier**: After loading the data, call `rsc.fit(num_prop, fs_algorithm, growth_size)`. Note that unlike in scikit-learn, this function doesn't take `X` and `y` directly as arguments; they are loaded beforehand as part of `load_data`. The two hyperparameters `num_prop` and `growth_size` work as follows.
    - `num_prop` is an upper bound on the number of proposition symbols allowed in the rules. The smaller `num_prop` is, the more interpretable the models are. The downside of having small `num_prop` is of course that the resulting model has low accuracy (i.e., it underfits), so an optimal value for `num_prop` is the one which strikes a balance between interpretability and accuracy.
    - `fs_algorithm` determines the algorithm used for selecting the Boolean features used by the classifier. Has two options, `dt` (which is the default) and `brute`. `dt` uses decision trees for feature selection, `brute` finds the set of features for which the error on training data is minimized. Note that running `brute` with a large `num_prop` can take a long time plus it can lead to overfitting.
    - `growth_size` is a float in the range (0, 1], determining the proportion of X used for learning rules. The remaining portion is used for pruning. If `growth_size` is set to 1, which is the default value, no pruning is performed. Also 2/3 seems to work well in practice.
- **Make predictions**: Use `rsc.predict(X)` to generate predictions. This function returns a pandas Series.
- **Visualize the classifier**: Simply print the classifier to visualize the learned rules (together with their support and confidence).

**Note**: At present, `RuleSetClassifier` does not support datasets with missing values. You will need to preprocess your data (e.g., removing missing values) before using the classifier.

### Background

The rule learning method implemented by `RuleSetClassifier` was inspired by and extends the approach taken in the [paper](https://arxiv.org/abs/2402.05680), which we refer here as the **ideal DNF-method**. The ideal DNF-method goes as follows. First, the input data is Booleanized. Then, a small number of promising features is selected. Finally, a DNF-formula is computed for those promising features for which the number of misclassified points is as small as possible.

The way `RuleSetClassifier` extends and modifies the ideal DNF-method is mainly as follows.
- We use an entropy-based Booleanization for numerical features with minimum description length principle working as a stopping rule.
- `RuleSetClassifier` is not restricted to binary classification tasks.
- We implement rule pruning as a postprocessing step. This is important, as it makes the rules shorter and hence more interpretable.

### Example

```python
import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from rsclassifier import RuleSetClassifier

# Load the data set.
iris = datasets.load_iris()
df = pd.DataFrame(data= iris.data, columns= iris.feature_names)
df['target'] = iris.target

# Split it into train and test.
X = df.drop(columns = ['target'], axis = 1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)

# Initialize RuleSetClassifier.
rsc = RuleSetClassifier()
# All the features of iris.csv are numerical.
rsc.load_data(X = X_train, y = y_train, numerical = X.columns)
# Fit the classifier with a maximum of 2 proposition symbols.
rsc.fit(num_prop = 2)

# Measure the accuracy of the resulting classifier.
train_accuracy = accuracy_score(rsc.predict(X_train), y_train)
test_accuracy = accuracy_score(rsc.predict(X_test), y_test)

# Display the classifier and its accuracies.
print()
print(rsc)
print(f'Rule set classifier training accuracy: {train_accuracy}')
print(f'Rule set classifier test accuracy: {test_accuracy}')
```

# Second module: discretization

This module contains the functions `find_pivots`, `booleanize_categorical_features`, and `booleanize_numerical_features`.

---

## find_pivots

`find_pivots` can be used for entropy-based supervised discretization of numeric features. This is the function that `RuleSetClassifier` also uses for Booleanizing numerical data.

### Parameters:
- **`x`** (`pandas.Series`): Contains the values of the numeric feature to be discretized.
- **`y`** (`pandas.Series`): Holds the corresponding target variable values.

### Returns:
- `list`: A list of pivot points that represent where the feature `x` should be split to achieve maximum information gain regarding the target variable `y`. The list of pivots can be empty if the feature `x` is not useful for predicting `y`.

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

## booleanize_categorical_features

Converts categorical features into Boolean features using one-hot encoding style.

### Parameters:
- **`X`** (`pandas.DataFrame`): The feature data.
- **`categorical_features`** (`list`): List of categorical features to be converted.

### Returns:
- `pandas.DataFrame`: A DataFrame with the Booleanized categorical features.

### Example:
```python
import pandas as pd
from discretization import booleanize_categorical_features

# Sample DataFrame
data = pd.DataFrame({
    'Feature1': ['A', 'B', 'A', 'C'],
    'Feature2': [1, 2, 3, 4]
})

# Booleanize the categorical feature
categorical_features = ['Feature1']
bool_data = booleanize_categorical_features(data, categorical_features)
print(bool_data)
```

## booleanize_numerical_features

Discretizes numerical features using entropy-based pivot points and converts them into Boolean features.

### Parameters:
- **`X`** (`pandas.DataFrame`): The feature data.
- **`y`** (`pandas.Series`): The target labels.
- **`numerical_features`** (`list`): List of numerical features to be discretized.
- **`silent`** (`bool`, optional): Whether to suppress progress output (default is `False`).

### Returns:
- `pandas.DataFrame`: A DataFrame with the Booleanized numerical features.

### Example:
```python
import pandas as pd
from discretization import booleanize_numerical_features

# Sample DataFrame and target
data = pd.DataFrame({
    'Feature1': [1.2, 3.4, 5.6, 7.8],
    'Feature2': [2.1, 4.3, 6.5, 8.7]
})
target = pd.Series([0, 1, 0, 1])

# Booleanize the numerical features
numerical_features = ['Feature1', 'Feature2']
bool_data = booleanize_numerical_features(data, target, numerical_features)
print(bool_data)
```