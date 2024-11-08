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

# `rsclassifier`

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

### How to use `RuleSetClassifier`

Let `rsc` be an instance of `RuleSetClassifier` and let `X` be a pandas dataframe (input features) and `y` a pandas series (target labels).
- **Load the data**: Use `rsc.load_data(X, y, boolean, categorical, numerical)` where `boolean`, `categorical` and `numerical` are (possibly empty) lists specifying which features in `X` are boolean, categorical or numerical, respectively. This function converts the data into a Boolean form for rule learning and store is to `rsc`.
- **Fit the classifier**: After loading the data, call `rsc.fit(num_prop, growth_size)`. Note that unlike in scikit-learn, this function doesn't take `X` and `y` directly as arguments; they are loaded beforehand as part of `load_data`. The two hyperparameters `num_prop` and `growth_size` work as follows.
    - `num_prop` is an upper bound on the number of proposition symbols allowed in the rules. The smaller `num_prop` is, the more interpretable the models are. The downside of having small `num_prop` is of course that the resulting model has low accuracy (i.e., it underfits), so an optimal value for `num_prop` is the one which strikes a balance between interpretability and accuracy. 
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

# `discretization`

This module contains the function `find_pivots`, which can be used for entropy-based supervised discretization of numeric features. This is the function that also `RuleSetClassifier` uses for Booleanizing numerical data.

The function takes two pandas series, `x` and `y`, as inputs:
- `x`: Contains the values of the numeric feature to be discretized.
- `y`: Holds the corresponding target variable values.

The output of the function is a list of pivot points that represent where the feature `x` should be split to achieve maximum information gain regarding the target variable `y`. These pivots can be used to Booleanize the feature `x`. 

Note that the list of pivots can be empty! This simply means that the feature `x` is most likely not useful for predicting the value of `y`.

### How does it work?

At a high-level the algorithm behind `find_pivots` can be described as follows.

- **Sorting**: The feature column `x` is first sorted to ensure that the midpoints calculated as potential pivots are meaningful and represent transitions between distinct data values.
- **Candidate Pivot Calculation**: For each pair of consecutive unique values in the sorted list, a midpoint is calculated. These midpoints serve as candidate pivots for splitting the dataset.
- **Split Evaluation**: Each candidate pivot is evaluated by splitting the dataset into two subsets:
  - One subset contains all records with feature values less than or equal to the pivot.
  - The other subset contains all records with feature values greater than the pivot.
- **Information Gain Calculation**: For each split, the information gain is calculated based on how well the split organizes the target values into homogeneous subgroups. This measure is used to assess the quality of the split.
- **Recursion**: If a split significantly increases the information gain (beyond a predefined threshold), the process is recursively applied to each of the resulting subsets. This recursive approach continues until no further significant information gain can be achieved.

For more details, see e.g. Section 7.2 in the book **Data mining: practical machine learning tools and techniques with Java implementations** by Ian H. Witten and Eibe Prank.

### Example

```python
import pandas as pd
from sklearn import datasets
from discretization import find_pivots

# Load the data set.
iris = datasets.load_iris()
df = pd.DataFrame(data= iris.data, columns= iris.feature_names)
df['target'] = iris.target

# Calculate pivots for the (numerical) feature "petal length (cm)".
pivots = find_pivots(df['petal length (cm)'], df['target']) # The pivots are 2.45 and 4.75.
```