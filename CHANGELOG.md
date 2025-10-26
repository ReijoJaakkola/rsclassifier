# Changelog

## v2.0.0 - (TBD)
### Breaking Changes
- **Strict pandas requirement**: `RuleSetClassifier` now requires pandas DataFrame for X and pandas Series for y. NumPy arrays are no longer automatically converted. This ensures proper feature name handling for categorical, numerical, and boolean features.
- **sklearn-compatible API**: Complete redesign to follow scikit-learn conventions:
  - Extends `BaseEstimator` and `ClassifierMixin`
  - All hyperparameters moved to `__init__()`
  - Removed `load_data()` method - data is now passed directly to `fit()`
  - Feature types (`boolean_features`, `categorical_features`, `numerical_features`) are now constructor parameters
  - Fitted attributes use trailing underscore convention (e.g., `rules_`, `semantics_`)

### Added
- **Model persistence support**:
  - `save_model()` and `load_model()` methods for convenient model saving/loading
  - `save_rules_as_text()` for exporting human-readable rules
  - `save_rules_as_json()` for programmatic rule access with metadata
  - Full pickle/joblib compatibility via `__getstate__()` and `__setstate__()`
- **sklearn integration**:
  - Works with `GridSearchCV`, `cross_val_score`, and pipelines
  - Proper `get_params()` and `set_params()` implementation
  - Compatible with all sklearn model selection tools

### Migration Guide
**Old API (v1.x):**
```python
rsc = RuleSetClassifier()
rsc.load_data(X=X_train, y=y_train, numerical=X.columns)
rsc.fit(num_prop=2)
```

**New API (v2.0):**
```python
rsc = RuleSetClassifier(
    num_prop=2,
    numerical_features=X.columns.tolist()
)
rsc.fit(X_train, y_train)
```

## v1.5.2 - (3/17/2025)
### Bugfix
- Fixed a bug in entropy based discretization.

## v1.5.2 - (3/15/2025)
### Changes
- Significant improvements in the performance of learning, discretization, evaluation and pruning.

### Bugfix
- Fixed a bug in pruning.

## v1.5.1 - (3/10/2025)
### Changes
- Improved README.

## v1.5.0 - (3/1/2025)
### Added
- **Better pruning of terms**: Literals that do not contribute to training accuracy are removed now from the literals.

## v1.4.2 - (2/27/2025)
### Bugfix
- Fixed a bug in booleanization.
- In `rsclassifier` setting `silent = True` also suppresses warnings.

## v1.4.1 - (1/5/2025)
### Bugfix
- Fixed the downloads badge and typos on README.

## v1.4.0 - (12/30/2024)
### Changes
- `booleanize_categorical_features` and `booleanize_numerical_features` are replaced by the `Booleanization` class.

## v1.3.3 - (12/22/2024)
### Changes
- Better example for `booleanize_numerical_features`.

## v1.3.2 - (12/22/2024)
### Bugfix
- Fixed an issue with `booleanize_numerical_features`.

## v1.3.1 - (12/22/2024)
### Added
- **New Booleanization functions**: `booleanize_categorical_features` and `booleanize_numerical_features`. The first one converts categorical features into Boolean features using one-hot encoding style. The second one discretizes numerical features using entropy-based pivot points and converts them into Boolean features.

## v1.3.0 - (12/8/2024)
### Added
- **Brute force option for feature selection**: The `fit` function of `rsclassifier` has a new argument `fs_algorithm`, which can be used to change the algorithm used for selecting the features used by `rsclassifier`. The default option is `dt`, which uses decision trees for feature selection, while the second option is `brute` which finds the set of features for which the error on training data is minimized.

## v1.2.1 - (11/8/2024)
### Bugfix
- Fixed the downloads badge on README.

## v1.2.0 - (11/5/2024)
### Added
- **Entropy-Based Discretization Module**: A new module named `discretization` has been introduced. It currently includes the `find_pivots` function, which supports entropy-based supervised discretization.

### Changes
- **Improved `find_pivots` Performance**: Enhancements have been made to the `find_pivots` function to improve its execution efficiency.

## v1.1.1 - (10/27/2024)
### Added
- Default value for `growth_size` is now 1.

### Bugfix
- Fixed an issue with rule pruning where setting `growth_size` to less than 1, then back to 1, would still trigger pruning due to the old pruning set being stored. Now, pruning will correctly be skipped when `growth_size` is reset to 1.

## v1.1 - (10/17/2024)
### Added
- Cross-validation is used for rule pruning.
- Support for Boolean features.
- Decision trees are used for feature selection.
- User is informed when num_prop is more than the total number of features.
- Show the total number of Boolean features after Booleanization.
- tqdm can be now disabled.