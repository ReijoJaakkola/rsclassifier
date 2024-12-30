# Changelog

# v1.4.0 - (12/29/2024)
### Changes
- `booleanize_categorical_features` and `booleanize_numerical_features` are replaced by the `Booleanization` class.

# v1.3.3 - (12/22/2024)
- Better example for `booleanize_numerical_features`.

# v1.3.2 - (12/22/2024)
### Bugfix
- Fixed an issue with `booleanize_numerical_features`.

# v1.3.1 - (12/22/2024)
### Added
- **New Booleanization functions**: `booleanize_categorical_features` and `booleanize_numerical_features`. The first one converts categorical features into Boolean features using one-hot encoding style. The second one discretizes numerical features using entropy-based pivot points and converts them into Boolean features.

# v1.3.0 - (12/8/2024)
### Added
- **Brute force option for feature selection**: The `fit` function of `rsclassifier` has a new argument `fs_algorithm`, which can be used to change the algorithm used for selecting the features used by `rsclassifier`. The default option is `dt`, which uses decision trees for feature selection, while the second option is `brute` which finds the set of features for which the error on training data is minimized.

# v1.2.1 - (11/8/2024)
- Fixed the downloads badge on README.

# v1.2.0 - (11/5/2024)
### Added
- **Entropy-Based Discretization Module**: A new module named `discretization` has been introduced. It currently includes the `find_pivots` function, which supports entropy-based supervised discretization.

### Changes
- **Improved `find_pivots` Performance**: Enhancements have been made to the `find_pivots` function to improve its execution efficiency.

# v1.1.1 - (10/27/2024)
### Added
- Default value for `growth_size` is now 1.

### Bugfix
- Fixed an issue with rule pruning where setting `growth_size` to less than 1, then back to 1, would still trigger pruning due to the old pruning set being stored. Now, pruning will correctly be skipped when `growth_size` is reset to 1.

# v1.1 - (10/17/2024)
### Added
- Cross-validation is used for rule pruning.
- Support for Boolean features.
- Decision trees are used for feature selection.
- User is informed when num_prop is more than the total number of features.
- Show the total number of Boolean features after Booleanization.
- tqdm can be now disabled.