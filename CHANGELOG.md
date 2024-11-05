# Changelog

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