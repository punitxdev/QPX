# API Reference: `qpx.Tabular`

This document serves as the comprehensive API reference for the `qpx.Tabular` wrapper class. Below you will find the exact method signatures, parameters, and return types for every function available.

---

## 1. Class Initialization

### `__init__(self, df)`
Initializes the Tabular wrapper around a pandas DataFrame.
- **`df`** *(pd.DataFrame)*: The raw dataset to process.

---

## 2. Analytical Profiling (`summary`)

### `shape()`
- **Returns**: *(tuple)* A tuple containing `(num_rows, num_columns)`.

### `num_rows()` / `num_columns()`
- **Returns**: *(int)* The integer count of rows or columns.

### `column_names()`
- **Returns**: *(list)* A list of all column headers in the DataFrame.

### `column_types()`
- **Returns**: *(dict)* A mapping of `{column_name: string_dtype}`.

### `numerical_columns()` / `bool_columns()` / `categorical_columns()` / `datetime_columns()`
- **Returns**: *(list)* A list of column names that match the specified data type.

### `memory_usage()`
- **Returns**: *(str)* A human-readable string representing the memory footprint (e.g., `"15.2 MB"`).

### `dataset_info()`
- **Returns**: *(dict)* A lightweight dictionary containing core metrics (rows, columns, missing count, memory).

### `summary()`
- **Returns**: *(dict)* A higher-level summary aggregating missing percentages, categorical counts, and numerical counts.

### `feature_overview()`
- **Returns**: *(pd.DataFrame)* A detailed DataFrame where the index represents your columns and the columns contain metadata (Missing %, Unique values, Memory usage).

### `dataset_health()`
- **Returns**: *(dict)* A health scorecard analyzing critical flaws like zero-variance columns and high missing rates.

### `ml_readiness()`
- **Returns**: *(dict)* A strict pass/fail assessment determining if the data can be fed into an ML algorithm (checks for nulls, non-numeric types, etc.).

### `column_profile(column_names=None)`
- **`column_names`** *(list, optional)*: Specific columns to profile. If None, profiles all.
- **Returns**: *(dict)* A deep statistical and structural dictionary profile for the specified features.

---

## 3. Statistical Diagnostics (`statistics`)

### `numeric_summary()`
- **Returns**: *(pd.DataFrame)* Statistical summary of numerical data (mean, median, std, skewness, kurtosis, IQR).

### `categorical_summary()`
- **Returns**: *(pd.DataFrame)* Statistical summary of categorical data (mode, unique count, top frequency).

### `outlier_summary(detailed=False)`
- **`detailed`** *(bool, default=False)*: If True, returns a dictionary containing the actual outlier index locations. If False, returns a DataFrame with outlier counts.
- **Returns**: *(pd.DataFrame or dict)* Outlier assessment based on IQR bounds.

### `multicollinearity_report()`
- **Returns**: *(pd.DataFrame)* Variance Inflation Factor (VIF) for all numerical features to detect multicollinearity.
- **Raises**: `ValueError` if the dataset contains less than two numerical features.

### `statistical_snapshot()`
- **Returns**: *(dict)* A master dictionary combining `numeric_summary`, `categorical_summary`, and `outlier_summary`.

---

## 4. Data Preprocessing (`preprocessing`)

### `sanitize_col_names(case="snake", remove_special_chars=True, replace_spaces="_", ensure_unique=True, return_mapping=False)`
- **Parameters**: Modifies column headers to be machine-readable.
- **Returns**: *(dict or None)* Mapping of old to new names if requested. Modifies DataFrame in place.

### `boolify(columns=None, true_value=True, false_value=False, return_info=False, inplace=False)`
- **Parameters**: Casts pseudo-boolean object columns into strict boolean types.
- **Returns**: *(dict or DataFrame)* Depending on `inplace` and `return_info`.
- **Raises**: `KeyError` if an explicitly provided column is not found in the DataFrame.

### `drop_constant(columns=None, ignore_na=True, return_info=False, inplace=False)`
- **Parameters**: Drops columns with zero variance (only 1 unique value).
- **Returns**: *(dict or DataFrame)*.

### `label_encode(columns=None, inplace=False, return_mapping=False)`
- **Parameters**: Numerically encodes categorical text variables using numerical labels.
- **Returns**: *(dict or DataFrame)*.
- **Raises**: `KeyError` if an explicitly provided column is not found in the DataFrame.

### `one_hot_encode(columns=None, inplace=False, return_info=False, drop_original=False, max_unique=20)`
- **Parameters**: Creates binary dummy variables for categories up to `max_unique` threshold.
- **Returns**: *(dict or DataFrame)*.
- **Raises**: `KeyError` if an explicitly provided column is not found in the DataFrame.

### `ordinal_encode(order, inplace=False, return_mapping=True, strict=True)`
- **Parameters**: `order` *(dict)* Mapping of `{column: [hierarchical_list]}`.
- **Returns**: *(dict or DataFrame)*.
- **Raises**: `KeyError` if a column in `order` is not found. `ValueError` if `strict=True` and unknown categories are encountered.

### `auto_encode(columns=None, order=None, max_onehot=10, inplace=False, return_report=False)`
- **Parameters**: Intelligently decides between one-hot and label encoding based on the cardinality of the column.
- **Returns**: *(dict or DataFrame)*.
- **Raises**: `KeyError` if an explicitly provided column is not found in the DataFrame.

### `compress_dtype(columns=None, downcast_int=True, downcast_float=True, convert_category=True, category_threshold=0.5, inplace=False, return_report=False)`
- **Parameters**: Slashes memory footprint by safely downcasting bit-precision (e.g., float64 to float32).
- **Returns**: *(dict or DataFrame)*.
- **Raises**: `KeyError` if an explicitly provided column is not found in the DataFrame.

### `smart_fill(columns=None, strategy="auto", value=None, missing_threshold=0.6, inplace=False, return_report=False)`
- **Parameters**: Imputes NaNs using median for numerics, mode for categoricals, or drops columns exceeding `missing_threshold`.
- **Returns**: *(dict or DataFrame)*.
- **Raises**: `KeyError` if an explicitly provided column is not found in the DataFrame.

### `drop_high_cardinality(columns=None, threshold=0.8, return_info=False, inplace=False)`
- **Parameters**: Drops nominal columns like IDs/Names where unique count / total rows > `threshold`.
- **Returns**: *(dict or DataFrame)*.
- **Raises**: `KeyError` if an explicitly provided column is not found in the DataFrame.

### `drop_highly_correlated(threshold=0.9, method="pearson", target_col=None, return_info=False, inplace=False)`
- **Parameters**: Prevents multicollinearity by dropping one feature from pairs that correlate above `threshold`. If `target_col` is provided, keeps the feature most correlated with the target.
- **Returns**: *(dict or DataFrame)*.

### `auto_preprocess(...)`
```python
auto_preprocess(
    target_col=None, sanitize=True, cast_numeric=True, drop_const=True, 
    drop_high_cardinality_cols=False, drop_correlated_cols=False, 
    boolify_cols=True, fill_nan=True, encode=True, compress=True, 
    missing_threshold=0.6, high_cardinality_threshold=0.8, 
    correlation_threshold=0.9, max_onehot=10, return_report=False
)
```
- **Returns**: *(dict or DataFrame)* The flagship automated data cleaning pipeline.

---

## 5. Data Visualization (`visuals`)

*Note: All visualization methods render directly using matplotlib/seaborn and do not return data objects.*

### `corr_map(target=None, ignore_cols=None, method="spearman")`
- **`target`** *(str, optional)*: Specific target column to isolate correlations against.
- **`ignore_cols`** *(list, optional)*: Columns to skip.
- **`method`** *(str, default="spearman")*: Correlation method (`pearson`, `kendall`, `spearman`).
- **Raises**: `ValueError` if `target` is provided but is not a numeric column.

### `pca_plot(input_cols=None, target=None, n_components=2, sample_space=None, figsize=(8, 6))`
- **`target`** *(str, optional)*: Target column used to color-code clusters.
- **`n_components`** *(int, default=2)*: Accepts `2` or `3` for 2D or 3D PCA plotting.
- **`sample_space`** *(int, optional)*: Randomly subsamples large datasets to speed up rendering.
- **Raises**: `ValueError` for invalid `n_components`, missing `target` columns, insufficient numerical features, or invalid `sample_space`.

### `relationship_map(target, input_cols=None, ignore_cols=None, sample_space=1000, figsize=(9, 7))`
- **`target`** *(str, required)*: The outcome variable to compare features against.
- **`input_cols`** *(list, optional)*: The subset of features to compare.
- **Raises**: `ValueError` if the specified `target` column is not found in the DataFrame.

### `distribution_map(cols=None, ignore_cols=None, sample_space=1000, figsize=(8, 6))`
- **`cols`** *(list, optional)*: Features to plot distributions for. Built-in logic ignores high-cardinality IDs automatically.

### `feature_cluster_map(ignore_cols=None, method="spearman", figsize=(10, 10))`
- **Parameters**: Generates a hierarchically-clustered heatmap. Excellent for visually spotting multicollinearity.
- **Raises**: `ValueError` if the DataFrame contains fewer than 2 numeric features.
