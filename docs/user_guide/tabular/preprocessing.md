# Automated Preprocessing (Preprocessing)

Data cleaning and feature engineering are often the most time-consuming parts of the data science workflow. The **Automated Preprocessing** methods in QPX Tabular provide robust, production-ready pipelines that automatically handle everything from column sanitization and type casting to missing value imputation, categorical encoding, and memory compression.

## Available Methods at a Glance

**Preprocessing Methods:**
- `sanitize_col_names()`
- `boolify()`
- `drop_constant()`
- `drop_high_cardinality()`
- `drop_highly_correlated()`
- `label_encode()`
- `one_hot_encode()`
- `ordinal_encode()`
- `auto_encode()`
- `compress_dtype()`
- `smart_fill()`
- `auto_preprocess()` (The all-in-one pipeline)

---

## Preprocessing Methods

### `sanitize_col_names()`
Cleans up and standardizes column names. It removes special characters, strips leading/trailing whitespace, and formats the names.

**Parameters:**
- `case` *(str, default="snake")*: Casing strategy (e.g., `"snake"`, `"camel"`, `"pascal"`, `"lower"`, `"upper"`).
- `remove_special_chars` *(bool, default=True)*: If True, strips non-alphanumeric characters.
- `replace_spaces` *(str, default="_")*: Character to replace spaces with.
- `ensure_unique` *(bool, default=True)*: Ensures no duplicate column names exist after sanitization.
- `return_mapping` *(bool, default=False)*: If True, returns a dictionary mapping old names to new names.

**Example:**
```python
import pandas as pd
import json
from qpx import Tabular

df = pd.read_csv("data.csv")
tab = Tabular(df)

df_clean, mapping = tab.sanitize_col_names(case="snake", return_mapping=True)
print(json.dumps(mapping, indent=2))
```
**Output:**
```json
{
  "order_id": "order_id",
  "order_date": "order_date",
  "order_year": "order_year"
}
```

---

### `boolify()`
Automatically detects columns that only contain two unique string values (like "Yes"/"No" or "True"/"False") and converts them into proper boolean types (`True`/`False`).

**Parameters:**
- `columns` *(list, optional)*: Specific columns to boolify. If None, checks all string columns.
- `true_value` *(any, default=True)*: Custom value to map to `True`.
- `false_value` *(any, default=False)*: Custom value to map to `False`.
- `return_info` *(bool, default=False)*: If True, returns a list of columns that were successfully boolified.

**Example:**
```python
df_bool, info = tab.boolify(return_info=True)
print(json.dumps(info, indent=2))
```
**Output:**
```json
[
  "is_weekend"
]
```

---

### `drop_constant()`
Detects and drops "constant" columns—columns where every single row has the exact same value. These columns provide zero predictive power to machine learning models.

**Parameters:**
- `columns` *(list, optional)*: Specific columns to check. If None, checks all columns.
- `ignore_na` *(bool, default=True)*: If True, ignores NaN values when determining if a column is constant.
- `return_info` *(bool, default=False)*: If True, returns a list of dropped columns.

**Example:**
```python
df_dropped, info = tab.drop_constant(return_info=True)
print(json.dumps(info, indent=2))
```
**Output:**
```json
[]
```

---

### `drop_high_cardinality()`
Automatically identifies and drops categorical columns that have too many unique values relative to the size of the dataset (like User IDs or Timestamps). High cardinality features often lead to overfitting.

**Parameters:**
- `columns` *(list, optional)*: Specific columns to check. If None, checks all object/category columns.
- `threshold` *(float, default=0.8)*: The maximum allowed ratio of unique values to total rows. For example, `0.8` means a column is dropped if 80% or more of its values are completely unique.
- `return_info` *(bool, default=False)*: If True, returns a list of dropped columns.

**Example:**
```python
df_clean, info = tab.drop_high_cardinality(threshold=0.8, return_info=True)
print(json.dumps(info, indent=2))
```
**Output:**
```json
[
  "order_id",
  "order_date"
]
```

---

### `drop_highly_correlated()`
Evaluates numerical columns and drops features that are highly correlated with one another, preventing multicollinearity which can destabilize machine learning models.

**Parameters:**
- `threshold` *(float, default=0.9)*: The correlation coefficient threshold above which a feature is dropped.
- `method` *(str, default="pearson")*: Correlation method (`"pearson"`, `"kendall"`, `"spearman"`).
- `return_info` *(bool, default=False)*: If True, returns a list of dropped columns.

**Example:**
```python
df_clean, info = tab.drop_highly_correlated(threshold=0.9, return_info=True)
print(json.dumps(info, indent=2))
```
**Output:**
```json
[
  "order_year_copy"
]
```

---

### `label_encode()`
Manually applies Label Encoding to specific categorical columns. It assigns a unique integer to each distinct category (e.g., "Red" -> 0, "Blue" -> 1).

**Parameters:**
- `columns` *(list, optional)*: Columns to encode. If None, encodes all categorical columns.
- `return_mapping` *(bool, default=False)*: If True, returns the dictionary showing exactly how values were mapped to integers.

**Example:**
```python
df_encoded, mapping = tab.label_encode(columns=["is_weekend"], return_mapping=True)
print(json.dumps(mapping, indent=2))
```
**Output:**
```json
{
  "is_weekend": {
    "No": 0,
    "Yes": 1
  }
}
```

---

### `one_hot_encode()`
Manually applies One-Hot Encoding to specific categorical columns. It creates a new binary column (0 or 1) for every unique category found in the original column.

**Parameters:**
- `columns` *(list, optional)*: Columns to encode. If None, attempts to encode all categorical columns.
- `drop_original` *(bool, default=False)*: If True, removes the original categorical column after encoding.
- `max_unique` *(int, default=20)*: Safety limit. If a column has more unique categories than this, it will be skipped to prevent memory explosion.
- `return_info` *(bool, default=False)*: If True, returns a report detailing what was encoded and what was skipped.

**Example:**
```python
df_encoded, info = tab.one_hot_encode(columns=["is_weekend"], return_info=True)
print(json.dumps(info, indent=2))
```
**Output:**
```json
{
  "encoded": [
    "is_weekend"
  ],
  "not_encoded": []
}
```

---

### `ordinal_encode()`
Manually encodes categorical columns based on a strict, predefined hierarchy or order provided by the user (e.g., "Low" -> 0, "Medium" -> 1, "High" -> 2).

**Parameters:**
- `order` *(dict)*: Required. A dictionary mapping column names to a list of ordered categories. Example: `{"size": ["small", "medium", "large"]}`.
- `return_mapping` *(bool, default=True)*: If True, returns the exact mapping dictionary used.
- `strict` *(bool, default=True)*: If True, throws an error if unknown categories are encountered that were not defined in your `order` dictionary.

**Example:**
```python
order_dict = {"is_weekend": ["No", "Yes"]}
df_encoded, mapping = tab.ordinal_encode(order=order_dict, return_mapping=True)
print(json.dumps(mapping, indent=2))
```
**Output:**
```json
{
  "is_weekend": {
    "no": 0,
    "yes": 1
  }
}
```

---

### `auto_encode()`
A smart, automated feature encoder. Instead of manually applying different encodings to different columns, this method analyzes the cardinality of each categorical column and automatically applies the most appropriate strategy.

**Parameters:**
- `columns` *(list, optional)*: Specific columns to encode. If None, encodes all categorical columns.
- `order` *(dict, optional)*: Predefined ordinal mapping dict to apply first (falls back to automatic for remaining columns).
- `max_onehot` *(int, default=10)*: Maximum cardinality a column can have to be eligible for One-Hot Encoding. Above this, it uses Label Encoding instead.
- `return_report` *(bool, default=False)*: If True, returns a detailed breakdown of which strategy was applied to which column.

**Example:**
```python
df_encoded, report = tab.auto_encode(return_report=True)
print(json.dumps(report, indent=2))
```
**Output:**
```json
{
  "label_encoded": [
    "is_weekend"
  ],
  "one_hot_encoded": [],
  "ordinal_encoded": [],
  "skipped": []
}
```

---

### `compress_dtype()`
Optimizes the memory footprint of your dataset. It safely downcasts large integer and float types (e.g., `int64` to `int8`) and converts low-cardinality string columns into pandas `category` types.

**Parameters:**
- `columns` *(list, optional)*: Specific columns to compress. If None, attempts to compress all columns.
- `downcast_int` *(bool, default=True)*: Downcasts integers to their smallest possible format (int8, int16, etc).
- `downcast_float` *(bool, default=True)*: Downcasts floats to their smallest possible format (float32).
- `convert_category` *(bool, default=True)*: Converts strings to `category` types if cardinality is low enough.
- `category_threshold` *(float, default=0.5)*: The cardinality threshold below which string columns are converted to categories.
- `return_report` *(bool, default=False)*: If True, returns a breakdown of memory saved and exactly which data types were changed.

**Example:**
```python
df_compressed, report = tab.compress_dtype(return_report=True)
print(json.dumps(report, indent=2))
```
**Output:**
```json
{
  "memory_before_compression": 47.35,
  "memory_after_compression": 23.18,
  "saved_mb": 24.17,
  "saved_percent": 51.04,
  "converted": {
    "order_year": {
      "before": "int64",
      "after": "int16"
    },
    "is_weekend": {
      "before": "str",
      "after": "category"
    }
  }
}
```

---

### `smart_fill()`
Automatically imputes (fills in) missing values across the entire dataset. It intelligently chooses the best strategy based on the data type.

**Parameters:**
- `columns` *(list, optional)*: Specific columns to target. If None, targets all columns with missing values.
- `strategy` *(str, default="auto")*: Imputation strategy (`"auto"`, `"mean"`, `"median"`, `"mode"`, `"constant"`). `"auto"` uses mean for numeric and mode for categorical.
- `value` *(any, optional)*: Required if `strategy="constant"`. The value to fill NaNs with.
- `missing_threshold` *(float, default=0.6)*: If a column has a missing percentage higher than this (e.g., > 60% missing), it will be recommended for dropping instead of filling.
- `return_report` *(bool, default=False)*: If True, returns a report of what was filled and what strategy was used.

**Example:**
```python
df_filled, report = tab.smart_fill(return_report=True)
print(json.dumps(report, indent=2))
```
**Output:**
```json
{
  "filled": {
    "order_year": {
      "semantic_type": "numeric",
      "strategy": "mean",
      "filled": 1
    }
  },
  "skipped": {},
  "recommended_drop": {}
}
```

---

### `auto_preprocess()`
The flagship preprocessing method of QPX Tabular. It runs an end-to-end, highly configurable preprocessing pipeline on your dataset. After applying this function, also check the dataset again and modify as per your need.

> [!IMPORTANT]
> **The `auto_preprocess` Pipeline runs in this exact order:**
> 1. Sanitizes column names.
> 2. Force-casts disguised numerics (numbers stored as strings).
> 3. Drops constant (zero variance) features.
> 4. Drops high-cardinality features (like unique IDs).
> 5. Drops highly correlated numerical features (to fix multicollinearity).
> 6. Boolifies binary string columns.
> 7. Smart-fills missing values (NaNs).
> 8. Auto-encodes remaining categorical features.
> 9. Compresses memory usage via downcasting.

**Parameters:**
You can toggle any step of the pipeline on or off, and tweak the thresholds used in those steps:
- `sanitize` *(bool, default=True)*: Toggle column sanitization.
- `cast_numeric` *(bool, default=True)*: Toggle numeric type casting.
- `drop_const` *(bool, default=True)*: Toggle dropping constant columns.
- `drop_high_cardinality_cols` *(bool, default=True)*: Toggle dropping high-cardinality columns.
- `drop_correlated_cols` *(bool, default=True)*: Toggle dropping multicollinear columns.
- `boolify_cols` *(bool, default=True)*: Toggle boolean conversion.
- `fill_nan` *(bool, default=True)*: Toggle missing value imputation.
- `encode` *(bool, default=True)*: Toggle automated categorical encoding.
- `compress` *(bool, default=True)*: Toggle memory downcasting.
- `missing_threshold` *(float, default=0.6)*: Threshold for dropping missing columns.
- `high_cardinality_threshold` *(float, default=0.8)*: Threshold for dropping unique categories.
- `correlation_threshold` *(float, default=0.9)*: Threshold for dropping multicollinear features.
- `max_onehot` *(int, default=10)*: Safety limit for One-Hot encoding.
- `return_report` *(bool, default=False)*: If True, returns the comprehensive transformation log.

**Example:**
```python
df_final, report = tab.auto_preprocess(return_report=True)
print(json.dumps(report, indent=2))
```
**Output:**
```json
{
  "initial_health": 100.0,
  "sanitized": {
    "order_id": "order_id"
  },
  "casted_to_numeric": [],
  "dropped_constant": [],
  "dropped_high_cardinality": [
    "order_id",
    "order_date"
  ],
  "dropped_correlated": [],
  "boolified": [
    "is_weekend"
  ],
  "nans_imputed_cols": {},
  "nans_imputed_count": 0,
  "encoded": {
    "label_encoded": [],
    "one_hot_encoded": [],
    "ordinal_encoded": [],
    "skipped": []
  },
  "memory_saved_mb": 24.17,
  "initial_shape": [171184, 10],
  "final_shape": [171184, 8],
  "final_health": 100.0
}
```
