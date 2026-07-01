# Analytical Profiling (Summary)

The first step in any data science workflow is understanding the data you are working with. The **Analytical Profiling** methods in KPX Tabular provide a suite of tools to instantly evaluate the shape, data types, and structural health of your dataset without writing boilerplate pandas code.

## Available Methods at a Glance

**Basic Information Methods:**
- `shape()`
- `num_rows()`
- `num_columns()`
- `column_names()`
- `column_types()`
- `numerical_columns()`
- `bool_columns()`
- `categorical_columns()`
- `datetime_columns()`
- `memory_usage()`
- `dataset_info()`

**Core Methods:**
- `summary()`
- `feature_overview()`
- `column_profile()`
- `dataset_health()`
- `ml_readiness()`

---

## Basic Information Methods

The `Tabular` class exposes several lightweight utility methods to quickly retrieve fundamental attributes of your dataset.

### `shape()`
Returns a tuple `(rows, columns)` representing the dimensionality of the dataset.

**Example:**
```python
print(tab.shape())
# Output: (171184, 10)
```

---

### `num_rows()`
Returns the total number of rows (integer).

**Example:**
```python
print(tab.num_rows())
# Output: 171184
```

---

### `num_columns()`
Returns the total number of columns (integer).

**Example:**
```python
print(tab.num_columns())
# Output: 10
```

---

### `column_names()`
Returns a list of all column names in the dataset.

**Example:**
```python
print(tab.column_names()[:5])
# Output: ['order_id', 'order_date', 'order_year', 'order_month', 'order_day']
```

---

### `column_types()`
Returns a dictionary mapping each column name to its data type.

**Example:**
```python
print(tab.column_types())
# Output: {'order_id': 'str', 'order_date': 'str', 'order_year': 'int64', ...}
```

---

### `numerical_columns()`
Returns a tuple containing a list of numerical column names and their count.

**Example:**
```python
print(tab.numerical_columns())
# Output: (['order_year', 'order_month', 'order_day', 'order_hour', ...], 6)
```

---

### `bool_columns()`
Returns a tuple containing a list of boolean column names and their count.

**Example:**
```python
print(tab.bool_columns())
# Output: ([], 0)
```

---

### `categorical_columns()`
Returns a tuple containing a list of categorical (object/string) column names and their count.

**Example:**
```python
print(tab.categorical_columns())
# Output: (['order_id', 'order_date', 'is_weekend', 'order_status'], 4)
```

---

### `datetime_columns()`
Returns a tuple containing a list of datetime column names and their count.

**Example:**
```python
print(tab.datetime_columns())
# Output: ([], 0)
```

---

### `memory_usage()`
Returns a human-readable string (e.g., "47.35 MB") of the total deep memory usage.

**Example:**
```python
print(tab.memory_usage())
# Output: 47.35 MB
```

---

### `dataset_info()`
Returns a tuple `(rows, columns, total_cells)`.

**Example:**
```python
print(tab.dataset_info())
# Output: (171184, 10, 1711840)
```

---

## Core Methods

### `summary()`
Returns a high-level, overarching summary of the dataset as a pandas DataFrame. It provides quick answers to fundamental questions about dataset size and composition.

**What it returns:**
- Total rows and columns.
- Total number of cells and overall missing values (with percentages).
- Number of duplicate rows.
- Total memory usage (formatted in B, KB, MB, or GB).
- Counts of column data types (numeric, categorical, boolean, datetime).

**Example:**
```python
import pandas as pd
from kpx import Tabular

df = pd.read_csv("data.csv")
tab = Tabular(df)

tab_summary = tab.summary()
print(tab_summary)
```
**Output:**
```text
                        Value
rows                   171184
columns                    10
total cells           1711840
missing values              0
missing %                 0.0
duplicate rows              0
memory usage         47.35 MB
numeric columns             6
categorical columns         4
boolean columns             0
datetime columns            0
```

---

### `feature_overview()`
Provides a column-by-column breakdown of the entire dataset.

**What it returns:**
A pandas DataFrame where each row represents a feature, showing:
- Data type (`Dtype`).
- Total missing values and missing percentage.
- Number of unique values.
- Memory usage for that specific feature.

**Example:**
```python
print(tab.feature_overview().head(5).to_string())
```
**Output:**
```text
       Feature  Dtype  Missing  Missing %  Unique    Memory
0     order_id    str        0        0.0  170933   9.47 MB
1   order_date    str        0        0.0  171184  12.24 MB
2   order_year  int64        0        0.0       3   1.31 MB
3  order_month  int64        0        0.0      12   1.31 MB
4    order_day  int64        0        0.0      31   1.31 MB
```

---

### `column_profile(column_names=None)`
Generates a detailed profile for specific columns (or all columns if none are provided). This is similar to `feature_overview()` but allows for targeted inspection of a subset of features (or useful when inspecting the underlying profile dictionaries).

**Parameters:**
- `column_names` *(str or list, optional)*: The specific column name(s) to profile.

**Example:**
```python
print(tab.column_profile().head(5).to_string())
```
**Output:**
```text
          name  dtype  missing  missing_percent  unique    memory
0     order_id    str        0              0.0  170933   9.47 MB
1   order_date    str        0              0.0  171184  12.24 MB
2   order_year  int64        0              0.0       3   1.31 MB
3  order_month  int64        0              0.0      12   1.31 MB
4    order_day  int64        0              0.0      31   1.31 MB
```

---

### `dataset_health()`
An advanced diagnostic tool that evaluates the dataset and assigns it a **Health Score (0-100)** based on its structural integrity. 

**What it checks for:**
- Missing values and duplicate rows.
- Empty columns (columns where every row is an empty string `""`).
- Constant columns (columns with only one unique value).
- Mixed type columns (columns containing multiple data types, e.g., strings mixed with integers).
- Duplicate column names.
- Infinite values in numeric columns.

**What it returns:**
A dictionary containing the `Health Score`, a `Status` string (e.g., "Excellent", "Good", "Poor", "Critical"), and a detailed `Issues` breakdown identifying exactly which columns are problematic.

**Example:**
```python
import json

health_report = tab.dataset_health()
print(json.dumps(health_report, indent=2))
```
**Output:**
```json
{
  "Health Score": 100,
  "Status": "Excellent",
  "Issues": {
    "Missing Values": {
      "count": 0,
      "columns": []
    },
    "Duplicate Rows": {
      "count": 0
    },
    "Empty Columns": {
      "count": 0,
      "columns": []
    },
    "Constant Columns": {
      "count": 0,
      "columns": []
    },
    "Mixed Type Columns": {
      "count": 0,
      "columns": []
    },
    "Duplicate Column Names": {
      "count": 0,
      "columns": []
    },
    "Infinite Values": {
      "count": 0,
      "columns": []
    }
  }
}
```

---

### `ml_readiness()`
A quick check to determine if the dataset is structurally safe to be fed directly into a machine learning algorithm.

**What it returns:**
A dictionary containing a boolean flag `isSafe`. If the dataset is not safe (e.g., it contains missing values, infinite values, or mixed types), it will also return an `issues` dictionary detailing exactly what needs to be fixed before the data is ready for modeling.

**Example (When Safe):**
```python
ready = tab.ml_readiness()
print(json.dumps(ready, indent=2))
```
**Output:**
```json
{
  "isSafe": true
}
```

**Example (When Not Safe):**
```python
# Assuming a dataset that has structural issues
bad_tab = kpx.Tabular(pd.read_csv("messy_data.csv"))
not_ready = bad_tab.ml_readiness()
print(json.dumps(not_ready, indent=2))
```
**Output:**
```json
{
  "isSafe": false,
  "issues": {
    "Missing Values": {
      "count": 152,
      "columns": [
        "age",
        "income"
      ]
    },
    "Mixed Type Columns": {
      "count": 1,
      "columns": [
        "product_id"
      ]
    }
  }
}
```
