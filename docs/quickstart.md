# Quickstart Guide

Get up and running with **QPX (Quick processing Xpress)** in under 5 minutes. This guide will walk you through the fastest way to profile, diagnose, and clean a dataset using the flagship `Tabular` module.

---

## 1. Initialization

QPX is built to work seamlessly with `pandas`. To get started, simply load your dataset as a pandas DataFrame and pass it to the `Tabular` wrapper class.

```python
import pandas as pd
from qpx import Tabular

# 1. Load your raw data
df = pd.read_csv("my_messy_data.csv")

# 2. Initialize the QPX Tabular environment
tab = Tabular(df)
```

---

## 2. Understand Your Data Instantly

Before cleaning data, you need to understand what's wrong with it. QPX provides two powerful methods to give you an instant bird's-eye view of your dataset.

### The High-Level Summary
Use the `summary()` method to get a quick snapshot of your dataset's shape, memory footprint, and column types.

```python
print(tab.summary())
```
**Output:**
```text
                        Value
rows                   171184
columns                    10
total cells           1711840
missing values           1402
missing %                 0.8
duplicate rows             12
memory usage         47.35 MB
numeric columns             6
categorical columns         4
```

### The Health Diagnostic
Use `dataset_health()` to scan for structural issues (like constant columns, massive missing values, or mixed types) and receive an overall **Health Score (0-100)**.

```python
import json
health_report = tab.dataset_health()
print(json.dumps(health_report, indent=2))
```
**Output:**
```json
{
  "Health Score": 85,
  "Status": "Good",
  "Issues": {
    "Missing Values": {
      "count": 1402,
      "columns": ["age", "income"]
    },
    "Duplicate Rows": {
      "count": 12
    }
  }
}
```

---

## 3. The "One-Liner" Preprocessing Pipeline

You don't need to write hundreds of lines of code to handle missing values, encode text, and compress memory. 

The **`auto_preprocess()`** method is QPX's flagship pipeline. It automatically runs 9 standard data science cleaning steps in the correct order, returning a clean DataFrame and a comprehensive log of exactly what was changed. After applying this function, also check the dataset again and modify as per your need.

```python
# Run the automated pipeline
df_clean, report = tab.auto_preprocess(return_report=True)

# Your data is now clean, encoded, and memory-optimized!
print("Original Shape:", report["initial_shape"])
print("Final Shape:", report["final_shape"])
print("Memory Saved:", report["memory_saved_mb"], "MB")
```

> [!TIP]
> **What just happened?**
> By default, `auto_preprocess()` automatically:
> 1. Sanitizes column names.
> 2. Fixes disguised numeric columns.
> 3. Drops constant/zero-variance features.
> 4. Drops high-cardinality features.
> 5. Drops highly multicollinear features.
> 6. Converts Yes/No strings to Booleans.
> 7. Smart-fills missing values (NaNs).
> 8. Auto-encodes remaining categorical features.
> 9. Downcasts memory footprint.

---

## Next Steps

You now know how to run the automated pipeline! 

However, QPX is highly configurable. If you want to disable certain steps in the pipeline, tweak the correlation thresholds, or run statistical methods manually, check out the detailed documentation:

- [Analytical Profiling (Summary)](user_guide/tabular/summary.md)
- [Statistical Diagnostics (Statistics)](user_guide/tabular/statistics.md)
- [Automated Preprocessing (Preprocessing)](user_guide/tabular/preprocessing.md)
