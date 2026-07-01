# KPX Tabular: Overview

Welcome to the **KPX Tabular** user guide! 

The `kpx.tabular` module is the flagship component of the KPX framework, specifically engineered to take the pain out of tabular data analysis and preparation. Whether you are dealing with a messy CSV full of hidden data types or preparing a clean DataFrame for a machine learning model, KPX Tabular provides the tools you need.

## The `Tabular` Class

At the heart of this module is the `Tabular` class. It acts as a wrapper around your `pandas` DataFrame, exposing a wide array of methods categorized into three main pillars:

1. **Analytical Functions**: High-level overviews to assess the structural health of your dataset.
2. **Statistical Functions**: Mathematical diagnostics to understand distributions, correlations, and outliers.
3. **Preprocessing Functions**: Powerful, automated tools to clean, encode, and compress your data.

### Initialization

To start using KPX Tabular, simply import the class and pass in your `pandas.DataFrame`.

```python
import pandas as pd
from kpx import Tabular

# 1. Load your raw data
df = pd.read_csv('my_messy_dataset.csv')

# 2. Initialize the Tabular instance
tab = Tabular(df)
```

Upon initialization, the `Tabular` class automatically categorizes your DataFrame's columns by data type (numeric, categorical, and boolean) so that downstream functions can operate efficiently on the relevant subsets of data.

---

## The Three Pillars of KPX Tabular

### 1. Analytical Profiling
Before cleaning your data, you need to understand it. The analytical functions provide a bird's-eye view of your dataset. You can quickly check for missing values, duplicate rows, empty columns, and assess whether your dataset is structurally ready for machine learning.
- *Key Methods*: `dataset_health()`, `ml_readiness()`, `column_profile()`

### 2. Statistical Diagnostics
Go deeper into the numbers. These methods allow you to pinpoint highly correlated features (multicollinearity), identify statistical outliers, and get detailed summaries of both your numeric and categorical distributions.
- *Key Methods*: `numeric_summary()`, `multicollinearity_report()`, `outlier_summary()`

### 3. Automated Preprocessing
The most powerful feature of KPX Tabular. Instead of writing dozens of lines of pandas code to handle string manipulation, missing value imputation, and categorical encoding, you can run a single automated pipeline or invoke specific transformations manually.
- *Key Methods*: `auto_preprocess()`, `smart_fill()`, `auto_encode()`, `compress_dtype()`

## What's Next?

In the following sections of the user guide, we will dive deep into each of these three pillars, providing comprehensive examples and explaining the parameters you can tune to perfectly fit your data workflow.
