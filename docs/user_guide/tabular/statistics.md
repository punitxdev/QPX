# Statistical Diagnostics (Statistics)

Once you understand the basic shape of your data, the next step is to dive deeper into the mathematical and categorical properties of your features. The **Statistical Diagnostics** methods in QPX Tabular provide detailed metrics, distribution insights, and collinearity checks to help you understand the statistical health of your dataset.

## Available Methods at a Glance

**Statistical Methods:**
- `statistical_snapshot()`
- `numeric_summary()`
- `categorical_summary()`
- `outlier_summary()`
- `multicollinearity_report()`

---

## Statistical Methods

### `statistical_snapshot()`
Returns a consolidated, high-level snapshot of key dataset metrics, ranging from size and memory usage to missing values, cardinality, and mixed types.

> [!NOTE] 
> **Understanding Cardinality:** Cardinality simply refers to the number of *unique* values in a column. An "Average Cardinality" metric tells you whether your categorical columns are mostly repetitive (like True/False) or highly distinct (like User IDs). High cardinality features often require special encoding techniques before machine learning.

**Example:**
```python
import pandas as pd
from qpx import Tabular

df = pd.read_csv("data.csv")
tab = Tabular(df)

print(tab.statistical_snapshot())
```
**Output:**
```text
                          Value
Metric                         
Rows                  171184.00
Columns                   10.00
Memory Usage (MB)         47.35
Numeric Columns            6.00
Categorical Columns        4.00
Boolean Columns            0.00
Datetime Columns           0.00
Total Missing Values       0.00
Missing Percentage         0.00
Columns with Missing       0.00
Duplicate Rows             0.00
Duplicate Columns          0.00
Empty Columns              0.00
Constant Columns           0.00
Infinite Values            0.00
Mixed-Type Columns         0.00
Average Missing %          0.00
Average Cardinality    64188.70
Average Unique %          18.73
```

---

### `numeric_summary()`
Generates a comprehensive statistical summary for all numerical columns in the dataset. 

**What it returns:**
A pandas DataFrame detailing standard metrics like Mean, Median, and Mode, alongside more complex statistical indicators:

> [!TIP]
> **Plain English Glossary:**
> - **Variance & Std (Standard Deviation):** Measures how spread out the numbers are. A low standard deviation means most numbers are very close to the average.
> - **CV (%) (Coefficient of Variation):** The standard deviation divided by the mean. It's a great way to compare the volatility of two different columns (e.g., age vs. income) on an equal playing field.
> - **Skewness:** Measures if your data leans to one side. If it's near `0`, your data is perfectly symmetrical (a bell curve). A positive skew means a long tail of high values (like wealth distribution), while a negative skew means a long tail of small values.

**Example:**
```python
print(tab.numeric_summary().head(3).to_string())
```
**Output:**
```text
       Feature         Mean  Median  Mode       Std   Variance   Min      Q1   Max  Range     CV (%)  Skewness
0   order_year  2024.588969  2025.0  2025  0.577269   0.333240  2024  2024.0  2026      2   0.028513  0.360380
1  order_month     6.502442     7.0     1  3.453728  11.928235     1     3.0    12     11  53.114318 -0.003755
2    order_day    15.781539    16.0    18  8.813376  77.675594     1     8.0    31     30  55.846111  0.000365
```

---

### `categorical_summary()`
Provides a detailed breakdown of all categorical (text-based or bucketed) columns in the dataset.

> [!NOTE] 
> **Understanding Entropy:** Entropy is a measure of randomness or unpredictability. 
> - **High Entropy:** The categories are evenly distributed (e.g., 50% apples, 50% oranges). It means the column contains a lot of unpredictable information.
> - **Low Entropy:** The column is highly predictable and one category dominates (e.g., 99% apples, 1% oranges). If entropy is `0`, it's a constant column with only one value.

**What it returns:**
A pandas DataFrame showing counts, missing values, unique cardinalities, the most frequent value (`mode`) and its percentage footprint (`mode %`), data `entropy`, and whether the feature is strictly `binary` (only 2 unique values).

**Example:**
```python
print(tab.categorical_summary().head(3).to_string())
```
**Output:**
```text
           Dtype   count  missing  missing %  unique                        mode  mode %    entropy  binary
order_id     str  171184        0        0.0  170933                   ORD-0689T    0.00  17.382256   False
order_date   str  171184        0        0.0  171184  2024-02-03 04:36:20.378297    0.00  17.385188   False
is_weekend   str  171184        0        0.0       2                          No   71.24   0.865635    True
```
*(In this example, notice how `is_weekend` has very low entropy (0.86) because it's a binary Yes/No field where "No" dominates 71% of the time, while `order_id` has massive entropy because almost every single row is a unique string).*

---

### `outlier_summary(detailed=False)`
Detects and summarizes **statistical outliers** in all numerical columns.

> [!TIP]
> **What is an Outlier?** An outlier is a data point that is abnormally distant from the rest of the data. For example, if you are looking at customer ages ranging from 18 to 65, an age of `142` is an outlier.
> 
> **How QPX finds them (IQR Method):** 
> QPX uses the Interquartile Range (IQR). It finds the middle 50% of your data (between the 25th percentile `Q1` and 75th percentile `Q3`). Any value that falls unusually far below `Q1` (the `lower bound`) or unusually far above `Q3` (the `upper bound`) is flagged as an outlier.

**Parameters:**
- `detailed` *(bool, default=False)*: If `True`, the output includes the exact calculations used to find the outliers (`Q1`, `Q3`, and `IQR`).

**Example:**
```python
print(tab.outlier_summary(detailed=True).head(3).to_string())
```
**Output:**
```text
             lower bound  upper bound  outlier count  outlier %      Q1      Q3   IQR
order_year        2022.5       2026.5              0        0.0  2024.0  2025.0   1.0
order_month         -7.5         20.5              0        0.0     3.0    10.0   7.0
order_day          -14.5         45.5              0        0.0     8.0    23.0  15.0
```

---

### `multicollinearity_report()`
Evaluates the numerical columns for **multicollinearity**, which is a critical step before training many machine learning models (like Linear Regression).

> [!IMPORTANT]
> **What is Multicollinearity?** Multicollinearity happens when two or more columns in your dataset are highly correlated—meaning they give the machine learning model the exact same redundant information. (For example, having both "Year of Birth" and "Age" in the same dataset).
> 
> **Understanding VIF (Variance Inflation Factor):**
> QPX calculates a VIF score to detect this redundancy:
> - **VIF < 5:** `Keep` (The column is mathematically unique).
> - **VIF 5 - 10:** `Review` (Moderate correlation, proceed with caution).
> - **VIF > 10:** `Consider Removing` (Severe redundancy, you should probably drop this column).

**What it returns:**
A pandas DataFrame detailing the `VIF` score, a suggested action `Status`, Tolerance, and the specific feature it is clashing (highly correlated) with.

**Example:**
```python
print(tab.multicollinearity_report().head(3).to_string())
```
**Output:**
```text
       Feature   VIF Status  Tolerance Highest Correlation With  Correlation
0   order_year  1.06   Keep      0.941              order_month        0.243
1  order_month  1.06   Keep      0.941               order_year        0.243
2    order_day  1.00   Keep      1.000               order_year        0.013
```
