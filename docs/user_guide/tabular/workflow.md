# Recommended Workflow

To get the absolute best out of **QPX Tabular**, we recommend following a structured workflow that mirrors the actual machine learning lifecycle. QPX is designed to drastically minimize the code you write at each of these steps.

Here is the ideal path from raw dataset to ML-ready data:

## 1. Initial Assessment (Profiling)
Before altering any data, understand what you are working with.

- **Check Dataset Health**: Run `tab.dataset_health()` to instantly get a scorecard of missing values, high cardinality, and memory usage.
- **Understand Structure**: Use `tab.feature_overview()` to see exactly what columns exist and what data types they actually hold.
- **Determine Readiness**: Run `tab.ml_readiness()` to see if your dataset could theoretically be passed into a model right now (it will likely fail initially due to missing values or strings, which is fine!).

## 2. Deep Dive Diagnostics (Statistics & Visuals)
Once you know the broad strokes, look closer at the distributions and correlations.

- **Check Distributions**: Use `tab.distribution_map()` to visually identify skewed numerical data or dominant categorical classes.
- **Find Target Relationships**: Run `tab.relationship_map(target="your_target")` to see how every feature interacts with the outcome you are trying to predict.
- **Spot Multicollinearity**: Run `tab.feature_cluster_map()` and `tab.multicollinearity_report()`. If you see tight clusters of highly correlated features, let the preprocessor know it needs to drop them.

## 3. Automated Preprocessing
Now that you understand the dataset's flaws, let QPX fix them. 

For most standard tabular datasets, a single call to `auto_preprocess()` is all you need:

```python
# The ultimate one-liner to clean, encode, and compress
report = tab.auto_preprocess(
    target_col="your_target",
    sanitize=True,
    cast_numeric=True,
    drop_const=True,
    drop_high_cardinality_cols=True,
    drop_correlated_cols=True,
    encode=True,
    compress=True,
    return_report=True
)
```

> [!IMPORTANT]
> **Always review the `report`** dictionary. It is your transparent audit log detailing exactly what columns were dropped, encoded, and imputed.

## 4. Final Validation
Your data is now preprocessed. Let's make sure it is perfect.

- **Re-run ML Readiness**: Call `tab.ml_readiness()`. It should now pass with flying colors.
- **Verify with PCA**: Run `tab.pca_plot(target="your_target")` to ensure that there is still variance and pattern separation in your dataset after dimensionality reduction and preprocessing.

---

**You're done!** Your internal dataframe (`tab.df`) is now perfectly sanitized, strictly typed, memory-optimized, encoded, and ready to be fed directly into XGBoost, LightGBM, or Scikit-Learn.
