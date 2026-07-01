import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def numeric_summary(df):
    """Generates detailed statistics (mean, median, skewness, kurtosis, etc.) for numerical columns."""
    metric = []

    for col in df.select_dtypes(include="number").columns:
        data = {}

        data["Feature"] = col
        data["Mean"] = df[col].mean()
        data["Median"] = df[col].median()
        data["Mode"] = df[col].mode().iloc[0] if not df[col].mode().empty else np.nan
        data["Std"] = df[col].std()
        data["Variance"] = df[col].var()
        data["Min"] = df[col].min()
        data["Q1"] = df[col].quantile(0.25)
        data["Max"] = df[col].max()
        data["Range"] = data["Max"] - data["Min"]
        data["CV (%)"] = (
            np.nan if data["Mean"] == 0 else (data["Std"] / data["Mean"]) * 100
        )
        data["Skewness"] = df[col].skew()

        metric.append(data)

    return pd.DataFrame(metric)


def categorical_summary(df, categorical_columns):
    """Generates detailed statistics (mode, unique count, top frequency) for categorical columns."""
    categorical_metric = {}

    for col in categorical_columns.columns:
        data = {}

        data["Dtype"] = str(df[col].dtype)
        data["count"] = df[col].shape[0]

        data["missing"] = int(
            (df[col].isna() | df[col].fillna("").astype(str).str.strip().eq("")).sum()
        )

        data["missing %"] = round(data["missing"] / len(df[col]) * 100, 2)

        data["unique"] = len(df[col].unique())
        data["mode"] = df[col].mode().iloc[0]
        data["mode %"] = round(df[col].value_counts(normalize=True).iloc[0] * 100, 2)
        p = df[col].value_counts(normalize=True)
        data["entropy"] = -np.sum(p * np.log2(p))
        data["binary"] = True if data["unique"] == 2 else False


        categorical_metric[col] = data

    return pd.DataFrame.from_dict(categorical_metric, orient="index")


def encoding_label(unique):
    if unique == 1:
        encoding = "Drop (Constant)"
    elif unique == 2:
        encoding = "Label Encoding"
    elif unique <= 10:
        encoding = "One-Hot Encoding"
    elif unique <= 50:
        encoding = "Frequency Encoding"
    else:
        encoding = "Target/Hash Encoding"
    return encoding


def outlier_summary(numeric_df, detailed=False):
    """
    Uses IQR thresholds to detect and quantify outliers across all numeric features.
    If detailed=True, returns actual quartile and IQR values in the report.
    """
    outlier_metric = {}
    for col in numeric_df.columns:
        data = {}
        # data['method'] = 'IQR'
        Q1 = numeric_df[col].quantile(0.25)
        Q3 = numeric_df[col].quantile(0.75)
        iqr = Q3 - Q1

        data["lower bound"] = Q1 - (1.5 * iqr)
        data["upper bound"] = Q3 + (1.5 * iqr)

        data["outlier count"] = (
            (numeric_df[col] < data["lower bound"])
            | (numeric_df[col] > data["upper bound"])
        ).sum()
        data["outlier %"] = (data["outlier count"] / numeric_df[col].shape[0]) * 100

        data["lower bound"] = Q1 - (1.5 * iqr)
        data["upper bound"] = Q3 + (1.5 * iqr)

        if detailed:
            data["Q1"] = Q1
            data["Q3"] = Q3
            data["IQR"] = iqr

        outlier_metric[col] = data

    return pd.DataFrame.from_dict(outlier_metric, orient="index")


def vif_status(vif):
    if vif < 5:
        return "Keep"
    elif vif < 10:
        return "Review"
    else:
        return "Consider Removing"


def multicollinearity_report(numeric_df):
    """
    Computes Variance Inflation Factor (VIF) to detect multicollinearity.
    
    Raises:
        ValueError: If numeric_df has fewer than 2 numerical features.
    """

    numeric_df = numeric_df.dropna()
    numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

    if numeric_df.shape[1] < 2:
        raise ValueError("VIF requires at least two numerical features.")

    # Correlation matrix
    corr = numeric_df.corr().abs()

    report = []

    for col in numeric_df.columns:
        y = numeric_df[col]
        X = numeric_df.drop(columns=col)

        model = LinearRegression()
        model.fit(X, y)

        r2 = model.score(X, y)

        vif = np.inf if r2 >= 0.999999 else 1 / (1 - r2)

        # Highest correlated feature
        highest_corr = corr[col].drop(col)

        report.append(
            {
                "Feature": col,
                "VIF": round(vif, 2),
                "Status": vif_status(vif),
                "Tolerance": round(1 / vif, 3),
                "Highest Correlation With": highest_corr.idxmax(),
                "Correlation": round(highest_corr.max(), 3),
            }
        )

    return pd.DataFrame(report)


def statistical_snapshot(df):
    """
    Combines numeric, categorical, and outlier summaries into one comprehensive 
    statistical snapshot DataFrame.
    """
    snapshot = {}

    snapshot["Rows"] = df.shape[0]
    snapshot["Columns"] = df.shape[1]
    snapshot["Memory Usage (MB)"] = round(
        df.memory_usage(deep=True).sum() / (1024**2), 2
    )

    snapshot["Numeric Columns"] = len(df.select_dtypes(include=np.number).columns)
    snapshot["Categorical Columns"] = len(
        df.select_dtypes(include=["object", "category"]).columns
    )
    snapshot["Boolean Columns"] = len(df.select_dtypes(include="bool").columns)
    snapshot["Datetime Columns"] = len(
        df.select_dtypes(include=["datetime", "datetimetz"]).columns
    )

    missing = df.isna().sum().sum()
    snapshot["Total Missing Values"] = int(missing)
    snapshot["Missing Percentage"] = round((missing / df.size) * 100, 2)
    snapshot["Columns with Missing"] = int((df.isna().sum() > 0).sum())

    snapshot["Duplicate Rows"] = int(df.duplicated().sum())
    snapshot["Duplicate Columns"] = int(df.columns.duplicated().sum())

    snapshot["Empty Columns"] = int(df.isna().all().sum())
    snapshot["Constant Columns"] = int((df.nunique(dropna=False) <= 1).sum())

    numeric = df.select_dtypes(include=np.number)
    snapshot["Infinite Values"] = (
        0 if numeric.empty else int(np.isinf(numeric).sum().sum())
    )

    snapshot["Mixed-Type Columns"] = int(
        df.apply(lambda s: s.dropna().map(type).nunique() > 1).sum()
    )

    snapshot["Average Missing %"] = round((df.isna().mean() * 100).mean(), 2)
    snapshot["Average Cardinality"] = round(df.nunique(dropna=True).mean(), 2)
    snapshot["Average Unique %"] = round(
        ((df.nunique(dropna=True) / len(df)) * 100).mean(), 2
    )

    return pd.DataFrame(snapshot.items(), columns=["Metric", "Value"]).set_index(
        "Metric"
    )
