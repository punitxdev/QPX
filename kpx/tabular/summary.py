import pandas as pd
import numpy as np

def shape(df):
    """Returns a tuple containing the (rows, columns) of the DataFrame."""
    return df.shape

def num_rows(df):
    """Returns the integer count of rows in the DataFrame."""
    return df.shape[0]

def num_columns(df):
    """Returns the integer count of columns in the DataFrame."""
    return df.shape[1]

def column_names(df):
    """Returns a list of all column headers in the DataFrame."""
    return(list(df.columns))

def column_types(df):
    """Returns a dictionary mapping column names to their data types."""
    c_type = {}
    for i in df.columns:
        c_type[i] = str((df[i].dtype))
    return c_type

def numerical_columns(df):
    """Returns a tuple of (list of numerical columns, count of numerical columns)."""
    numerical_cols = list(df.select_dtypes(include="number").columns)
    return (numerical_cols, len(numerical_cols))

def bool_columns(df):
    """Returns a tuple of (list of boolean columns, count of boolean columns)."""
    bool_cols = df.select_dtypes(include="bool").columns.tolist()
    return (bool_cols, len(bool_cols))

def categorical_columns(df):
    """Returns a tuple of (list of categorical columns, count of categorical columns)."""
    categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    return(categorical_cols, len(categorical_cols))

def datetime_columns(df):
    """Returns a tuple of (list of datetime columns, count of datetime columns)."""
    datetime_cols = list(df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
    return(datetime_cols, len(datetime_cols))

def memory_usage(df):
    """Returns a human-readable string representing the memory footprint (e.g., '15.2 MB')."""
    memory = df.memory_usage(deep=True).sum()
    return format_memory(memory)

def dataset_info(df):
    """Returns a lightweight dictionary containing core metrics (rows, columns, size)."""
    rows, cols = df.shape
    return(rows, cols, df.size)


def format_memory(bytes_):
    """Formats bytes into a human-readable string (B, KB, MB, GB)."""
    if bytes_ < 1024:
        return f"{bytes_} B"
    elif bytes_ < 1024**2:
        return f"{bytes_/1024:.2f} KB"
    elif bytes_ < 1024**3:
        return f"{bytes_/1024**2:.2f} MB"
    else:
        return f"{bytes_/1024**3:.2f} GB"

def column_profile(df, column_names=None):
    """
    Generates a deep statistical and structural profile for specific columns.
    If column_names is None, profiles all columns.
    """
    profiles = []
    c_names = []
    if column_names is None:
        c_names = df.columns
    elif isinstance(column_names, str):
        c_names = [column_names]
    else:
        c_names = column_names
    for i in c_names:
        profile = {
            "name":i,
            "dtype": str(df[i].dtype),
            "missing": df[i].isna().sum(),
            "missing_percent": round((df[i].isna().sum()/(df.shape[0]))*100,2),
            "unique": df[i].nunique(),
            "memory": format_memory(df[i].memory_usage(deep=True))
        }

        profiles.append(profile)

    return pd.DataFrame(profiles)  

def feature_overview(df):
    """
    Returns a detailed DataFrame outlining missing percentages, unique values, 
    and memory usage for every feature.
    """
    overview = {
        "Feature": [],
        "Dtype": [],
        "Missing": [],
        "Missing %": [],
        "Unique": [],
        "Memory": []
    }

    for col in df.columns:
        overview["Feature"].append(col)
        overview["Dtype"].append(str(df[col].dtype))
        overview["Missing"].append(df[col].isna().sum())
        overview["Missing %"].append(round(df[col].isna().mean() * 100, 2))
        overview["Unique"].append(df[col].nunique())
        overview["Memory"].append(format_memory(df[col].memory_usage(deep=True)))

    return pd.DataFrame(overview)

def summary(df):
    """
    Returns a structured overview DataFrame with high-level summary statistics.
    """
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "total cells": df.size,
        "missing values": int(df.isna().sum().sum()),
        "missing %":round((df.isna().sum().sum() / df.size) * 100, 2),
        "duplicate rows":int(df.duplicated().sum()),
        "memory usage": memory_usage(df),
        "numeric columns": numerical_columns(df)[1],
        "categorical columns": categorical_columns(df)[1],
        "boolean columns": bool_columns(df)[1],
        "datetime columns": datetime_columns(df)[1]
    }   

    return pd.DataFrame.from_dict(summary, orient="index", columns=["Value"])

def dataset_health(df):
    """
    Analyzes missing values, zero-variance columns, and high-cardinality issues 
    to return an overall health score and warnings dictionary.
    """
    missing_count = int(df.isna().sum().sum())
    missing_columns = df.columns[df.isna().any()].tolist()

    duplicate_rows = int(df.duplicated().sum())

    empty_columns = (
        df.fillna("")
        .astype(str)
        .apply(lambda col: col.str.strip().eq("").all())
    )
    empty_columns = empty_columns[empty_columns].index.tolist()

    constant_columns = [
        col for col in df.columns
        if df[col].nunique(dropna=True) == 1
    ]

    mixed_type_columns = [
        col for col in df.columns
        if df[col].dropna().map(type).nunique() > 1
    ]

    duplicate_column_names = (
        df.columns[df.columns.duplicated()]
        .unique()
        .tolist()
    )

    numeric_df = df.select_dtypes(include=np.number)

    infinite_columns = numeric_df.columns[
        np.isinf(numeric_df).any(axis=0)
    ].tolist()

    score = 100

    score -= min(20, missing_count)
    score -= min(15, duplicate_rows)
    score -= min(15, len(empty_columns) * 5)
    score -= min(10, len(constant_columns) * 2)
    score -= min(10, len(mixed_type_columns) * 5)
    score -= min(15, len(duplicate_column_names) * 5)
    score -= min(15, len(infinite_columns) * 5)

    score = max(score, 0)

    if score >= 90:
        status = "Excellent"
    elif score >= 75:
        status = "Good"
    elif score >= 60:
        status = "Fair"
    elif score >= 40:
        status = "Poor"
    else:
        status = "Critical"

    return {
        "Health Score": score,
        "Status": status,
        "Issues": {
            "Missing Values": {
                "count": missing_count,
                "columns": missing_columns
            },
            "Duplicate Rows": {
                "count": duplicate_rows
            },
            "Empty Columns": {
                "count": len(empty_columns),
                "columns": empty_columns
            },
            "Constant Columns": {
                "count": len(constant_columns),
                "columns": constant_columns
            },
            "Mixed Type Columns": {
                "count": len(mixed_type_columns),
                "columns": mixed_type_columns
            },
            "Duplicate Column Names": {
                "count": len(duplicate_column_names),
                "columns": duplicate_column_names
            },
            "Infinite Values": {
                "count": len(infinite_columns),
                "columns": infinite_columns
            }
        }
    }

def ml_readiness(df):
    """
    Runs structural checks to determine if the dataset is safely ready 
    for machine learning algorithms. Returns a dictionary with pass/fail and issues.
    """
    health = dataset_health(df)

    issues = {}

    for issue, details in health["Issues"].items():
        if details["count"] > 0:
            issues[issue] = details

    response =  {
        "isSafe": len(issues) == 0,
        "issues": issues
    }

    if response['isSafe'] == True:
        del response['issues']
    
    return response
    

             

