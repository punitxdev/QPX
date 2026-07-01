import re
import warnings
import numpy as np
import pandas as pd
from . import summary


def sanitize_col_names(
    df,
    case="snake",
    remove_special_chars=True,
    replace_spaces="_",
    ensure_unique=True,
    return_mapping=False,
):
    """
    Standardizes column headers (e.g., lowercasing, replacing spaces with underscores) 
    to make them machine-readable and easy to access.
    """
    df = df.copy()
    mapping = {}
    new_columns = []

    for col in df.columns:
        new_col = str(col).strip()

        if remove_special_chars:
            new_col = re.sub(r"[^\w\s]", "", new_col)

        new_col = re.sub(r"\s+", replace_spaces, new_col)

        if case == "snake":
            new_col = new_col.lower()
        elif case == "lower":
            new_col = new_col.lower().replace("_", "")
        elif case == "upper":
            new_col = new_col.upper().replace("_", "")
        elif case == "camel":
            parts = re.split(r"[_\s]+", new_col)
            new_col = parts[0].lower() + "".join(word.title() for word in parts[1:])
        elif case == "pascal":
            parts = re.split(r"[_\s]+", new_col)
            new_col = "".join(word.title() for word in parts)

        new_col = re.sub(r"_+", "_", new_col).strip("_")

        mapping[col] = new_col
        new_columns.append(new_col)

    if ensure_unique:
        seen = {}
        unique_columns = []

        for col in new_columns:
            if col not in seen:
                seen[col] = 0
                unique_columns.append(col)
            else:
                seen[col] += 1
                unique_columns.append(f"{col}_{seen[col]}")

        new_columns = unique_columns

    df.columns = new_columns

    if return_mapping:
        return df, mapping

    return df


def boolify(
    df,
    columns=None,
    true_value=True,
    false_value=False,
    return_info=False,
    inplace=False,
):
    """
    Converts pseudo-boolean text columns (like 'Yes'/'No', 'On'/'Off') into strict boolean types.
    """
    TRUE_VALUES = {
        "true",
        "t",
        "yes",
        "y",
        "1",
        "on",
        "enable",
        "enabled",
        "active",
        "pass",
        "passed",
        "success",
        "successful",
    }

    FALSE_VALUES = {
        "false",
        "f",
        "no",
        "n",
        "0",
        "off",
        "disable",
        "disabled",
        "inactive",
        "fail",
        "failed",
        "failure",
        "unsuccessful",
    }

    if not inplace:
        df = df.copy()

    info = []

    if columns is None:
        columns = df.columns.tolist()

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column \"{col}\" not found in DataFrame.")

        unique = df[col].dropna().unique()
        normalized = {str(v).strip().lower() for v in unique}

        if 1 <= len(normalized) <= 2 and normalized.issubset(
            TRUE_VALUES | FALSE_VALUES
        ):
            mapping = {}

            for value in unique:
                key = str(value).strip().lower()

                if key in TRUE_VALUES:
                    mapping[value] = true_value
                else:
                    mapping[value] = false_value

            df[col] = df[col].map(mapping)
            info.append(col)

    if return_info:
        return df, info

    return df


def cast_to_numeric(df, columns=None, inplace=False, return_info=False):
    """
    Attempts to safely cast categorical/string columns to numeric data types, 
    treating 'Unknown', '-', and 'NaN' as missing values.
    """
    if not inplace:
        df = df.copy()

    if columns is None:
        columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    converted_cols = []

    na_strings = ["", "nan", "none", "na", "n/a", "null", "-", "?", ".", "unknown"]

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column \"{col}\" not found in DataFrame.")
            
        temp_series = df[col].copy()
        if isinstance(temp_series.dtype, pd.CategoricalDtype):
             temp_series = temp_series.astype("object")
             
        # Standardize missing string values to actual NaNs for a fair check
        mask = temp_series.astype(str).str.strip().str.lower().isin(na_strings)
        temp_series[mask] = np.nan

        original_na = temp_series.isna().sum()
        numeric_series = pd.to_numeric(temp_series, errors="coerce")
        new_na = numeric_series.isna().sum()

        if original_na == new_na and len(temp_series.dropna()) > 0:
            df[col] = numeric_series
            converted_cols.append(col)
        elif len(temp_series.dropna()) == 0:
            df[col] = temp_series.astype(float)
            converted_cols.append(col)

    if return_info:
        return df, converted_cols
    return df


def drop_constant(df, columns=None, ignore_na=True, return_info=False, inplace=False):
    """
    Removes columns that contain only a single unique value (zero variance).
    """
    df = df.copy()
    if columns is None:
        columns = df.columns.tolist()

    const_cols = []
    for col in columns:
        unique = df[col].nunique(dropna=ignore_na)
        if unique <= 1:
            const_cols.append(col)

    if return_info:
        return df.drop(columns=const_cols, inplace=inplace), const_cols
    return df.drop(columns=const_cols, inplace=inplace)


def drop_high_cardinality(
    df, columns=None, threshold=0.8, target_col=None, return_info=False, inplace=False
):
    """
    Drops nominal columns where almost every row has a unique value (like names or IDs).
    """
    if not inplace:
        df = df.copy()

    if columns is None:
        columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        
    if target_col and target_col in columns:
        columns.remove(target_col)

    high_card_cols = []
    total_rows = len(df)

    if total_rows > 0:
        for col in columns:
            if col not in df.columns:
                raise KeyError(f"Column \"{col}\" not found in DataFrame.")

            unique_count = df[col].nunique(dropna=True)
            if (unique_count / total_rows) >= threshold:
                high_card_cols.append(col)

    if return_info:
        return df.drop(columns=high_card_cols, inplace=inplace), high_card_cols
    return df.drop(columns=high_card_cols, inplace=inplace)


def drop_highly_correlated(
    df, threshold=0.9, method="pearson", target_col=None, return_info=False, inplace=False
):
    """
    Identifies and drops one feature from pairs of highly correlated features 
    to reduce multicollinearity.
    """
    if not inplace:
        df = df.copy()

    numeric_df = df.select_dtypes(include="number")
    
    if target_col and target_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[target_col])
        
    if numeric_df.empty:
        if return_info:
            return df, []
        return df

    corr_matrix = numeric_df.corr(method=method).abs()
    
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    df.drop(columns=to_drop, inplace=True)

    if return_info:
        return df, to_drop
    return df


# ==================== Encoding strategy ======================
def label_encode(df, columns=None, inplace=False, return_mapping=False):
    """
    Numerically encodes categorical text variables using integer labels.
    """
    if inplace:
        df = df.copy()

    if columns is None:
        columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    mapping = {}
    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column \"{col}\" not found in DataFrame.")

        unique = sorted(df[col].dropna().unique())
        mapping[col] = {val: i for i, val in enumerate(unique)}

        df[col] = df[col].map(mapping[col])

    if return_mapping:
        return df, mapping

    return df


def one_hot_encode(
    df,
    columns=None,
    inplace=False,
    return_info=False,
    drop_original=False,
    max_unique=20,
):
    """
    Creates binary dummy variables for low-cardinality categorical columns.
    """
    if not inplace:
        df = df.copy()

    if columns is None:
        columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    encoded_cols = []
    not_encoded_cols = []

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column \"{col}\" not found in DataFrame.")

        unique = df[col].nunique()
        if 2 <= unique <= max_unique:
            df = pd.get_dummies(
                df, columns=[col], prefix=col, drop_first=drop_original, dtype=int
            )
            encoded_cols.append(col)
        else:
            not_encoded_cols.append(col)

    if return_info:
        return df, {"encoded": encoded_cols, "not_encoded": not_encoded_cols}

    return df


def ordinal_encode(
    df,
    order,
    inplace=False,
    return_mapping=False,
    strict=True,
):
    """
    Encodes specific categorical columns according to a strict, user-defined hierarchical order.
    
    Raises:
        KeyError: If a column specified in `order` is not found in the DataFrame.
        ValueError: If `strict=True` and the column contains unknown categories not present in `order`.
    """
    if not inplace:
        df = df.copy()

    mappings = {}

    for col, values in order.items():
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found.")

        mapping = {str(v).strip().lower(): i for i, v in enumerate(values)}

        column = df[col].astype("string").str.strip().str.lower()

        if strict:
            unknown = set(column.dropna().unique()) - set(mapping)

            if unknown:
                raise ValueError(f"Unknown categories in '{col}': {unknown}")

        df[col] = column.map(mapping)

        mappings[col] = mapping

    if return_mapping:
        return df, mappings

    return df


def auto_encode(
    df,
    columns=None,
    order=None,
    max_onehot=10,
    inplace=False,
    return_report=False,
):
    """
    Intelligently applies one-hot or label encoding based on column cardinality.
    """
    if not inplace:
        df = df.copy()

    if columns is None:
        columns = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    if order is None:
        order = {}

    report = {
        "label_encoded": [],
        "one_hot_encoded": [],
        "ordinal_encoded": [],
        "skipped": [],
    }

    label_cols = []
    onehot_cols = []
    ordinal_order = {}

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column \"{col}\" not found in DataFrame.")

        if col in order:
            ordinal_order[col] = order[col]
            report["ordinal_encoded"].append(col)
            continue

        n = df[col].nunique(dropna=True)

        if n == 2:
            label_cols.append(col)
            report["label_encoded"].append(col)

        elif 3 <= n <= max_onehot:
            onehot_cols.append(col)
            report["one_hot_encoded"].append(col)

        else:
            report["skipped"].append(f"{col} (nunique={n}, dtype={df[col].dtype})")

    if ordinal_order:
        df = ordinal_encode(
            df,
            order=ordinal_order,
            inplace=True,
            return_mapping=False,
            strict=False,
        )

    if label_cols:
        df = label_encode(
            df,
            columns=label_cols,
            inplace=True,
            return_mapping=False,
        )

    if onehot_cols:
        df = one_hot_encode(
            df,
            columns=onehot_cols,
            inplace=True,
            return_info=False,
            drop_original=False,
            max_unique=max_onehot,
        )

    if return_report:
        return df, report

    return df


# ========================================= Data type strategy =====================


def compress_dtype(
    df,
    columns=None,
    downcast_int=True,
    downcast_float=True,
    convert_category=True,
    category_threshold=0.5,
    inplace=False,
    return_report=False,
):
    """
    Safely downcasts integers and floats to drastically reduce memory footprint 
    without losing precision.
    """

    if not inplace:
        df = df.copy()

    if columns is None:
        columns = df.columns.tolist()

    memory_before = df.memory_usage(deep=True).sum()

    report = {
        "memory_before_compression": round(memory_before / 1024**2, 3),
        "memory_after_compression": None,
        "saved_mb": None,
        "saved_percent": None,
        "converted": {},
    }

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column \"{col}\" not found in DataFrame.")

        dtype = df[col].dtype

        if pd.api.types.is_bool_dtype(dtype):
            continue

        elif downcast_int and pd.api.types.is_integer_dtype(dtype):
            new_dtype = pd.to_numeric(
                df[col],
                downcast="integer",
            ).dtype

            if new_dtype != dtype:
                df[col] = df[col].astype(new_dtype)

                report["converted"][col] = {
                    "before": str(dtype),
                    "after": str(new_dtype),
                }

        elif downcast_float and pd.api.types.is_float_dtype(dtype):
            new_dtype = pd.to_numeric(
                df[col],
                downcast="float",
            ).dtype

            if new_dtype != dtype:
                df[col] = df[col].astype(new_dtype)

                report["converted"][col] = {
                    "before": str(dtype),
                    "after": str(new_dtype),
                }

        elif convert_category and (
            pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype)
        ):
            total = df[col].count()
            unique = df[col].nunique(dropna=True)

            if total > 0 and unique > 1 and unique / total <= category_threshold:
                df[col] = df[col].astype("category")

                report["converted"][col] = {
                    "before": str(dtype),
                    "after": "category",
                }

    memory_after = df.memory_usage(deep=True).sum()

    report["memory_after_compression"] = round(memory_after / 1024**2, 3)
    report["saved_mb"] = round((memory_before - memory_after) / 1024**2, 3)

    if memory_before != 0:
        report["saved_percent"] = round(
            ((memory_before - memory_after) / memory_before) * 100,
            2,
        )
    else:
        report["saved_percent"] = 0

    if return_report:
        return df, report

    return df


def smart_fill(
    df,
    columns=None,
    strategy="auto",
    value=None,
    missing_threshold=0.6,
    inplace=False,
    return_report=False,
):
    """
    Intelligently imputes missing values based on data types and distribution.
    """

    if not inplace:
        df = df.copy()

    if columns is None:
        columns = df.columns.tolist()

    report = {
        "filled": {},
        "skipped": {},
        "recommended_drop": {},
    }

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column \"{col}\" not found in DataFrame.")

        if df[col].isna().sum() == 0:
            continue

        missing_ratio = _missing_ratio(df[col])

        if missing_ratio >= missing_threshold:
            report["recommended_drop"][col] = {
                "missing_percent": round(missing_ratio * 100, 2)
            }
            continue

        semantic_type = _infer_semantic_type(df[col])

        if strategy == "auto":
            fill_strategy = _choose_strategy(
                df[col],
                semantic_type,
            )
        else:
            fill_strategy = strategy

        before = df[col].isna().sum()

        df[col] = _apply_strategy(
            df[col],
            fill_strategy,
            value,
        )

        after = df[col].isna().sum()

        report["filled"][col] = {
            "semantic_type": semantic_type,
            "strategy": fill_strategy,
            "filled": before - after,
        }

    if return_report:
        return df, report

    return df


def _missing_ratio(series):
    return series.isna().mean()


def _infer_semantic_type(series):

    if pd.api.types.is_bool_dtype(series):
        return "binary"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    unique = series.nunique(dropna=True)
    total = len(series)

    if unique == 2:
        return "binary"

    if unique / total <= 0.1:
        return "category"

    return "text"


def _choose_strategy(series, semantic_type):

    if semantic_type == "numeric":
        skew = series.dropna().skew()

        if abs(skew) < 1:
            return "mean"

        return "median"

    if semantic_type == "binary":
        return "mode"

    if semantic_type == "category":
        return "mode"

    if semantic_type == "datetime":
        return "ffill"

    return "skip"


def _apply_strategy(series, strategy, value=None):

    if strategy == "mean":
        return series.fillna(series.mean())

    elif strategy == "median":
        return series.fillna(series.median())

    elif strategy == "mode":
        mode = series.mode(dropna=True)
        if len(mode):
            return series.fillna(mode.iloc[0])
        return series

    elif strategy == "ffill":
        return series.ffill()

    elif strategy == "bfill":
        return series.bfill()

    elif strategy == "constant":
        return series.fillna(value)

    elif strategy == "drop":
        return series.dropna()

    return series


# ===================================== Full automate preprocesssing Strategy =============================

def auto_preprocess(
    df,
    target_col=None,
    sanitize=True,
    cast_numeric=True,
    drop_const=True,
    drop_high_cardinality_cols=False,
    drop_correlated_cols=False,
    boolify_cols=True,
    fill_nan=True,
    encode=True,
    compress=True,
    missing_threshold=0.6,
    high_cardinality_threshold=0.8,
    correlation_threshold=0.9,
    max_onehot=10,
    return_report=False,
):
    """
    The flagship preprocessing pipeline. Executes a full, configurable sequence 
    of sanitization, cleaning, imputation, encoding, and compression in one line.
    """
    df = df.copy()
    initial_shape = df.shape
    report = {}
    
    if return_report:
        init_health = summary.dataset_health(df)
        report["initial_health"] = f"{init_health['Health Score']} ({init_health['Status']})"

    if sanitize:
        if return_report:
            df, mapping = sanitize_col_names(df, return_mapping=True)
            changed = [k for k, v in mapping.items() if str(k) != v]
            report["sanitized"] = changed if len(changed) < 10 else len(changed)
        else:
            df = sanitize_col_names(df)

    unnamed_cols = [col for col in df.columns if str(col).lower().startswith("unnamed")]
    if unnamed_cols:
        warnings.warn(f"Automatically dropping likely index columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)
        if return_report:
            report["dropped_unnamed"] = unnamed_cols

    if cast_numeric:
        if return_report:
            df, num_cols = cast_to_numeric(df, return_info=True)
            report["casted_to_numeric"] = num_cols if len(num_cols) < 10 else len(num_cols)
        else:
            df = cast_to_numeric(df)

    if drop_const:
        if return_report:
            df, const_cols = drop_constant(df, return_info=True)
            report["dropped_constant"] = const_cols
        else:
            df = drop_constant(df)

    _, high_card_cols = drop_high_cardinality(df, threshold=high_cardinality_threshold, target_col=target_col, return_info=True)
    if high_card_cols:
        warnings.warn(f"High cardinality columns detected: {high_card_cols}. Consider using alternative encoding strategies.")
        if drop_high_cardinality_cols:
            df = df.drop(columns=high_card_cols)
            if return_report:
                report["dropped_high_cardinality"] = high_card_cols

    if boolify_cols:
        if return_report:
            df, bool_cols = boolify(df, return_info=True)
            report["boolified"] = bool_cols if len(bool_cols) < 10 else len(bool_cols)
        else:
            df = boolify(df)

    if fill_nan:
        if return_report:
            df, fill_rep = smart_fill(df, missing_threshold=missing_threshold, return_report=True)
            imputed = [k for k, v in fill_rep["filled"].items() if v["filled"] > 0]
            report["nans_imputed_cols"] = imputed if len(imputed) < 10 else len(imputed)
            report["nans_imputed_count"] = sum(v["filled"] for v in fill_rep["filled"].values())
        else:
            df = smart_fill(df, missing_threshold=missing_threshold)

    if encode:
        if return_report:
            df, enc_rep = auto_encode(df, max_onehot=max_onehot, return_report=True)
            encoded_cols = enc_rep["label_encoded"] + enc_rep["one_hot_encoded"]
            report["encoded"] = encoded_cols if len(encoded_cols) < 10 else len(encoded_cols)
        else:
            df = auto_encode(df, max_onehot=max_onehot)

    if drop_correlated_cols:
        if return_report:
            df, corr_cols = drop_highly_correlated(df, threshold=correlation_threshold, target_col=target_col, return_info=True)
            report["dropped_correlated"] = corr_cols
        else:
            df = drop_highly_correlated(df, threshold=correlation_threshold, target_col=target_col)

    if compress:
        if return_report:
            df, comp_rep = compress_dtype(df, return_report=True)
            report["memory_saved_mb"] = comp_rep.get("saved_mb", 0)
        else:
            df = compress_dtype(df)

    leftover_cols = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if leftover_cols:
        if target_col and target_col in leftover_cols:
            leftover_cols.remove(target_col)
            
        if leftover_cols:
            warnings.warn(f"Silently dropping remaining non-numeric columns: {leftover_cols}")
            df = df.drop(columns=leftover_cols)
            if return_report:
                report["left_no_useful"] = leftover_cols

    initial_rows = len(df)
    df.dropna(inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        warnings.warn(f"Dropped {rows_dropped} rows containing remaining missing values.")
        if return_report:
            report["dropped_nan_rows"] = rows_dropped

    if return_report:
        report["initial_shape"] = initial_shape
        report["final_shape"] = df.shape
        final_health = summary.dataset_health(df)
        report["final_health"] = f"{final_health['Health Score']} ({final_health['Status']})"
        return df, report
    return df
