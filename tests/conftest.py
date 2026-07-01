import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def clean_df():
    """A perfectly clean, well-formed dataframe."""
    return pd.DataFrame({
        "age": [25, 30, 35, 40, 45],
        "salary": [50000, 60000, 70000, 80000, 90000],
        "category": ["A", "B", "A", "C", "B"],
        "is_active": [True, False, True, True, False]
    })

@pytest.fixture
def dirty_df():
    """A dirty dataframe with missing values, constants, high cardinality, etc."""
    return pd.DataFrame({
        "id": ["id1", "id2", "id3", "id4", "id5"],  # High cardinality string identifier
        "constant_col": ["yes", "yes", "yes", "yes", "yes"],  # Zero variance
        "age": [25, np.nan, 35, 40, 150],  # Missing value and outlier
        "salary": [50000, 60000, 70000, np.nan, 90000],
        "category": ["A", "B", "Unknown", "-", "B"],  # Messy strings
        "pseudo_bool": ["yes", "no", "yes", "yes", "no"],  # Text boolean (2 unique)
        "highly_correlated_age": [25, np.nan, 35, 40, 150]  # Exact copy of age
    })

@pytest.fixture
def edge_case_df():
    """A dataframe with no numerical columns to test gracefully handling errors."""
    return pd.DataFrame({
        "col1": ["cat", "dog", "mouse"],
        "col2": ["yes", "no", "yes"],
        "col3": [pd.NaT, pd.NaT, pd.NaT]
    })
