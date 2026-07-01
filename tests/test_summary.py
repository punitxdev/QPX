import pytest
from kpx.tabular import summary

def test_shape(clean_df):
    assert summary.shape(clean_df) == (5, 4)
    assert summary.num_rows(clean_df) == 5
    assert summary.num_columns(clean_df) == 4

def test_column_types(clean_df):
    types = summary.column_types(clean_df)
    assert "age" in types
    assert types["age"] == "int64"

def test_dataset_health(clean_df, dirty_df):
    clean_health = summary.dataset_health(clean_df)
    assert clean_health["Health Score"] == 100
    assert clean_health["Status"] == "Excellent"

    dirty_health = summary.dataset_health(dirty_df)
    assert dirty_health["Health Score"] < 100
    assert "constant_col" in dirty_health["Issues"]["Constant Columns"]["columns"]

def test_ml_readiness(clean_df, dirty_df):
    clean_ready = summary.ml_readiness(clean_df)
    assert clean_ready["isSafe"] is True

    dirty_ready = summary.ml_readiness(dirty_df)
    assert dirty_ready["isSafe"] is False
    assert "Missing Values" in dirty_ready["issues"]

def test_feature_overview(clean_df):
    overview = summary.feature_overview(clean_df)
    assert overview.shape == (4, 6)
    assert list(overview.columns) == ["Feature", "Dtype", "Missing", "Missing %", "Unique", "Memory"]
