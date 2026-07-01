import pytest
from qpx.tabular import statistics

def test_numeric_summary(clean_df):
    summary = statistics.numeric_summary(clean_df)
    assert "Mean" in summary.columns
    assert summary.loc[summary["Feature"] == "age", "Mean"].iloc[0] == 35.0

def test_outlier_summary(clean_df, dirty_df):
    clean_outliers = statistics.outlier_summary(clean_df.select_dtypes(include="number"))
    assert clean_outliers.loc["age", "outlier count"] == 0

    dirty_outliers = statistics.outlier_summary(dirty_df.select_dtypes(include="number"))
    assert dirty_outliers.loc["age", "outlier count"] == 1 # 150 is an outlier

def test_multicollinearity_report(clean_df, edge_case_df):
    # Should work on clean_df
    numeric_df = clean_df.select_dtypes(include="number")
    report = statistics.multicollinearity_report(numeric_df)
    assert len(report) == 2

    # Should raise ValueError if < 2 numerical columns
    with pytest.raises(ValueError):
        statistics.multicollinearity_report(edge_case_df)

def test_statistical_snapshot(clean_df):
    snapshot = statistics.statistical_snapshot(clean_df)
    assert "Rows" in snapshot.index
    assert snapshot.loc["Rows", "Value"] == 5.0
