import pytest
from unittest.mock import patch
from kpx.tabular import visuals

@patch("matplotlib.pyplot.show")
def test_corr_map(mock_show, clean_df, edge_case_df):
    # Should run cleanly
    visuals.corr_map(clean_df)
    mock_show.assert_called()

    # Test error handling
    with pytest.raises(ValueError):
        visuals.corr_map(clean_df, target="invalid_column")

@patch("matplotlib.pyplot.show")
def test_distribution_map(mock_show, clean_df):
    visuals.distribution_map(clean_df)
    mock_show.assert_called()

@patch("matplotlib.pyplot.show")
def test_relationship_map(mock_show, clean_df):
    visuals.relationship_map(clean_df, target="salary")
    mock_show.assert_called()

    with pytest.raises(ValueError):
        visuals.relationship_map(clean_df, target="invalid_target")

@patch("matplotlib.pyplot.show")
def test_feature_cluster_map(mock_show, clean_df, edge_case_df):
    visuals.feature_cluster_map(clean_df)
    mock_show.assert_called()

    with pytest.raises(ValueError):
        visuals.feature_cluster_map(edge_case_df)
