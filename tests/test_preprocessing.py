import pytest
from kpx.tabular import preprocessing

def test_boolify(dirty_df):
    boolified = preprocessing.boolify(dirty_df)
    assert boolified["pseudo_bool"].dtype == "bool"
    assert boolified["pseudo_bool"].iloc[0] == True
    assert boolified["pseudo_bool"].iloc[1] == False

def test_drop_constant(dirty_df):
    dropped = preprocessing.drop_constant(dirty_df)
    assert "constant_col" not in dropped.columns

def test_drop_high_cardinality(dirty_df):
    dropped = preprocessing.drop_high_cardinality(dirty_df)
    assert "id" not in dropped.columns

def test_keyerror_handling(clean_df):
    # Ensure invalid columns raise KeyError as implemented earlier
    with pytest.raises(KeyError):
        preprocessing.boolify(clean_df, columns=["invalid_column"])
    
    with pytest.raises(KeyError):
        preprocessing.cast_to_numeric(clean_df, columns=["invalid_column"])

    with pytest.raises(KeyError):
        preprocessing.label_encode(clean_df, columns=["invalid_column"])

def test_auto_preprocess(dirty_df):
    cleaned, report = preprocessing.auto_preprocess(dirty_df, return_report=True)
    assert "constant_col" not in cleaned.columns
    assert "id" not in cleaned.columns
    assert cleaned["pseudo_bool"].dtype == "bool"
    # Smart fill should handle missing values
    assert cleaned.isna().sum().sum() == 0
