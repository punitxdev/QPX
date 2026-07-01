from . import summary
from . import statistics
from . import preprocessing
from . import visuals


class Tabular:
    def __init__(self, df):
        self.df = df
        self.numeric_df = df.select_dtypes(include="number")
        self.categorical_df = df.select_dtypes(include=["object", "category"])
        self.bool_df = df.select_dtypes(include="bool")

    # analytical functions

    def shape(self):
        return summary.shape(self.df)

    def num_rows(self):
        return summary.num_rows(self.df)

    def num_columns(self):
        return summary.num_columns(self.df)

    def column_names(self):
        return summary.column_names(self.df)

    def column_types(self):
        return summary.column_types(self.df)

    def numerical_columns(self):
        return summary.numerical_columns(self.df)

    def bool_columns(self):
        return summary.bool_columns(self.df)

    def categorical_columns(self):
        return summary.categorical_columns(self.df)

    def datetime_columns(self):
        return summary.datetime_columns(self.df)

    def memory_usage(self):
        return summary.memory_usage(self.df)

    def dataset_info(self):
        return summary.dataset_info(self.df)

    def summary(self):
        return summary.summary(self.df)

    def feature_overview(self):
        return summary.feature_overview(self.df)

    def dataset_health(self):
        return summary.dataset_health(self.df)

    def ml_readiness(self):
        return summary.ml_readiness(self.df)

    def column_profile(self, column_names=None):
        return summary.column_profile(self.df, column_names=column_names)

    # statistical functions

    def numeric_summary(self):
        return statistics.numeric_summary(self.df)

    def categorical_summary(self):
        return statistics.categorical_summary(self.df, self.categorical_df)

    def outlier_summary(self, detailed=False):
        return statistics.outlier_summary(self.numeric_df, detailed=detailed)

    def multicollinearity_report(self):
        return statistics.multicollinearity_report(self.numeric_df)

    def statistical_snapshot(self):
        return statistics.statistical_snapshot(self.df)

    # Preprocessing functions

    def sanitize_col_names(
        self,
        case="snake",
        remove_special_chars=True,
        replace_spaces="_",
        ensure_unique=True,
        return_mapping=False,
    ):
        return preprocessing.sanitize_col_names(
            self.df,
            case,
            remove_special_chars,
            replace_spaces,
            ensure_unique,
            return_mapping,
        )

    def boolify(
        self,
        columns=None,
        true_value=True,
        false_value=False,
        return_info=False,
        inplace=False,
    ):
        return preprocessing.boolify(
            self.df, columns, true_value, false_value, return_info, inplace
        )

    def drop_constant(
        self, columns=None, ignore_na=True, return_info=False, inplace=False
    ):
        return preprocessing.drop_constant(
            self.df, columns, ignore_na, return_info, inplace
        )

    def label_encode(
        self,
        columns=None,
        inplace=False,
        return_mapping=False,
    ):
        return preprocessing.label_encode(self.df, columns, inplace, return_mapping)

    def one_hot_encode(
        self,
        columns=None,
        inplace=False,
        return_info=False,
        drop_original=False,
        max_unique=20,
    ):
        return preprocessing.one_hot_encode(
            self.df, columns, inplace, return_info, drop_original, max_unique
        )

    def ordinal_encode(self, order, inplace=False, return_mapping=True, strict=True):
        return preprocessing.ordinal_encode(
            self.df, order, inplace, return_mapping, strict
        )

    def auto_encode(
        self,
        columns=None,
        order=None,
        max_onehot=10,
        inplace=False,
        return_report=False,
    ):
        return preprocessing.auto_encode(
            self.df,
            columns,
            order,
            max_onehot,
            inplace,
            return_report,
        )

    def compress_dtype(
        self,
        columns=None,
        downcast_int=True,
        downcast_float=True,
        convert_category=True,
        category_threshold=0.5,
        inplace=False,
        return_report=False,
    ):
        return preprocessing.compress_dtype(
            self.df,
            columns,
            downcast_int,
            downcast_float,
            convert_category,
            category_threshold,
            inplace,
            return_report,
        )

    def smart_fill(
        self,
        columns=None,
        strategy="auto",
        value=None,
        missing_threshold=0.6,
        inplace=False,
        return_report=False,
    ):
        return preprocessing.smart_fill(
            self.df,
            columns,
            strategy,
            value,
            missing_threshold,
            inplace,
            return_report,
        )

    def drop_high_cardinality(
        self,
        columns=None,
        threshold=0.8,
        return_info=False,
        inplace=False,
    ):
        return preprocessing.drop_high_cardinality(
            self.df, columns, threshold, return_info, inplace
        )

    def drop_highly_correlated(
        self,
        threshold=0.9,
        method="pearson",
        target_col=None,
        return_info=False,
        inplace=False,
    ):
        return preprocessing.drop_highly_correlated(
            self.df, threshold, method, target_col, return_info, inplace
        )

    def auto_preprocess(
        self,
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
        return preprocessing.auto_preprocess(
            df=self.df,
            target_col=target_col,
            sanitize=sanitize,
            cast_numeric=cast_numeric,
            drop_const=drop_const,
            drop_high_cardinality_cols=drop_high_cardinality_cols,
            drop_correlated_cols=drop_correlated_cols,
            boolify_cols=boolify_cols,
            fill_nan=fill_nan,
            encode=encode,
            compress=compress,
            missing_threshold=missing_threshold,
            high_cardinality_threshold=high_cardinality_threshold,
            correlation_threshold=correlation_threshold,
            max_onehot=max_onehot,
            return_report=return_report,
        )

    # visualization functions

    def corr_map(self, target=None, ignore_cols=None, method="spearman"):
        return visuals.corr_map(self.df, target=target, ignore_cols=ignore_cols, method=method)

    def pca_plot(self, input_cols=None, target=None, n_components=2, sample_space=None, figsize=(8, 6)):
        return visuals.pca_plot(
            self.df,
            input_cols=input_cols,
            target=target,
            n_components=n_components,
            sample_space=sample_space,
            figsize=figsize
        )

    def relationship_map(
        self,
        target,
        input_cols=None,
        ignore_cols=None,
        sample_space=1000,
        figsize=(9, 7),
    ):
        return visuals.relationship_map(
            self.df, target, input_cols, ignore_cols, sample_space, figsize
        )

    def distribution_map(
        self, cols=None, ignore_cols=None, sample_space=1000, figsize=(8, 6)
    ):
        return visuals.distribution_map(
            self.df, cols, ignore_cols, sample_space, figsize
        )

    def feature_cluster_map(self, ignore_cols=None, method="spearman", figsize=(10, 10)):
        return visuals.feature_cluster_map(self.df, ignore_cols, method, figsize)
