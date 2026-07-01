import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set a premium global theme for QPX visualizations
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)



def corr_map(df, target=None, ignore_cols=None, method="spearman"):
    """
    Generates a heatmap displaying the correlation matrix of numerical features.
    
    Raises:
        ValueError: If a target is provided but it is not a numeric column in the dataset.
    """
    numeric_df = df.select_dtypes(include=["number", "bool"])
    if ignore_cols:
        numeric_df = numeric_df.drop(columns=[c for c in ignore_cols if c in numeric_df.columns])
    corr = numeric_df.corr(method=method)

    if target is None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, center=0, annot_kws={"size": 10}, linewidths=0.5)
        plt.title("Correlation Matrix", pad=15, fontweight="bold")
    else:
        if target not in corr.columns:
            raise ValueError(f"'{target}' is not a numeric column.")

        target_corr = (
            corr[[target]].sort_values(by=target, ascending=False).drop(index=target)
        )

        height = max(4, len(target_corr) * 0.5)
        plt.figure(figsize=(4, height))
        sns.heatmap(target_corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1, center=0, annot_kws={"size": 10}, linewidths=0.5)
        plt.title(f"Correlation with '{target}'", pad=15, fontweight="bold")

    plt.tight_layout()
    plt.show()


def pca_plot(
    df,
    input_cols=None,
    target=None,
    n_components=2,
    sample_space=None,
    figsize=(9, 7),
):
    """
    Performs Principal Component Analysis (PCA) and plots the variance.
    
    Raises:
        ValueError: For invalid `n_components`, missing target, insufficient numerical features, 
                    zero data rows after dropping NaNs, or an invalid sample size.
    """
    if n_components not in (2, 3):
        raise ValueError("Only n_components=2 or 3 are supported.")

    if input_cols is None:
        input_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

    if target is not None and target in input_cols:
        input_cols.remove(target)

    X = df[input_cols].copy()

    target_col = None
    if target is not None:
        if target not in df.columns:
            raise ValueError(f"'{target}' not found in DataFrame.")
        target_col = df[target]

    mask = X.isna().any(axis=1)

    if target_col is not None:
        mask |= target_col.isna()

    if mask.any():
        print(f"Dropped {mask.sum()} rows containing missing values.")
        X = X.loc[~mask]
        if target_col is not None:
            target_col = target_col.loc[~mask]

    if X.empty:
        raise ValueError("No data available after removing missing values.")

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        raise ValueError("PCA requires at least two numeric features.")

    if n_components > len(numeric_cols):
        raise ValueError(
            f"n_components={n_components} exceeds the number of numeric features ({len(numeric_cols)})."
        )

    X = X[numeric_cols]

    X_scaled = StandardScaler().fit_transform(X)

    model = PCA(n_components=n_components)
    components = model.fit_transform(X_scaled)

    pc_df = pd.DataFrame(
        components,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )

    if target_col is not None:
        pc_df[target] = target_col.values

    plot_df = pc_df

    if sample_space is not None:
        if sample_space <= 0:
            raise ValueError("sample_size must be greater than 0.")

        if len(pc_df) > sample_space:
            plot_df = pc_df.sample(sample_space, random_state=42)
            print(
                f"Showing {len(plot_df):,} randomly sampled points out of {len(pc_df):,}."
            )

    n = len(plot_df)

    if n < 500:
        size = 60
    elif n < 2000:
        size = 25
    elif n < 10000:
        size = 10
    else:
        size = 4

    if n_components == 2:
        plt.figure(figsize=figsize)

        if target is not None:
            sns.scatterplot(
                data=plot_df,
                x="PC1",
                y="PC2",
                hue=target,
                palette="tab10",
                s=size,
                alpha=0.6,
                linewidth=0,
                legend="full" if len(plot_df[target].unique()) < 10 else False,
            )

        else:
            sns.scatterplot(
                data=plot_df,
                x="PC1",
                y="PC2",
                s=size,
                alpha=0.5,
                linewidth=0,
                legend="full" if len(plot_df[target].unique()) < 10 else False,
            )

        plt.xlabel(f"PC1 ({model.explained_variance_ratio_[0] * 100:.2f}%)", fontweight="bold")
        plt.ylabel(f"PC2 ({model.explained_variance_ratio_[1] * 100:.2f}%)", fontweight="bold")
        plt.title("Principal Component Analysis", pad=15, fontweight="bold")
        sns.despine()
        plt.tight_layout()
        plt.show()

    else:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        if target is None:
            ax.scatter(
                plot_df["PC1"],
                plot_df["PC2"],
                plot_df["PC3"],
                s=size,
                alpha=0.6,
            )

        else:
            for cls in plot_df[target].unique():
                subset = plot_df[plot_df[target] == cls]

                ax.scatter(
                    subset["PC1"],
                    subset["PC2"],
                    subset["PC3"],
                    s=size,
                    alpha=0.6,
                    label=str(cls),
                )

            if plot_df[target].nunique() < 10:
                ax.legend(title=target)

        ax.set_xlabel(f"PC1 ({model.explained_variance_ratio_[0] * 100:.2f}%)", fontweight="bold")
        ax.set_ylabel(f"PC2 ({model.explained_variance_ratio_[1] * 100:.2f}%)", fontweight="bold")
        ax.set_zlabel(f"PC3 ({model.explained_variance_ratio_[2] * 100:.2f}%)", fontweight="bold")
        ax.set_title("Principal Component Analysis", pad=15, fontweight="bold")

        plt.tight_layout()
        plt.show()


def relationship_map(
    df, target, input_cols=None, ignore_cols=None, sample_space=1000, figsize=(9, 7)
):
    """
    Intelligently generates bivariate plots (scatter, box, or count plots) 
    to show the relationship between input features and the target variable.
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in DataFrame.")

    if input_cols is None:
        input_cols = [c for c in df.columns if c != target]

    if ignore_cols is None:
        ignore_cols = []

    plot_df = df.copy()
    if len(plot_df) > sample_space:
        plot_df = plot_df.sample(sample_space, random_state=42)

    n = len(plot_df)
    if n < 500:
        size = 60
    elif n < 2000:
        size = 25
    elif n < 10000:
        size = 10
    else:
        size = 4

    target_is_categorical = (
        pd.api.types.is_object_dtype(plot_df[target])
        or isinstance(plot_df[target].dtype, pd.CategoricalDtype)
        or plot_df[target].nunique() < 15
    )

    for col in input_cols:
        if col == target or col in ignore_cols:
            continue

        if (
            plot_df[col].nunique(dropna=False) <= 1
            or col.lower().endswith(("id", "_id"))
            or plot_df[col].nunique(dropna=False) / len(plot_df) > 0.95
            or (
                pd.api.types.is_string_dtype(plot_df[col])
                and plot_df[col].nunique(dropna=False) > 50
            )
        ):
            print(f"Skipping '{col}' (Identifier / Text / High-cardinality)")
            continue

        col_is_categorical = (
            pd.api.types.is_object_dtype(plot_df[col])
            or isinstance(plot_df[col].dtype, pd.CategoricalDtype)
            or plot_df[col].nunique() < 15
        )

        plt.figure(figsize=figsize)

        if target_is_categorical and col_is_categorical:
            sns.countplot(data=plot_df, x=col, hue=target, palette="Set2")
            plt.title(f"Distribution of {col} by {target}", pad=15, fontweight="bold")
            plt.xticks(rotation=45, ha="right")

        elif target_is_categorical and not col_is_categorical:
            sns.violinplot(
                data=plot_df, x=target, y=col, palette="Set2", inner="quartile"
            )
            plt.title(f"Distribution of {col} across {target} classes", pad=15, fontweight="bold")
            plt.xticks(rotation=45, ha="right")

        elif not target_is_categorical and col_is_categorical:
            sns.boxplot(data=plot_df, x=col, y=target, palette="Set2")
            plt.title(f"Distribution of {target} across {col} categories", pad=15, fontweight="bold")
            plt.xticks(rotation=45, ha="right")

        else:
            sns.regplot(
                data=plot_df,
                x=col,
                y=target,
                scatter_kws={"s": size, "alpha": 0.7, "linewidth": 0, "color": "#4C72B0"},
                line_kws={"color": "#C44E52"},
            )
            plt.title(f"Relationship between {col} and {target}", pad=15, fontweight="bold")

        plt.xlabel(col, fontweight="bold")
        plt.ylabel(target, fontweight="bold")
        sns.despine()
        plt.tight_layout()
        plt.show()


def distribution_map(
    df, cols=None, ignore_cols=None, sample_space=1000, figsize=(8, 6)
):
    """
    Generates univariate distribution plots for features. Automatically selects 
    histograms for numerical data and count plots for categorical data.
    """
    if cols is None:
        cols = df.columns

    if ignore_cols is None:
        ignore_cols = []

    plot_df = df.copy()
    if len(plot_df) > sample_space:
        plot_df = plot_df.sample(sample_space, random_state=42)

    for col in cols:
        if col in ignore_cols:
            print(f"Skipping '{col}' (User ignored)")
            continue

        if (
            plot_df[col].nunique(dropna=False) <= 1
            or col.lower().endswith(("id", "_id"))
            or plot_df[col].nunique(dropna=False) / len(plot_df) > 0.95
            or (
                pd.api.types.is_object_dtype(plot_df[col])
                and plot_df[col].nunique(dropna=False) > 50
            )
        ):
            print(f"Skipping '{col}' (Identifier / Constant / High-cardinality)")
            continue

        plt.figure(figsize=figsize)

        is_categorical = (
            pd.api.types.is_object_dtype(plot_df[col])
            or isinstance(plot_df[col].dtype, pd.CategoricalDtype)
            or plot_df[col].nunique() < 15
        )

        if is_categorical:
            sns.countplot(
                data=plot_df,
                x=col,
                palette="Set2",
                order=plot_df[col].value_counts().index,
            )
            plt.title(f"Count Distribution of {col}", pad=15, fontweight="bold")
            plt.xticks(rotation=45, ha="right")
        else:
            sns.histplot(data=plot_df, x=col, kde=True, color="#4C72B0", linewidth=0)
            plt.title(f"Density Distribution of {col}", pad=15, fontweight="bold")

        plt.xlabel(col, fontweight="bold")
        plt.ylabel("Frequency", fontweight="bold")
        sns.despine()
        plt.tight_layout()
        plt.show()


def feature_cluster_map(df, ignore_cols=None, method="spearman", figsize=(10, 10)):
    """
    Generates a hierarchically-clustered heatmap of correlations to visually 
    group highly correlated multicollinear features.
    """
    numeric_df = df.select_dtypes(include=["number", "bool"])
    if ignore_cols:
        numeric_df = numeric_df.drop(columns=[c for c in ignore_cols if c in numeric_df.columns])
    if numeric_df.shape[1] < 2:
        raise ValueError("Not enough numeric features for clustering.")

    corr = numeric_df.corr(method=method)
    corr.fillna(0, inplace=True)

    cg = sns.clustermap(
        corr,
        cmap="coolwarm",
        annot=True if corr.shape[1] < 15 else False,
        fmt=".2f",
        figsize=figsize,
        vmin=-1,
        vmax=1,
        center=0,
        annot_kws={"size": 10},
        linewidths=0.5
    )

    plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

    cg.fig.suptitle("Feature Correlation Clustermap", y=1.02, fontweight="bold", fontsize=14)
    plt.show()
