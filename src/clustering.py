import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.cluster import KMeans


NUMERIC_FEATURES = [
    "Age",
    "Total Spend"
]

CATEGORICAL_FEATURES = [
    "Membership Type",
    "Discount Applied"
]


def prepare_features(df):
    """Prepare numerical and categorical customer features."""

    numeric_data = df[NUMERIC_FEATURES].copy()
    categorical_data = df[CATEGORICAL_FEATURES].copy()

    return numeric_data, categorical_data


def train_clustering_model(df, n_clusters=7):
    """Train the final K-Means customer segmentation model."""

    numeric_data, categorical_data = prepare_features(df)

    scaler = MinMaxScaler()

    X_numeric = scaler.fit_transform(
        numeric_data
    )

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    X_categorical = encoder.fit_transform(
        categorical_data
    )

    X_final = pd.concat(
        [
            pd.DataFrame(X_numeric),
            pd.DataFrame(X_categorical)
        ],
        axis=1
    ).values

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    clusters = model.fit_predict(X_final)

    result = df.copy()
    result["Cluster"] = clusters

    return result, model, scaler, encoder