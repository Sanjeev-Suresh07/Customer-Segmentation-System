from sklearn.metrics import silhouette_score


def calculate_silhouette_score(X, labels):
    """Calculate the Silhouette Score for clustering."""

    return silhouette_score(X, labels)