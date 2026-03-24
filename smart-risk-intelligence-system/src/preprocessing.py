import numpy as np
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split


def load_data():
    """
    Load KDD Cup 99 dataset (cyber attack dataset)
    """

    data = fetch_kddcup99(percent10=True)

    X = data.data
    y = data.target

    # Convert bytes → integer labels
    y = np.array([0 if label == b'normal.' else 1 for label in y])

    return X, y


def preprocess_features(X):
    """
    Convert mixed data → numeric only
    """

    X_numeric = []

    for row in X:
        numeric_row = []
        for val in row:
            try:
                numeric_row.append(float(val))
            except:
                numeric_row.append(0.0)
        X_numeric.append(numeric_row)

    return np.array(X_numeric)


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    return (X - mean) / std


def prepare_data():
    X, y = load_data()

    X = preprocess_features(X)
    X = normalize(X)

    # Create regression target (risk score)
    y_score = y * 50 + np.random.randint(0, 50, size=len(y))

    # Classification labels
    y_class = y.astype(int)

    X_train, X_test, y_train, y_test, yc_train, yc_test = train_test_split(
        X, y_score, y_class, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, yc_train, yc_test