def add_bias(X):
    """Prepend a column of ones so w[0] is the bias term."""
    X = np.asarray(X, dtype=float)
    return np.column_stack([np.ones(len(X)), X])


def grad_J(X, y, w):
    """Gradient of the BCE loss, (1/N) X^T (p - y), from Q1."""
    return X.T @ (sigmoid(X @ w) - y) / len(y)


def accuracy(X, y, w):
    """Fraction correct using the 0.5 decision threshold."""
    return float(np.mean((sigmoid(X @ w) >= 0.5) == y))


# Augmented design matrices and float labels, used from here on
Xb_train = add_bias(X_train)
Xb_test = add_bias(X_test)
yf_train = y_train.astype(float)
yf_test = y_test.astype(float)

print(f"Xb_train: {Xb_train.shape}, Xb_test: {Xb_test.shape}")
