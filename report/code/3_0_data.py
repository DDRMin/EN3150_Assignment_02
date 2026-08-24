import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Two Gaussian blobs, one per class
X, y = make_blobs(
    n_samples=1200,
    centers=[[-2.2, -0.5], [2.0, 1.2]],
    cluster_std=[1.45, 1.65],
    random_state=8
)

# Stratified 80:20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train: {X_train.shape}, test: {X_test.shape}")
print(f"Positive class share - train: {y_train.mean():.3f}, test: {y_test.mean():.3f}")
