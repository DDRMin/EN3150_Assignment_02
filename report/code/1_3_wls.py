# ---------------------------------------------------------------
# NumPy implementation of the OLS and WLS closed-form estimators
# ---------------------------------------------------------------

# Boolean outlier mask over the full dataset, then mapped onto each split
is_out = np.zeros(len(y), dtype=bool)
is_out[outlier_idx] = True

out_train = is_out[idx_train]   # which training rows are outliers
out_test  = is_out[idx_test]    # which test rows are outliers

print(f"Outliers in training set: {out_train.sum()} / {len(out_train)}")
print(f"Outliers in test set    : {out_test.sum()} / {len(out_test)}")


def design_matrix(x):
    """Return X with the lead column of ones for the intercept bias."""
    x = np.asarray(x).reshape(len(x), -1)
    return np.column_stack([np.ones(len(x)), x])


Xd_train = design_matrix(X_train)
Xd_test  = design_matrix(X_test)

# Sample-specific weights: a_i = 0.05 for known outliers, a_i = 1.0 for inliers
a = np.where(out_train, 0.05, 1.0)
A = np.diag(a)                  # diagonal weighting matrix

# OLS :  w = (X^T X)^-1 X^T y        (the special case A = I)
# WLS :  w = (X^T A X)^-1 X^T A y
# np.linalg.solve is the numerically stable way to evaluate these closed forms
w_ols = np.linalg.solve(Xd_train.T @ Xd_train, Xd_train.T @ y_train)
w_wls = np.linalg.solve(Xd_train.T @ A @ Xd_train, Xd_train.T @ A @ y_train)

# Check the closed-form OLS reproduces the sklearn fit from Q2
assert np.allclose(w_ols, [w0, w1]), "Closed-form OLS disagrees with sklearn"


def mse(Xd, y_true, w, mask=None):
    """Mean squared error, optionally over a subset of rows."""
    resid = y_true - Xd @ w
    if mask is not None:
        resid = resid[mask]
    return float(np.mean(resid ** 2))


# Overall test MSE is dominated by the outliers WLS deliberately ignores, so the
# inlier-only MSE is reported too - that is what shows the benefit of weighting
print(f"\n{'Model':<6}{'w0':>9}{'w1':>9}{'MSE_train':>12}{'MSE_test':>11}"
      f"{'MSE_test (inliers)':>21}")
for name, w in (("OLS", w_ols), ("WLS", w_wls)):
    print(f"{name:<6}{w[0]:>9.4f}{w[1]:>9.4f}"
          f"{mse(Xd_train, y_train, w):>12.4f}"
          f"{mse(Xd_test, y_test, w):>11.4f}"
          f"{mse(Xd_test, y_test, w, ~out_test):>21.4f}")
print(f"{'True':<6}{2.5:>9.4f}{1.8:>9.4f}")

print("\nEuclidean distance from the true parameters [2.5, 1.8]:")
print(f"  ||w_ols - w_true|| = {np.linalg.norm(w_ols - [2.5, 1.8]):.4f}")
print(f"  ||w_wls - w_true|| = {np.linalg.norm(w_wls - [2.5, 1.8]):.4f}")
