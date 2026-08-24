# OLS / Ridge / Lasso on 1 informative + 9 pure-noise features

from sklearn.model_selection import GridSearchCV

# Nine pure noise variables appended to the single real feature. This draws from
# the same rng stream as Listing 1, so the cells must be run in order.
X_noise = rng.normal(size=(X.shape[0], 9))
X10 = np.column_stack([X, X_noise])      # 160 x 10: column 0 real, columns 1-9 noise

# Same 80:20 split and seed as Q2, so the two questions stay comparable
X10_train, X10_test, y10_train, y10_test = train_test_split(
    X10, y, test_size=0.20, random_state=42)
assert np.array_equal(y10_train, y_train)   # confirms it is the same partition

alphas = [1e-3, 1e-2, 1e-1, 1.0, 10.0]      # regularization strengths to search

# Each estimator sits behind a StandardScaler in a Pipeline, so the scaler is
# refitted inside every CV fold - fitting it once on the whole training set
# would leak validation-fold statistics into training. OLS has no alpha to tune.
models = {
    'OLS':   (LinearRegression(),    {}),
    'Ridge': (Ridge(),               {'reg__alpha': alphas}),
    'Lasso': (Lasso(max_iter=50000), {'reg__alpha': alphas}),
}

rows, coefs = [], {}
for name, (est, grid) in models.items():
    pipe = Pipeline([('scale', StandardScaler()), ('reg', est)])
    search = GridSearchCV(pipe, grid, cv=5, scoring='neg_mean_squared_error')
    search.fit(X10_train, y10_train)

    best_alpha = search.best_params_.get('reg__alpha')       # None for OLS
    cv_mse   = -search.best_score_                           # mean 5-fold CV MSE
    test_mse = mean_squared_error(y10_test, search.predict(X10_test))

    # Clip numerically tiny values to exactly zero so Lasso's dropped features
    # are unambiguous
    w = search.best_estimator_.named_steps['reg'].coef_
    w = np.where(np.abs(w) < 1e-8, 0.0, w)
    coefs[name] = w
    n_zero = int(np.sum(w[1:] == 0.0))       # noise coefficients driven to zero
    rows.append((name, best_alpha, cv_mse, test_mse, n_zero))

print(f"{'Model':<8}{'alpha*':>9}{'CV MSE':>11}{'Test MSE':>11}{'noise coefs = 0':>18}")
for name, alpha, cv_mse, test_mse, n_zero in rows:
    alpha_str = '-' if alpha is None else f'{alpha:g}'
    print(f"{name:<8}{alpha_str:>9}{cv_mse:>11.4f}{test_mse:>11.4f}"
          f"{f'{n_zero} / 9':>18}")

# Coefficients on the standardized features, so magnitudes are comparable
print("\nFitted coefficients (feature 0 is the real one, 1-9 are pure noise):")
print(f"{'':<8}" + ''.join(f'{i:>8}' for i in range(10)))
for name in models:
    print(f"{name:<8}" + ''.join(f'{c:>8.3f}' for c in coefs[name]))
