import warnings

from sklearn.model_selection import GridSearchCV, StratifiedKFold

# sklearn 1.8+ deprecates penalty= in favour of l1_ratio, but Q3 asks for
# penalty in {l1, l2}, so keep penalty= and hide the deprecation notices
warnings.filterwarnings("ignore", message="'penalty' was deprecated")
warnings.filterwarnings("ignore", message="Inconsistent values: penalty")

# Same pipeline as Q2, but with liblinear so both l1 and l2 are supported
grid_clf = Pipeline([
    ("pre", preprocessor),
    ("model", LogisticRegression(solver="liblinear", max_iter=1000)),
])

param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__penalty": ["l1", "l2"],
}

# 5-fold stratified CV keeps the class balance in every fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(grid_clf, param_grid, cv=cv, scoring="f1")
search.fit(X_train, y_train)

best_model = search.best_estimator_    # refit on the full training set

print(f"Best parameters : C = {search.best_params_['model__C']}, "
      f"penalty = {search.best_params_['model__penalty']}")
print(f"Best CV F1-score: {search.best_score_:.4f}")

# Mean CV F1 for every combination in the grid
print(f"\n{'C':>8}{'l1':>10}{'l2':>10}")
results = search.cv_results_
for C in param_grid["model__C"]:
    scores = []
    for penalty in ("l1", "l2"):
        i = next(k for k, p in enumerate(results["params"])
                 if p["model__C"] == C and p["model__penalty"] == penalty)
        scores.append(results["mean_test_score"][i])
    print(f"{C:>8}{scores[0]:>10.4f}{scores[1]:>10.4f}")
