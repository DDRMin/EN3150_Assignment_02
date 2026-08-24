# Predicted probability of the positive class
y_proba = best_model.predict_proba(X_test)[:, 1]

# Default predict() is the same as thresholding at 0.50
assert np.array_equal((y_proba >= 0.50).astype(int), best_model.predict(X_test))


def metrics_at(t):
    """Precision, recall, F1 and confusion matrix at threshold t."""
    pred = (y_proba >= t).astype(int)
    return (precision_score(y_test, pred), recall_score(y_test, pred),
            f1_score(y_test, pred), confusion_matrix(y_test, pred))


print(f"{'Threshold':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}"
      f"{'TP':>6}{'FP':>5}{'FN':>5}{'TN':>5}")
for t in (0.50, 0.35):
    prec, rec, f1, cm_t = metrics_at(t)
    tn, fp, fn, tp = cm_t.ravel()
    print(f"{t:>10.2f}{prec:>11.4f}{rec:>9.4f}{f1:>9.4f}"
          f"{tp:>6}{fp:>5}{fn:>5}{tn:>5}")

# Change from 0.50 to 0.35
p50, r50, _, _ = metrics_at(0.50)
p35, r35, _, _ = metrics_at(0.35)
print(f"\nPrecision change: {p50:.4f} -> {p35:.4f}  ({p35 - p50:+.4f})")
print(f"Recall change   : {r50:.4f} -> {r35:.4f}  ({r35 - r50:+.4f})")
