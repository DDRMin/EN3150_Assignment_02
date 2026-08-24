from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)

# Predict with the best model found by the Q3 grid search
y_pred = best_model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion matrix:\n{cm}")

tn, fp, fn, tp = cm.ravel()
print(f"\nTN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")

# Accuracy of a baseline that always predicts the majority (negative) class
baseline = 1 - y_test.mean()
print(f"\nMajority-class baseline accuracy: {baseline:.4f}")
