# Scale the second feature by 100, NO standardization
X_train_s = X_train.copy(); X_train_s[:, 1] *= 100
X_test_s  = X_test.copy();  X_test_s[:, 1]  *= 100

Xb_train_s = add_bias(X_train_s)
Xb_test_s  = add_bias(X_test_s)

# Condition number of the Hessian at w = 0, before vs after scaling
def cond_number(X, y):
    p = sigmoid(X @ np.zeros(X.shape[1]))
    H = (X.T * (p * (1 - p))) @ X / len(y)
    ev = np.linalg.eigvalsh(H)
    return ev[-1] / ev[0]

print(f"kappa(H) unscaled : {cond_number(Xb_train, yf_train):.1f}")
print(f"kappa(H) x2*100   : {cond_number(Xb_train_s, yf_train):.1e}")

# Same BGD settings as Q3.3
w_bad, loss_bad, iters_bad = batch_gradient_descent(Xb_train_s, yf_train, eta=0.1, T=300)

print("\nfirst 10 losses:", np.round(loss_bad[:10], 3))
print(f"final loss      : {loss_bad[-1]:.3e}")
print(f"test accuracy   : {accuracy(Xb_test_s, yf_test, w_bad):.4f}")
