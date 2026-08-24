def batch_gradient_descent(X, y, eta=0.1, T=300, tol=1e-8):
    """Batch GD from w = 0, stopping early when the loss stops changing."""
    w = np.zeros(X.shape[1])
    history = [bce_loss(X @ w, y)]     # J_0, the loss before any update
    n_iter = 0

    for t in range(1, T + 1):
        w = w - eta * grad_J(X, y, w)          # w_(t+1) = w_(t) - eta * grad
        history.append(bce_loss(X @ w, y))
        n_iter = t
        if abs(history[-1] - history[-2]) < tol:
            break

    return w, np.array(history), n_iter


w_bgd, loss_bgd, iters_bgd = batch_gradient_descent(
    Xb_train, yf_train, eta=0.1, T=300, tol=1e-8)

print(f"Final weights    : w0 = {w_bgd[0]:.6f}, w1 = {w_bgd[1]:.6f}, "
      f"w2 = {w_bgd[2]:.6f}")
print(f"Iterations        : {iters_bgd}")
print(f"Early stopped     : {iters_bgd < 300}   "
      f"(|J_t - J_t-1| = {abs(loss_bgd[-1] - loss_bgd[-2]):.3e}, tol = 1e-08)")
print(f"Initial loss J_0  : {loss_bgd[0]:.8f}")
print(f"Final loss J_T    : {loss_bgd[-1]:.8f}")
print(f"Train accuracy    : {accuracy(Xb_train, yf_train, w_bgd):.4f}")
print(f"Test accuracy     : {accuracy(Xb_test, yf_test, w_bgd):.4f}")
