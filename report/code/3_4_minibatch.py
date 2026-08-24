def minibatch_gradient_descent(X, y, B=32, eta=0.05, E=100, seed=0):
    """Mini-batch GD from w = 0, reshuffling the indices every epoch."""
    rng_mb = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    history = [bce_loss(X @ w, y)]     # J_0, before any update
    N = len(y)

    for epoch in range(1, E + 1):
        idx = rng_mb.permutation(N)            # fresh shuffle each epoch
        for start in range(0, N, B):
            batch = idx[start:start + B]       # one mini-batch of at most B rows
            w = w - eta * grad_J(X[batch], y[batch], w)
        history.append(bce_loss(X @ w, y))     # full training loss once per epoch

    return w, np.array(history), E


w_mbgd, loss_mbgd, epochs_mbgd = minibatch_gradient_descent(
    Xb_train, yf_train, B=32, eta=0.05, E=100)

updates_per_epoch = int(np.ceil(len(yf_train) / 32))

print(f"Batch size B      : 32  ({updates_per_epoch} updates per epoch)")
print(f"Final weights     : w0 = {w_mbgd[0]:.6f}, w1 = {w_mbgd[1]:.6f}, "
      f"w2 = {w_mbgd[2]:.6f}")
print(f"Epochs            : {epochs_mbgd}")
print(f"Final loss J_E    : {loss_mbgd[-1]:.8f}")
print(f"Train accuracy    : {accuracy(Xb_train, yf_train, w_mbgd):.4f}")
print(f"Test accuracy     : {accuracy(Xb_test, yf_test, w_mbgd):.4f}")

print(f"\nAfter 100 passes over the data:")
print(f"  Batch GD loss      : {loss_bgd[100]:.8f}")
print(f"  Mini-batch GD loss : {loss_mbgd[100]:.8f}")
