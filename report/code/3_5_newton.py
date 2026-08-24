def hessian_J(X, w):
    """Hessian of BCE loss: (1/N) X^T S X. 
    Computed efficiently without building the huge N x N diagonal matrix S."""
    p = sigmoid(X @ w)
    s = p * (1 - p)
    # (X.T * s) broadcasts the multiplication across columns, equivalent to X.T @ diag(s)
    return (X.T * s) @ X / len(X)

def newtons_method(X, y, T=20, tol=1e-8, damp=1e-6):
    """Damped Newton's method for logistic regression."""
    D_plus_1 = X.shape[1]
    w = np.zeros(D_plus_1)
    history = [bce_loss(X @ w, y)]
    n_iter = 0
    
    for t in range(1, T + 1):
        g = grad_J(X, y, w)
        H = hessian_J(X, w)
        
        # Add damping term to guarantee numerical invertibility
        H_damped = H + damp * np.eye(D_plus_1)
        
        # Solve H_damped * delta = g  (much more stable than explicit inverse)
        delta = np.linalg.solve(H_damped, g)
        
        w = w - delta
        history.append(bce_loss(X @ w, y))
        n_iter = t
        
        if abs(history[-1] - history[-2]) < tol:
            break
            
    return w, np.array(history), n_iter

w_newton, loss_newton, iters_newton = newtons_method(Xb_train, yf_train, T=20, tol=1e-8)

print(f"Final weights    : w0 = {w_newton[0]:.6f}, w1 = {w_newton[1]:.6f}, w2 = {w_newton[2]:.6f}")
print(f"Iterations       : {iters_newton}")
print(f"Early stopped    : {iters_newton < 20}   (|J_t - J_t-1| = {abs(loss_newton[-1] - loss_newton[-2]):.3e})")
print(f"Final loss J_T   : {loss_newton[-1]:.8f}")
print(f"Test accuracy    : {accuracy(Xb_test, yf_test, w_newton):.4f}")
