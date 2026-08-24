def sigmoid(z):
    """Numerically stable logistic sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0

    # z >= 0: use 1/(1+e^-z), where e^-z <= 1 so it cannot overflow
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))

    # z < 0: use e^z/(1+e^z), where e^z <= 1 so it cannot overflow either
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def bce_loss(z, y):
    """Binary cross-entropy from logits z, without ever forming p."""
    z = np.asarray(z, dtype=float)
    # ln(1+e^z) - y*z  is the whole per-sample loss (see the answer below)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def logits(X, w):
    """Linear scores z = Xw for the augmented design matrix X."""
    return X @ w


# Quick check against the closed-form values at z = 0
print(f"sigmoid(0)          = {sigmoid(np.array([0.0]))[0]:.4f}   (expect 0.5)")
print(f"bce_loss(0, y=1)    = {bce_loss(np.array([0.0]), np.array([1.0])):.4f}   "
      f"(expect ln 2 = {np.log(2):.4f})")
