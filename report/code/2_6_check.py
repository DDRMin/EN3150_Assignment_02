# Numerical check of the hand calculations
b0, b1, b2 = -7.2, 0.08, 1.4

# (a) p at x1 = 45 study hours, x2 = 3.4 GPA
z = b0 + b1 * 45 + b2 * 3.4
p = 1 / (1 + np.exp(-z))
print(f"(a) z = {z:.4f},  p = {p:.4f}")

# (b) study hours needed for p = 0.70 at x2 = 3.4
z_target = np.log(0.70 / (1 - 0.70))          # logit of the target probability
x1 = (z_target - b0 - b2 * 3.4) / b1
print(f"(b) z = {z_target:.4f},  x1 = {x1:.4f} hours")
