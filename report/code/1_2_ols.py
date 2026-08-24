# 80:20 train/test split (as per the assignment instructions). The row indices
# are split alongside the data so Q3 can tell which rows are the outliers.
idx_all = np.arange(len(y))
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, idx_all, test_size=0.20, random_state=42)

ols = LinearRegression()      # OLS on the training split
ols.fit(X_train, y_train)

w0 = ols.intercept_           # fitted intercept
w1 = ols.coef_[0]             # fitted slope
mse_train = mean_squared_error(y_train, ols.predict(X_train))
mse_test  = mean_squared_error(y_test,  ols.predict(X_test))

print(f"Intercept (w0) = {w0:.4f}")
print(f"Slope     (w1) = {w1:.4f}")
print(f"MSE train      = {mse_train:.4f}")
print(f"MSE test       = {mse_test:.4f}")

# Flatten to 1-D for plotting
Xf, X_tr, X_te = (np.asarray(a).ravel() for a in (X, X_train, X_test))
y_f, y_tr, y_te = (np.asarray(a).ravel() for a in (y, y_train, y_test))

x_line = np.linspace(Xf.min() - 0.3, Xf.max() + 0.3, 200)

fig, ax = plt.subplots(figsize=(8, 5.5))

# Candidate lines
ax.plot(x_line, w0 + w1 * x_line, color='#ef6c00', linewidth=2,
        label=rf'OLS fit:  $y = {w0:.2f} + {w1:.2f}x$')
ax.plot(x_line, 2.5 + 1.8 * x_line, color='#2e7d32', linewidth=2, linestyle='--',
        label=r'True line:  $y = 2.5 + 1.8x$')

# Data
ax.scatter(X_tr, y_tr, s=45, color='#1f3b73', zorder=3,
           edgecolor='black', linewidth=0.6, label='Training data')
ax.scatter(X_te, y_te, s=55, marker='^', color='#00838f', zorder=3,
           edgecolor='black', linewidth=0.6, label='Test data')
ax.scatter(Xf[outlier_idx], y_f[outlier_idx], s=170, facecolors='none',
           edgecolors='#b71c1c', linewidth=1.8, zorder=4,
           label='Injected outliers')

# Residual of each outlier against the OLS fit
for i in np.atleast_1d(outlier_idx):
    ax.vlines(Xf[i], w0 + w1 * Xf[i], y_f[i],
              color='#b71c1c', linestyle=':', linewidth=1.6, zorder=2)

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('OLS Fit on Synthetic Data with Injected Outliers')
ax.annotate(rf'MSE$_\mathrm{{train}}$ = {mse_train:.3f}' '\n'
            rf'MSE$_\mathrm{{test}}$ = {mse_test:.3f}',
            xy=(0.98, 0.03), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=10, linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#b0b0b0', alpha=0.9))
ax.legend(framealpha=0.9, loc='upper left')

plt.tight_layout()
save_fig('q1_2_ols_fit_outliers')
plt.show()
