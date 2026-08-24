# ---------------------------------------------------------------
# WLS vs OLS fitted lines
# ---------------------------------------------------------------
Xf   = np.asarray(X).ravel()
X_tr = np.asarray(X_train).ravel()
X_te = np.asarray(X_test).ravel()

x_line = np.linspace(Xf.min() - 0.3, Xf.max() + 0.3, 200)

fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(x_line, w_ols[0] + w_ols[1] * x_line, color='#ef6c00', linewidth=2,
        label=rf'OLS:  $y = {w_ols[0]:.2f} + {w_ols[1]:.2f}x$')
ax.plot(x_line, w_wls[0] + w_wls[1] * x_line, color='#1f3b73', linewidth=2,
        label=rf'WLS ($a_i = 0.05$):  $y = {w_wls[0]:.2f} + {w_wls[1]:.2f}x$')
ax.plot(x_line, 2.5 + 1.8 * x_line, color='#2e7d32', linewidth=2, linestyle='--',
        label=r'True line:  $y = 2.5 + 1.8x$')

ax.scatter(X_tr[~out_train], y_train[~out_train], s=42, color='#546e7a',
           zorder=3, edgecolor='black', linewidth=0.5, alpha=0.8,
           label='Training inliers')
ax.scatter(X_te[~out_test], y_test[~out_test], s=52, marker='^',
           color='#00838f', zorder=3, edgecolor='black', linewidth=0.5,
           alpha=0.8, label='Test inliers')
ax.scatter(Xf[is_out], y[is_out], s=170, marker='X', color='#b71c1c',
           zorder=4, edgecolor='black', linewidth=0.8,
           label=r'Down-weighted outliers')

ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Weighted vs. Ordinary Least Squares under Known Outliers')
ax.legend(framealpha=0.9, loc='upper left', fontsize=9.5)

plt.tight_layout()
save_fig('q1_3_wls_vs_ols')
plt.show()
