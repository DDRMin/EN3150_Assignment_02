# Coefficient magnitudes: real feature vs. the nine noise features

fig, ax = plt.subplots(figsize=(9.5, 5))

idx = np.arange(10)
width = 0.27

for i, (name, color) in enumerate(zip(('OLS', 'Ridge', 'Lasso'),
                                      ('#ef6c00', '#1f3b73', '#2e7d32'))):
    ax.bar(idx + (i - 1) * width, coefs[name], width, label=name,
           color=color, edgecolor='black', linewidth=0.5, zorder=3)

ax.axhline(0, color='black', linewidth=0.9, zorder=2)
ax.axvline(0.5, color='#b71c1c', linestyle='--', linewidth=1.3, zorder=2)

ax.set_xticks(idx)
ax.set_xticklabels(['$x$\n(real)'] + [f'$n_{i}$' for i in range(1, 10)])
ax.set_xlabel('Feature')
ax.set_ylabel('Fitted coefficient (standardized features)')
ax.set_title('Coefficients on 1 Informative + 9 Pure-Noise Features')
ax.legend(framealpha=0.9, loc='upper right')

plt.tight_layout()
save_fig('q1_4_coefficients')
plt.show()
