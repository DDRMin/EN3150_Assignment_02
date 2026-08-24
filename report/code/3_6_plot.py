fig, ax = plt.subplots(figsize=(7.5, 5.5))

# Plot Batch GD
ax.plot(np.arange(len(loss_bgd)), loss_bgd, color='#1f3b73', linewidth=2,
        marker='o', markersize=5, markevery=15,
        label=f'Batch GD ({iters_bgd} iters)')

# Plot Newton's Method
ax.plot(np.arange(len(loss_newton)), loss_newton, color='#b71c1c', linewidth=2.5,
        marker='X', markersize=7, markevery=1,
        label=f"Newton's Method ({iters_newton} iters)")

# Set logarithmic scale for the y-axis
ax.set_yscale('log')

ax.set_xlabel('Iteration', fontsize=12)
ax.set_ylabel(r'Loss $J(\mathbf{w})$ (log scale)', fontsize=12)
ax.set_title('Convergence: Batch Gradient Descent vs Newton\'s Method', fontsize=14)


ax.legend(framealpha=0.9, loc='upper right', fontsize=11)

plt.tight_layout()
save_fig('q3_6_convergence')
plt.show()
