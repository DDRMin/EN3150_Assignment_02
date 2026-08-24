fig, ax = plt.subplots(figsize=(8, 5.5))

# One BGD iteration and one MBGD epoch both cost a single pass over the data,
# so plotting them on a shared x-axis is a fair comparison
ax.plot(np.arange(len(loss_bgd)), loss_bgd, color='#1f3b73', linewidth=2,
        label=f'Batch GD ($\\eta = 0.1$, {iters_bgd} iterations)')
ax.plot(np.arange(len(loss_mbgd)), loss_mbgd, color='#ef6c00', linewidth=2,
        label=f'Mini-batch GD ($B = 32$, $\\eta = 0.05$, {epochs_mbgd} epochs)')

ax.axvline(epochs_mbgd, color='#b71c1c', linestyle=':', linewidth=1.4,
           label=f'End of mini-batch training (epoch {epochs_mbgd})')

ax.set_xlabel('Pass over the training set  (BGD iteration / MBGD epoch)')
ax.set_ylabel(r'Training loss $J(\mathbf{w})$')
ax.set_title('Training Loss Progression: Mini-Batch vs Batch Gradient Descent')
ax.legend(framealpha=0.9, loc='upper right')

plt.tight_layout()
save_fig('q3_4_minibatch')
plt.show()
