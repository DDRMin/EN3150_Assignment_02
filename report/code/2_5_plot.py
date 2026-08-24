from sklearn.metrics import precision_recall_curve

prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)

fig, ax = plt.subplots(figsize=(7.5, 5.5))

ax.plot(rec_curve, prec_curve, color='#1f3b73', linewidth=2,
        label='Precision-Recall curve')

# Mark the two operating points
for t, color, marker in ((0.50, '#ef6c00', 'o'), (0.35, '#b71c1c', 'X')):
    prec, rec, _, _ = metrics_at(t)
    ax.scatter(rec, prec, s=150, color=color, marker=marker, zorder=4,
               edgecolor='black', linewidth=0.8,
               label=rf'$p_{{th}} = {t:.2f}$:  P = {prec:.3f}, R = {rec:.3f}')

# A classifier with no skill sits at the positive-class rate
ax.axhline(y_test.mean(), color='#546e7a', linestyle=':', linewidth=1.5,
           label=f'No-skill baseline ({y_test.mean():.3f})')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Trade-off on the Test Set')
ax.set_ylim(0, 1.05)
ax.legend(framealpha=0.9, loc='lower left', fontsize=9.5)

plt.tight_layout()
save_fig('q2_5_precision_recall')
plt.show()
