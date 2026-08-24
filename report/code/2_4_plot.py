fig, ax = plt.subplots(figsize=(5.5, 5))

ConfusionMatrixDisplay(cm, display_labels=["negative (0)", "positive (1)"]).plot(
    ax=ax, cmap="Blues", colorbar=False)

ax.grid(False)      # the global grid setting would draw over the cells
ax.set_title("Confusion Matrix (Test Set)")

plt.tight_layout()
save_fig("q2_4_confusion_matrix")
plt.show()
