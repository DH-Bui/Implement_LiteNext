def PlotAllMetrics(history):
    history_np = np.array(history)
    epochs = history_np[:, 0]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history_np[:, 2], label='Dice', linewidth=2)
    plt.plot(epochs, history_np[:, 3], label='IoU', linestyle='--')
    plt.plot(epochs, history_np[:, 4], label='Precision', alpha=0.7)
    plt.plot(epochs, history_np[:, 5], label='Recall', alpha=0.7)

    plt.title('Chỉ số đánh giá theo Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

PlotAllMetrics(final_history)
