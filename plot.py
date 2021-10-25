import matplotlib.pyplot as plt


def plot_training_history(history, title):
    """
    Plot model training performances.

    :param history: history produced by `model.fit`
    :returns: a tuple of (figure, axes) for further modification of the graph.
    """
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), sharex=True)

    epochs = range(len(history.history['f1_score']))

    # Plot training and validation loss.
    axs[0].plot(epochs, history.history['loss'],
                label="Train Loss", c='b')
    axs[0].plot(epochs, history.history['val_loss'],
                label="Validation Loss", c='orange')
    axs[0].set_ylim(0.0, 1.0)

    # Plot all the scores.
    axs[1].plot(epochs, history.history['f1_score'],
                label="Train F1 Score", c='b')
    axs[1].plot(epochs, history.history['recall_score'],
                label="Train Recall Score", c='b', linestyle='dashed')
    axs[1].plot(epochs, history.history['precision_score'],
                label="Train Precision Score", c='b', linestyle='dotted')

    axs[1].plot(epochs, history.history['val_f1_score'],
                label="Validation F1 Score", c='orange')
    axs[1].plot(epochs, history.history['val_recall_score'],
                label="Validation Recall Score", c='orange', linestyle='dashed')
    axs[1].plot(epochs, history.history['val_precision_score'],
                label="Validation Precision Score", c='orange', linestyle='dotted')

    for ax in axs:
        ax.legend()

    return fig, axs
