def visualize_loss(epochs, loss, label, title, training=True):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss[0], label=label[0], marker='o')
    plt.plot(epochs, loss[1], label=label[1], marker='s')
    if len(label) > 2:
        plt.plot(epochs, loss[2], label=label[2], marker='^')

    # Labels and title
    plt.xlabel('Epoch')
    if training:
        plt.ylabel('Training Loss')
    elif training is False:
        plt.ylabel('Validation Loss')
    else:
        plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_train_val_loss(train_loss, val_loss, epochs, label=['Train Loss', 'Validation Loss'], title='Training and Validation Loss'):
    """
    Plot training and validation loss over epochs.

    Parameters:
    - train_loss: List or array of training loss values.
    - val_loss: List or array of validation loss values.
    - epochs: List or array of epoch numbers.
    - label: List of labels for the two curves.
    - title: Title of the chart.
    """
    visualize_loss(epochs, [train_loss, val_loss], label, title, training=None)

def visualize_f1(data1, data2, labels, label1='Keras', label2='Scratch', ylabel='f1_score', title='Comparison'):
    import matplotlib.pyplot as plt
    import numpy as np
    """
    Plot side-by-side bar chart comparing two arrays.

    Parameters:
    - data1: List or array of first category values.
    - data2: List or array of second category values.
    - labels: List of labels for each bar group (x-axis categories).
    - label1: Label for the first dataset (legend).
    - label2: Label for the second dataset (legend).
    - ylabel: Label for the y-axis.
    - title: Title of the chart.
    """
    x = np.arange(len(labels))  # label locations
    width = 0.35  # width of each bar

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, data1, width, label=label1)
    bars2 = ax.bar(x + width/2, data2, width, label=label2)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    # ax.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.05))

    plt.tight_layout()
    plt.show()
