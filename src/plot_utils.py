import matplotlib.pyplot as plt

def plot_train_val_metrics(df_metrics, exp_code=None, save_path=None):
    """
    Plots loss, accuracy, precision, recall, F1, and ROC AUC for train and val from a metrics DataFrame.
    Optionally saves the figure to a file.
    """
    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    if exp_code:
        fig.suptitle(f'Train and Val Metrics per Epoch - {exp_code}', fontsize=14, y=1.03)
    # Loss
    axs[0, 0].plot(df_metrics['train_loss'], label='Train Loss')
    axs[0, 0].plot(df_metrics['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[0, 0].grid(True)
    # Accuracy
    axs[0, 1].plot(df_metrics['train_acc'], label='Train Accuracy')
    axs[0, 1].plot(df_metrics['val_acc'], label='Val Accuracy')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[0, 1].grid(True)
    # Precision
    axs[1, 0].plot(df_metrics['train_precision'], label='Train Precision')
    axs[1, 0].plot(df_metrics['val_precision'], label='Val Precision')
    axs[1, 0].set_title('Precision')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].legend()
    axs[1, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[1, 0].grid(True)
    # Recall
    axs[1, 1].plot(df_metrics['train_recall'], label='Train Recall')
    axs[1, 1].plot(df_metrics['val_recall'], label='Val Recall')
    axs[1, 1].set_title('Recall')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].legend()
    axs[1, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[1, 1].grid(True)
    # F1 Score
    axs[2, 0].plot(df_metrics['train_f1'], label='Train F1')
    axs[2, 0].plot(df_metrics['val_f1'], label='Val F1')
    axs[2, 0].set_title('F1 Score')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('F1 Score')
    axs[2, 0].legend()
    axs[2, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[2, 0].grid(True)
    # ROC AUC
    axs[2, 1].plot(df_metrics['train_rocauc'], label='Train ROC AUC')
    axs[2, 1].plot(df_metrics['val_rocauc'], label='Val ROC AUC')
    axs[2, 1].set_title('ROC AUC')
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('ROC AUC')
    axs[2, 1].legend()
    axs[2, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[2, 1].grid(True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)


def plot_train_metrics(df_metrics, exp_code=None, save_path=None):
    """
    Plots loss, accuracy, precision, recall, F1, and ROC AUC for train and val from a metrics DataFrame.
    Optionally saves the figure to a file.
    """
    fig, axs = plt.subplots(3, 2, figsize=(16, 14))
    if exp_code:
        fig.suptitle(f'Train and Val Metrics per Epoch - {exp_code}', fontsize=14, y=1.03)
    # Loss
    axs[0, 0].plot(df_metrics['train_loss'], label='Train Loss')
    #axs[0, 0].plot(df_metrics['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[0, 0].grid(True)
    # Accuracy
    axs[0, 1].plot(df_metrics['train_acc'], label='Train Accuracy')
    #axs[0, 1].plot(df_metrics['val_acc'], label='Val Accuracy')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[0, 1].grid(True)
    # Precision
    axs[1, 0].plot(df_metrics['train_precision'], label='Train Precision')
    #axs[1, 0].plot(df_metrics['val_precision'], label='Val Precision')
    axs[1, 0].set_title('Precision')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].legend()
    axs[1, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[1, 0].grid(True)
    # Recall
    axs[1, 1].plot(df_metrics['train_recall'], label='Train Recall')
    #axs[1, 1].plot(df_metrics['val_recall'], label='Val Recall')
    axs[1, 1].set_title('Recall')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].legend()
    axs[1, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[1, 1].grid(True)
    # F1 Score
    axs[2, 0].plot(df_metrics['train_f1'], label='Train F1')
    #axs[2, 0].plot(df_metrics['val_f1'], label='Val F1')
    axs[2, 0].set_title('F1 Score')
    axs[2, 0].set_xlabel('Epoch')
    axs[2, 0].set_ylabel('F1 Score')
    axs[2, 0].legend()
    axs[2, 0].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[2, 0].grid(True)
    # ROC AUC
    axs[2, 1].plot(df_metrics['train_rocauc'], label='Train ROC AUC')
    #axs[2, 1].plot(df_metrics['val_rocauc'], label='Val ROC AUC')
    axs[2, 1].set_title('ROC AUC')
    axs[2, 1].set_xlabel('Epoch')
    axs[2, 1].set_ylabel('ROC AUC')
    axs[2, 1].legend()
    axs[2, 1].xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[2, 1].grid(True)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()
    plt.close(fig)