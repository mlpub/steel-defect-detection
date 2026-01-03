import torch
import pandas as pd


def train_one_epoch(model, loader, criterion, optimizer, device, threshold=0.5):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_preds = []
    all_labels = []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        labels = labels.float()
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)  # Ensure shape [batch, 1]

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward() # compute gradients
        optimizer.step() # update weights

        total_loss += loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > threshold).float()   

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    # concat all batches
    all_probs = torch.cat(all_probs).detach().numpy().ravel()
    all_preds = torch.cat(all_preds).detach().numpy().ravel()
    all_labels = torch.cat(all_labels).detach().numpy().ravel()

    #accuracy = correct / total
    return total_loss / len(loader), all_labels, all_preds, all_probs



def validate_one_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            labels = labels.float()
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)  # Ensure shape [batch, 1]

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # concat all batches
    all_probs = torch.cat(all_probs).numpy().ravel()
    all_preds = torch.cat(all_preds).numpy().ravel()
    all_labels = torch.cat(all_labels).numpy().ravel()

    #accuracy = correct / total
    return total_loss / len(loader), all_labels, all_preds, all_probs


def build_metric_df(
    arr_train_loss, arr_train_acc, arr_train_precision, arr_train_recall, arr_train_f1, arr_train_rocauc, arr_train_conf_matrix,
    arr_val_loss, arr_val_acc, arr_val_precision, arr_val_recall, arr_val_f1, arr_val_rocauc, arr_val_conf_matrix
):
    """
    Build a DataFrame from lists of metrics for train and validation.
    """
    
    df_metrics = pd.DataFrame({
        'train_loss': arr_train_loss,
        'train_acc': arr_train_acc,
        'train_precision': arr_train_precision,
        'train_recall': arr_train_recall,
        'train_f1': arr_train_f1,
        'train_rocauc': arr_train_rocauc,
        'train_conf_matrix': arr_train_conf_matrix,

        'val_loss': arr_val_loss,
        'val_acc': arr_val_acc,
        'val_precision': arr_val_precision,
        'val_recall': arr_val_recall,
        'val_f1': arr_val_f1,
        'val_rocauc': arr_val_rocauc,
        'val_conf_matrix': arr_val_conf_matrix
    })
    # Extract confusion matrix values into separate columns for train and val
    df_metrics[['train_tn', 'train_fp', 'train_fn', 'train_tp']] = pd.DataFrame(
        df_metrics['train_conf_matrix'].tolist(), index=df_metrics.index
    )
    df_metrics[['val_tn', 'val_fp', 'val_fn', 'val_tp']] = pd.DataFrame(
        df_metrics['val_conf_matrix'].tolist(), index=df_metrics.index
    )
    return df_metrics

def build_metric_df_train_only(
    arr_train_loss, arr_train_acc, arr_train_precision, arr_train_recall, arr_train_f1, arr_train_rocauc, arr_train_conf_matrix,
    
):
    """
    Build a DataFrame from lists of metrics for train and validation.
    """
    
    df_metrics = pd.DataFrame({
        'train_loss': arr_train_loss,
        'train_acc': arr_train_acc,
        'train_precision': arr_train_precision,
        'train_recall': arr_train_recall,
        'train_f1': arr_train_f1,
        'train_rocauc': arr_train_rocauc,
        'train_conf_matrix': arr_train_conf_matrix,

    })
    # Extract confusion matrix values into separate columns for train and val
    df_metrics[['train_tn', 'train_fp', 'train_fn', 'train_tp']] = pd.DataFrame(
        df_metrics['train_conf_matrix'].tolist(), index=df_metrics.index
    )

    return df_metrics

    
