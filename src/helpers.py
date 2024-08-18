import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

# ------------------------------------------------------------------------

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms


# Assuming transforms and other required modules are already imported
def train(model, train_loader, device, anchors, optimizer, num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    all_metrics = {'precision': [], 'recall': [], 'AP': []}
    anchors = anchors.to(device)
    
    for epoch in range(num_epochs):
        
        model.train()
        
        for images, targets in train_loader:
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            gt_bboxes = targets[0]['boxes'].to(device)
            gt_labels = targets[0]['labels'].to(device)
            gt_masks  = targets[0]['masks'].to(device)
            actual = {"boxes": gt_bboxes, "labels": gt_labels, "masks": gt_masks}

            predictions = model(images, gt_masks)

            loss = model.compute_loss(predictions, actual, anchors)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')




def validate(model, val_loader, device, anchors, num_classes):
    """
    Parameters:
        - model (nn.Module): The model to be trained.
        - train_loader (DataLoader): DataLoader for the training data.
        - optimizer (Optimizer): The optimizer used for training the model.
        - device (str): The device on which the training will be performed (e.g., 'cpu', 'cuda').
        - anchors (Tensor): The anchor boxes used for bounding box predictions.
        - num_classes (int): The number of classes in the dataset.
    """
    model.eval()
    val_loss = 0
    all_metrics = {'precision': [], 'recall': [], 'AP': []}
    with torch.no_grad():
        for images, targets in val_loader:
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            loss = model.compute_loss(outputs, targets, anchors)
            val_loss += loss.item()

            precision, recall, ap, mAP = calculate_metrics(outputs, targets, num_classes)
            all_metrics['precision'].append(precision)
            all_metrics['recall'].append(recall)
            all_metrics['AP'].append(ap)

    return val_loss / len(val_loader), all_metrics, mAP


def test(model, test_loader, device, anchors, num_classes):
    """
    Parameters:
        - model (nn.Module): The model to be trained.
        - train_loader (DataLoader): DataLoader for the training data.
        - optimizer (Optimizer): The optimizer used for training the model.
        - device (str): The device on which the training will be performed (e.g., 'cpu', 'cuda').
        - anchors (Tensor): The anchor boxes used for bounding box predictions.
        - num_classes (int): The number of classes in the dataset.
    """
    model.eval()
    test_loss = 0
    all_metrics = {'precision': [], 'recall': [], 'AP': []}
    with torch.no_grad():
        for images, targets in test_loader:
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(images)
            loss = model.compute_loss(outputs, targets, anchors)
            test_loss += loss.item()

            precision, recall, ap, mAP = calculate_metrics(outputs, targets, num_classes)
            all_metrics['precision'].extend(precision)
            all_metrics['recall'].extend(recall)
            all_metrics['AP'].extend(ap)

    return test_loss / len(test_loader), all_metrics, mAP

# ------------------------------------------------------------------------



def calculate_metrics(predictions, targets, num_classes):
    all_true_labels = []
    all_pred_scores = []
    all_pred_labels = []

    for i in range(len(targets)):
        true_labels = targets[i]['labels'].cpu().numpy()
        pred_scores = predictions['pred_logits'][i].cpu().detach().numpy()

        # Ensure the number of predicted labels matches the number of true labels
        if len(pred_scores) >= len(true_labels):
            pred_scores = pred_scores[:len(true_labels)]
        else:
            true_labels = true_labels[:len(pred_scores)]

        pred_labels = pred_scores.argmax(axis=1)
        all_true_labels.extend(true_labels)
        all_pred_scores.extend(pred_scores)
        all_pred_labels.extend(pred_labels)

    all_true_labels = np.array(all_true_labels)
    all_pred_scores = np.array(all_pred_scores)
    all_pred_labels = np.array(all_pred_labels)

    precision_list = []
    recall_list = []
    ap_list = []

    for class_id in range(num_classes):
        # Binarize the labels for the current class
        true_class = (all_true_labels == class_id).astype(int)
        if np.sum(true_class) == 0:
            continue

        pred_class_scores = all_pred_scores[:, class_id]

        precision, recall, _ = precision_recall_curve(true_class, pred_class_scores)
        ap = average_precision_score(true_class, pred_class_scores)

        precision_list.append(precision)
        recall_list.append(recall)
        ap_list.append(ap)

    # Calculate mean average precision (mAP)
    mAP = np.mean(ap_list) if ap_list else 0

    return precision_list, recall_list, ap_list, mAP
