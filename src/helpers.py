import numpy as np
import torch, statistics
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
from sklearn.metrics import confusion_matrix
import pandas as pd
from datetime import datetime



def train(model, train_loader, device, anchors, optimizer, num_epochs, dataset):
    """
    Parameters:
        model (nn.Model): ExtendedMask2Former model
        train_loader (DataLoader): Dataloader object for the training data.
        device (string): String value for the device we are going to use [cuda or cpu]
        anchors (tensor): Tensor containing the anchors
        optimizer (torch.optim): Optimizer of our choice
        num_epochs (int): Number of epochs for the training
    """
    # Define the dataframe we are going to use to save the accuracy metrics
    metrics_df = pd.DataFrame(columns=['epoch', 'loss', 'precision', 'recall', 'mAP', 'mAPCOCO'])

    
    
    for epoch in range(num_epochs):    
        model.train()
        per_epoch_predictions = []
        per_epoch_ground_truths = []
        per_epoch_loss = []
        
        for images, targets in train_loader:
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Concatenate and stack data
            batched_bboxes = torch.cat([t['boxes'] for t in targets]).to(device)
            batched_labels = torch.cat([t['labels'] for t in targets]).to(device)
            batched_masks  = torch.stack([t['masks'] for t in targets]).to(device)
            batched_mask_labels = torch.stack([t['mask_labels'] for t in targets]).to(device)
            
            # Get the predictions of the model and the actual data to feed to the loss function
            predictions = model(images, batched_masks)
            actual = {'boxes': batched_bboxes, 'labels': batched_labels, 'masks': batched_masks, 'mask_labels': batched_mask_labels}
            
            # Get the loss value and background propagate the value 
            loss = model.compute_loss(predictions, actual, anchors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Get the predictions, actual data and loss for each batch to calculate the epoch metrics
            per_epoch_loss.append(loss)
            per_epoch_predictions.append(predictions)
            per_epoch_ground_truths.append(actual)
        
        # Get Mean Average Precisions for IoU of [0.5] and [0.5-0.95] with Mean Average Precision
        map = model.calculate_map(predictions, targets, [0.5])
        mapCOCO = model.calculate_map(predictions, targets, torch.arange(0.5, 1.0, 0.05))
        loss = statistics.mean(per_epoch_loss)
        
        # Add epoch metrics
        new_row = {'epoch': epoch, 'loss': loss, 'precision': map['precision'], 'recall': map['recall'], 'mAP': map['mAP'], 'mAPCOCO': mapCOCO['mAP']}
        metrics_df.loc[len(metrics_df)] = new_row  # type: ignore
        
        # Save model info every 25 epochs to check progress with evaluation
        if epoch % 25 == 0 and epoch != 0: torch.save(model, 'results/model_{}_{}.pt'.format(dataset, epoch)) # type: ignore
        print('For the epoch:{} the loss is: {}, the precision is: {}, the recall is: {}, the mAP[0.5] is: {} and the mAP[0.5-0.95] is: {}'.format(loss, map['precision'], map['recall'], map['mAP'], mapCOCO['mAP']))
        





def evaluate_model(model, data_loader, device, anchors):
    model.eval()
    total_loss = 0
    all_preds = []

    with torch.no_grad():  # Disable gradient computation
        for images, targets in data_loader:
            images = torch.stack(images).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Concatenate and stack data
            batched_bboxes = torch.cat([t['boxes'] for t in targets]).to(device)
            batched_labels = torch.cat([t['labels'] for t in targets]).to(device)
            batched_masks  = torch.stack([t['masks'] for t in targets]).to(device)
            batched_mask_labels = torch.stack([t['mask_labels'] for t in targets]).to(device)
            
            # Get the predictions of the model and the actual data to feed to the loss function
            predictions = model(images, batched_masks)
            actual = {'boxes': batched_bboxes, 'labels': batched_labels, 'masks': batched_masks, 'mask_labels': batched_mask_labels}
            
            # Get the loss value and background propagate the value 
            loss = model.compute_loss(predictions, actual, anchors)
            total_loss += loss.item()

            # Assuming the model outputs predicted class labels in 'pred_labels'
            pred_labels = predictions['pred_mask_labels'].max(dim=1)[1]  # Get the argmax of class probabilities
            all_preds.extend(pred_labels.cpu().numpy())

        # Calculate average loss
        avg_loss = total_loss / len(data_loader)

    return avg_loss

