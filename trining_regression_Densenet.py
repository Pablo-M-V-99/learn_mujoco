import os
import wandb
import torch
import datetime
import numpy as np
import torch.nn as nn             # neural network modules
import torch.optim as optim       # for the optimization algorithms
from model_Densenet import SiameseDenseNet, ModelType
from torch.utils.data import DataLoader         # easier dataset management, creates mini batches
from classificazioneDataset import DeformationDataset
from sklearn.metrics import (recall_score, precision_score, accuracy_score, mean_squared_error,
                             mean_absolute_error, r2_score)


def validation_epoch(model, validation_dataloader, criterion, classification):
    running_validation_loss = 0.0
    with torch.no_grad():
        model.eval()
        if classification:
            classes_predicted = torch.tensor([], dtype=torch.int)
        else:
            validation_predictions = []
        validation_labels = torch.tensor([])
        for batch_number, (img1, img2, labels) in enumerate(validation_dataloader):
            img1 = img1.to(device)
            img2 = img2.to(device)
            validation_labels = torch.cat([validation_labels, labels], dim=0)
            labels = labels.to(device)  # true value
            # Forward pass
            outputs = model(img1, img2)  # prediction

            if classification:
                validation_loss = sum(criterion(outputs[:, i], labels[:, i]) for i in range(5))
                classes_predicted = torch.cat([classes_predicted, torch.argmax(outputs, dim=-1).cpu()], dim=0, )

            else:
                validation_loss = criterion(outputs, labels)
                validation_predictions.append(outputs)
            running_validation_loss += validation_loss.item()

        running_validation_loss /= len(validation_dataloader)

        if classification:
            validation_accuracy = [accuracy_score(validation_labels[:, head], classes_predicted[:, head]) for head in
                                   range(outputs.shape[1])]
            validation_precision = [
                precision_score(validation_labels[:, head], classes_predicted[:, head], average=None) for head
                in range(outputs.shape[1])]
            validation_recalls = [recall_score(validation_labels[:, head], classes_predicted[:, head], average=None) for
                                  head in
                                  range(outputs.shape[1])]
            return running_validation_loss, validation_accuracy, validation_precision, validation_recalls, validation_labels, classes_predicted
        else:
            validation_predictions = torch.cat(validation_predictions, dim=0)
            validation_MSE = mean_squared_error(validation_labels.cpu(), validation_predictions.cpu())
            validation_MAE = mean_absolute_error(validation_labels.cpu(), validation_predictions.cpu())
            validation_R2 = r2_score(validation_labels.cpu(), validation_predictions.cpu())
            return running_validation_loss, validation_MSE, validation_MAE, validation_R2, validation_labels, validation_predictions


wandb.require("core")

use_wandb = True

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 16
max_epochs = 10
max_epochs_no_improvement = 6
learning_rate = 1e-3
max_batch_size = 16
validation_batch_size = 64
accumulation_steps = batch_size // max_batch_size if batch_size > max_batch_size else 1

hyperparams = {'regression': True,
               'learning_rate': learning_rate,
               'batch_size': batch_size,
               'validation_batch_size': validation_batch_size, }

if use_wandb:
    wandb.init(project='estimate_deformations',
               entity='Pablo_MV',
               save_code=False,
               config=hyperparams)

model = SiameseDenseNet(model_type=ModelType.regression).to(device)

train_dataset = DeformationDataset(
    # datasets_path='datasets',
    datasets_path="/media/brahe/dataset_Pablo/training",
    training=True,
    classification=False,
    max_translation_x=0,
    max_translation_y=0,
    max_rotation=np.deg2rad(0),
    flip_left_right=True,
    gradient_thresholding=True)

validation_dataset = DeformationDataset(
    # datasets_path='datasets',
    datasets_path="/media/brahe/dataset_Pablo/validation",
    training=False,
    classification=False)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=True,
                          num_workers=os.cpu_count() // 2)

validation_loader = DataLoader(dataset=validation_dataset,
                               batch_size=validation_batch_size,
                               shuffle=False,
                               drop_last=False,
                               num_workers=os.cpu_count() // 2)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
best_loss = torch.inf
epochs_no_improvement = 0

for epoch in range(max_epochs):
    print(f'\nEpoch: {epoch + 1}/{max_epochs}')

    model.train()
    running_train_loss = 0.0
    train_accuracy = torch.zeros(4)

    for batch_number, (img1, img2, labels) in enumerate(train_loader):
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)  # etichetta 'corretta'
        # Forward pass
        outputs = model(img1, img2)  # predizioni del modello
        loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()
        if ((batch_number + 1) * max_batch_size) % batch_size == 0 or batch_size < max_batch_size:
            optimizer.step()
            optimizer.zero_grad()

        running_train_loss += loss.item()

    train_accuracy /= len(train_dataset)
    running_train_loss /= len(train_loader)
    running_train_loss *= accumulation_steps

    running_validation_loss, validation_MSE, validation_MAE, validation_R2, validation_labels, validation_predictions = \
        (validation_epoch(model, validation_loader, criterion, False))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{current_time}: {running_train_loss}, {running_validation_loss}")

    if use_wandb:
        results = {}
        for i in range(train_dataset.axis):
            results[f"conf_mat_{i}"] = wandb.plot.confusion_matrix(probs=None,
                                                                   y_true=validation_labels[:, i].numpy(),
                                                                   preds=classes_predicted[:, i].numpy(),
                                                                   class_names=train_dataset.deformation_classes,
                                                                   title=f'Axis {i}')
            results[f'Accuracy_{i}'] = validation_accuracy[i]
            for j in range(len(train_dataset.deformation_classes)):
                results[f'Precision_{i}_{j}'] = validation_precision[i][j]
                results[f'Recall_{i}_{j}'] = validation_recalls[i][j]
        results['Train loss'] = running_train_loss
        results['Valid loss'] = running_validation_loss

        wandb.log(results)












