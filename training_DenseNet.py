import os
import torch
import datetime
import numpy as np
import torch.nn.functional as F
import torch.optim as optim       # for the optimization algorithms
import torch.nn as nn             # neural network modules
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import DenseNet121_Weights
from torch.utils.data import DataLoader         # easier dataset management, creates mini batches
from classificazioneDataset import DeformationDataset
from sklearn.metrics import (recall_score, precision_score, accuracy_score, mean_squared_error,
                             mean_absolute_error, r2_score)
from enum import Enum
from typing import Union


class ModelType(Enum):
    regression = "regression"
    classification = "general"


class SiameseMultiHeadNetwork(nn.Module):
    def __init__(self,
                 features_extractor: nn.Module,
                 model_type: ModelType,
                 output_layers: Union[nn.ModuleList, nn.Sequential]
                 ):
        super(SiameseMultiHeadNetwork, self).__init__()
        self.features_extractor = features_extractor
        self.output_layers = output_layers
        self.model_type = model_type
        if self.model_type == ModelType.classification:
            self.forward = self.forward_classification
        elif self.model_type == ModelType.regression:
            self.forward = self.forward_regression

    def forward_one(self, x):
        x = self.features_extractor(x)
        return x

    def forward_classification(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        distance = output1 - output2
        return torch.stack([classifier(distance) for classifier in self.output_layers], dim=1)

    def forward_regression(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        distance = output1 - output2
        return self.output_layers(distance)

    @torch.no_grad()
    def predict(self, input1, input2):
        return F.softmax(self.forward(input1, input2), dim=-1)


class SiameseDenseNet(SiameseMultiHeadNetwork):
    def __init__(self, model_type: ModelType):
        features_extractor = models.densenet121()
        features_extractor.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if model_type == ModelType.classification:
            classification_layers = nn.ModuleList([nn.Sequential(nn.Dropout(p=0.25),
                                                       nn.Linear(features_extractor.classifier.in_features, 256),
                                                       nn.ReLU(),
                                                       nn.Dropout(p=0.25),
                                                       nn.Linear(256, 128),
                                                       nn.ReLU(),
                                                       nn.Linear(128, 5))
                                         for _ in range(5)])
            features_extractor.classifier = nn.Identity()

            super(SiameseDenseNet, self).__init__(features_extractor=features_extractor,
                                                  output_layers=classification_layers,
                                                  model_type=model_type)

        if model_type == ModelType.regression:
            regression_layers = nn.Sequential(nn.Linear(features_extractor.classifier.in_features, 256),
                                              nn.ReLU(),
                                              nn.Linear(256, 128),
                                              nn.ReLU(),
                                              nn.Linear(128, 5))
            features_extractor.classifier = nn.Identity()

            super(SiameseDenseNet, self).__init__(features_extractor=features_extractor,
                                                  output_layers=regression_layers,
                                                  model_type=model_type)


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
            labels = labels.to(device)      # true value
            # Forward pass
            outputs = model(img1, img2)     # prediction

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
            validation_precision = [precision_score(validation_labels[:, head], classes_predicted[:, head], average=None) for head
                              in range(outputs.shape[1])]
            validation_recalls = [recall_score(validation_labels[:, head], classes_predicted[:, head], average=None) for head in
                            range(outputs.shape[1])]
            return running_validation_loss, validation_accuracy, validation_precision, validation_recalls, validation_labels, classes_predicted
        else:
            validation_predictions = torch.cat(validation_predictions, dim=0)
            validation_MSE = mean_squared_error(validation_labels.cpu(), validation_predictions.cpu())
            validation_MAE = mean_absolute_error(validation_labels.cpu(), validation_predictions.cpu())
            validation_R2 = r2_score(validation_labels.cpu(), validation_predictions.cpu())
            return running_validation_loss, validation_MSE, validation_MAE, validation_R2, validation_labels, validation_predictions


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# classification = True
classification = False

# Hyperparameters
batch_size = 16
max_epochs = 10
max_epochs_no_improvement = 6
learning_rate = 1e-3
max_batch_size = 16
validation_batch_size = 64
accumulation_steps = batch_size // max_batch_size if batch_size > max_batch_size else 1

if classification:
    model = SiameseDenseNet(model_type=ModelType.classification).to(device)
else:
    model = SiameseDenseNet(model_type=ModelType.regression).to(device)

train_dataset = DeformationDataset(
                                   datasets_path='datasets',
                                   # datasets_path="/media/brahe/dataset_Pablo/testing",
                                   training=True,
                                   classification=classification,
                                   max_translation_x=0,
                                   max_translation_y=0,
                                   max_rotation=np.deg2rad(0),
                                   flip_left_right=True,
                                   gradient_thresholding=True)

validation_dataset = DeformationDataset(
                                  datasets_path='datasets',
                                  # datasets_path="/media/brahe/dataset_Pablo/validation",
                                  training=False,
                                  classification=classification)

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
if classification:
    criterion = nn.CrossEntropyLoss()
else:
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
        labels = labels.to(device)      # etichetta 'corretta'
        # Forward pass
        outputs = model(img1, img2)     # predizioni del modello
        if classification:
            loss = sum(criterion(outputs[:, i], labels[:, i]) for i in range(5)) / accumulation_steps
        else:
            loss = criterion(outputs, labels) / accumulation_steps
        loss.backward()
        if ((batch_number + 1) * max_batch_size) % batch_size == 0 or batch_size < max_batch_size:
            optimizer.step()
            optimizer.zero_grad()

        running_train_loss += loss.item()

    train_accuracy /= len(train_dataset)
    running_train_loss /= len(train_loader)
    running_train_loss *= accumulation_steps

    if classification:
        running_validation_loss, validation_accuracy, validation_precision, validation_recalls, validation_labels, classes_predicted = (
            validation_epoch(model, validation_loader, criterion, classification))
    else:
        running_validation_loss, validation_MSE, validation_MAE, validation_R2, validation_labels, validation_predictions = (
            validation_epoch(model, validation_loader, criterion, classification))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"{current_time}: {running_train_loss}, {running_validation_loss}")
