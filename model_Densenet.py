import torch
import torch.nn as nn             # neural network modules
from enum import Enum
from typing import Union
import torch.nn.functional as F
import torchvision.models as models


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