from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchmetrics

from .core import BaseModel, BaseModelArgs, SemSupModel, SemSupModelArgs


@dataclass
class ResNetSemSupArgs(SemSupModelArgs):
    predict_strategy: str = None
    pretrained_model: bool = False
    lr: float = 2e-4
    adam_epsilon: float = 1e-8
    image_type: str = "cifar"


@dataclass
class ResNetBaselineArgs(BaseModelArgs):
    num_classes: int = None
    label_model: str = None # ignored
    pretrained_model: bool = False
    lr: float = 2e-4
    adam_epsilon: float = 1e-8
    image_type: str = "cifar"


@dataclass
class DEVISEVisBaselineArgs(SemSupModelArgs):
    use_gile: bool = False
    pretrained_model: bool = False
    label_model: str = None # ignored
    image_type: str = "cifar"


class ResNetSemSup(SemSupModel):
    def __init__(self, args: ResNetSemSupArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)

        self.model = models.resnet18(pretrained=self.args.pretrained_model)
        # modify architecture to fit the smaller 32 x 32 cifar image dims
        if self.args.image_type.lower() == "cifar":
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            self.model.maxpool = nn.Identity()

        # modify the fc layer
        self.model.fc = nn.Linear(
            self.model.fc.in_features, self.label_model.config.hidden_size, bias=False
        )

        self.accuracy = torchmetrics.Accuracy()
        self.metrics = {"val_acc": self.accuracy}

    def forward(self, batch, label_rep):
        input_data, targets = batch
        input_rep = self.model(input_data)  # (bs, d_model)
        logits = input_rep @ label_rep  # (bs, n_class)
        loss = F.cross_entropy(input=logits, target=targets)
        return logits, targets, loss


class ResNetBaseline(BaseModel):
    def __init__(self, args: ResNetBaselineArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)

        self.label_model = None
        self.model = models.resnet18(pretrained=self.args.pretrained_model)
        # modify architecture to fit the smaller 32 x 32 cifar image dims
        if self.args.image_type.lower() == "cifar":
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            self.model.maxpool = nn.Identity()

        # modify the fc layer
        self.model.fc = nn.Linear(
            self.model.fc.in_features, self.args.num_classes
        )

        self.accuracy = torchmetrics.Accuracy()
        self.metrics = {"val_acc": self.accuracy}

    def forward(self, batch):
        batch = batch["input_loader"]
        input_data, targets = batch
        logits = self.model(input_data)  # (bs, d_model)
        loss = F.cross_entropy(input=logits, target=targets)
        return logits, targets, loss


class DEVISEVisBaseline(BaseModel):
    def __init__(self, args: DEVISEVisBaselineArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)
        self.label_model = None
        self.model = models.resnet18(pretrained=self.args.pretrained_model)
        # modify architecture to fit the smaller 32 x 32 cifar image dims
        if self.args.image_type.lower() == "cifar":
            self.model.conv1 = nn.Conv2d(
                3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            )
            self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, 300, bias=False)
        self.accuracy = torchmetrics.Accuracy()
        self.metrics = {"val_acc": self.accuracy}
        
    def forward(self, batch):
        label_rep = batch["label_loader"]["glove_emb"].squeeze().t() #(300, n_class)
        input_data, targets = batch["input_loader"]
        input_rep = self.model(input_data) #(bs, 300)
        
        if self.args.use_gile:
            label_rep = torch.tanh(label_rep)
            input_rep = torch.tanh(input_rep)

        logits = input_rep @ label_rep  # (bs, n_class)
        loss = F.cross_entropy(input=logits, target=targets)
        return logits, targets, loss
