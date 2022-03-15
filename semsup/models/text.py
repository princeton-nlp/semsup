"""
Text classification baselines
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
from .core import BaseModel, BaseModelArgs, SemSupModel, SemSupModelArgs, get_text_model


@dataclass
class BertSemSupArgs(SemSupModelArgs):
    model: str = None
    pretrained_model: bool = False
    lr: float = 2e-5
    adam_epsilon: float = 1e-8


@dataclass
class BertBaselineArgs(BaseModelArgs):
    model: str = None
    pretrained_model: bool = False
    lr: float = 2e-5
    adam_epsilon: float = 1e-8
    num_labels: int = None 


@dataclass
class DEVISEBaselineArgs(SemSupModelArgs):
    model: str = None
    pretrained_model: bool = False
    use_gile: bool = False
    label_model: str = None # ignored
    lr: float = 2e-5
    adam_epsilon: float = 1e-8   


class BertSemSup(SemSupModel):
    def __init__(self, args: BertSemSupArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)
        self.model = get_text_model(
            model = self.args.model,
            pretrained = self.args.pretrained_model,
        )
        self.projection = nn.Linear(512, 512, bias=False)
        self.accuracy = torchmetrics.Accuracy()
        self.metrics = {"val_acc": self.accuracy}

    def forward(self, batch, label_rep):
        targets = batch.pop("labels")
        input_rep = self.model(**batch).pooler_output  # (bs, d_model)
        input_rep = self.projection(input_rep)
        logits = input_rep @ label_rep  # (bs, n_class)
        loss = F.cross_entropy(input=logits, target=targets)
        return logits, targets, loss


class BertBaseline(BaseModel):
    def __init__(self, args: BertBaselineArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)
        self.label_model = None
        self.model = get_text_model(
            model = self.args.model,
            pretrained = self.args.pretrained_model,
            classifier = True,
            num_labels = args.num_labels
        )
        
        self.accuracy = torchmetrics.Accuracy()
        self.metrics = {"val_acc": self.accuracy}

    def forward(self, batch):
        batch = batch["input_loader"]
        targets = batch.pop("labels")
        logits = self.model(**batch).logits
        loss = F.cross_entropy(input=logits, target=targets)
        return logits, targets, loss


class DEVISEBaseline(BaseModel):
    def __init__(self, args: DEVISEBaselineArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)
        self.label_model = None
        self.model = get_text_model(
            model = self.args.model,
            pretrained = self.args.pretrained_model,
        )
        self.projection = nn.Linear(512, 300, bias=False)
        self.accuracy = torchmetrics.Accuracy()
        self.metrics = {"val_acc": self.accuracy}

    def forward(self, batch):
        label_rep = batch["label_loader"]["glove_emb"].squeeze().t() #(300, n_class)

        batch = batch["input_loader"]
        targets = batch.pop("labels")
        input_rep = self.model(**batch).pooler_output  # (bs, d_model)
        input_rep = self.projection(input_rep) # (bs, 300)
        
        if self.args.use_gile:
            label_rep = torch.tanh(label_rep)
            input_rep = torch.tanh(input_rep)

        logits = input_rep @ label_rep  # (bs, n_class)
        loss = F.cross_entropy(input=logits, target=targets)
        return logits, targets, loss
