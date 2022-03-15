"""
Copied from RCV1 and modified to work with 20NG and the new semsup API
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
from transformers import BertPreTrainedModel, BertModel
from .core import BaseModel, BaseModelArgs


class BertForSemanticEmbedding(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        # Create a placeholder for the label_model
        self.label_model = BertModel(config)

        # Projection layer for label embedding model
        if not self.config.share_label_model:
            self.label_projection = nn.Linear(config.hidden_size, config.label_model_hidden_size)      

        # Initialize weights and apply final processing
        # self.post_init()    

    def forward(self, input_loader=None, label_loader=None):
        """
        :input_loader contains the inputs to be fed to self.roberta
        :label_loader contains the inputs to be fed to the label model self.label_model
        """

        # STEP 1: Store the labels
        # During training, some classes might be held-out
        # Mask those classes so that they are not treated as negatives
        targets = input_loader.pop('labels')
        # represented_labels = label_loader.pop('represented_labels')[0]
        # labels = labels[:, represented_labels]

        # STEP 2: Forward pass through the input model
        outputs = self.bert(**input_loader)
        # outputs[1] for the BERT model and outputs[0] for the RoBERTa model
        sequence_output = outputs[1]
        input_cls_repr = sequence_output

        # STEP 3: Forward pass through the label model
        # The label loader adds an extra dimension at the beginning of the tensors
        # Remove them
        for key in label_loader.keys():
            label_loader[key] = torch.squeeze(label_loader[key], 0)
        with torch.no_grad():
            label_representations = self.label_model(**label_loader)[1] # (n_class, d_model)
            if self.config.share_label_model:
                label_representations = self.bert(**label_loader)[1] # (n_class, d_model)
            else:
                label_representations = self.label_model(**label_loader)[1] # (n_class, d_model)
                label_representations = label_representations @ self.label_projection.weight

         # Normalize the label representations if required
        if self.config.normalize_label_embeddings:
            label_representations = nn.functional.normalize(label_representations, dim=1)

        # Compute the logits
        logits = input_cls_repr @ label_representations.T  # (bs, n_class)

        return logits, targets


class BertForWord2Vec(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # NOTE: No label model

        # Projection layer for label embedding model
        if not self.config.share_label_model:
            # self.label_projection = nn.Linear(config.hidden_size, config.label_model_hidden_size)
            self.label_projection = nn.Linear(config.label_model_hidden_size, config.hidden_size)

        # Initialize weights and apply final processing
        # self.post_init()    

    def forward(self, input_loader=None, label_loader=None):
        """
        :input_loader contains the inputs to be fed to self.roberta
        :label_loader contains the inputs to be fed to the label model self.label_model
        """

        # STEP 1: Store the labels
        # During training, some classes might be held-out
        # Mask those classes so that they are not treated as negatives
        targets = input_loader.pop('labels')
        # represented_labels = label_loader.pop('represented_labels')[0]
        # labels = labels[:, represented_labels]

        # STEP 2: Forward pass through the input model
        outputs = self.bert(**input_loader)
        # outputs[1] for the BERT model and outputs[0] for the RoBERTa model
        sequence_output = outputs[1]
        input_cls_repr = sequence_output

        # If GILE model, compute the sigmoid
        if self.config.use_gile:
            input_cls_repr = torch.tanh(input_cls_repr)

        # STEP 3: Get the label model vectors
        # The label loader adds an extra dimension at the beginning of the tensors
        # Remove them
        for key in label_loader.keys():
            label_loader[key] = torch.squeeze(label_loader[key], 0)

        label_representations = label_loader['glove_emb'] @ self.label_projection.weight.T

        if self.config.use_gile:
            label_representations = torch.tanh(label_representations)

        # Normalize the label representations if required
        if self.config.normalize_label_embeddings:
            label_representations = nn.functional.normalize(label_representations, dim=1)

        # Compute the logits
        logits = input_cls_repr @ label_representations.T  # (bs, n_class)

        return logits, targets


@dataclass
class BertRCV1toNGArgs(BaseModelArgs):
    checkpoint: str = None

    # ignored since this is only on eval
    lr: float = 2e-5
    adam_epsilon: float = 1e-8


class BertRCV1toNG(BaseModel):
    def __init__(self, args: BertRCV1toNGArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)
        self.model = BertForSemanticEmbedding.from_pretrained(
            self.args.checkpoint
        )
        self.accuracy = torchmetrics.Accuracy()
        self.metrics = {"val_acc": self.accuracy}

    def forward(self, batch):
        input_batch = batch["input_loader"]
        label_batch = batch["label_loader"]
        
        logits, targets = self.model(input_batch, label_batch)
        loss = F.cross_entropy(input=logits, target=targets)
        return logits, targets, loss


class BertRCV1toNGDEVISE(BaseModel):
    def __init__(self, args: BertRCV1toNGArgs, *margs, **kwargs):
        super().__init__(args, *margs, **kwargs)
        self.model = BertForWord2Vec.from_pretrained(
            self.args.checkpoint
        )
        self.accuracy = torchmetrics.Accuracy()
        self.metrics = {"val_acc": self.accuracy}

    def forward(self, batch):
        input_batch = batch["input_loader"]
        label_batch = batch["label_loader"]
        
        logits, targets = self.model(input_batch, label_batch)
        loss = F.cross_entropy(input=logits, target=targets)
        return logits, targets, loss