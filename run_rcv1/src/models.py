"""
Models for classification using semantic label embeddings.
"""

import torch
from torch import nn
import sys
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from transformers import AutoModel

# Import configs
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.bert.configuration_bert import BertConfig

# Loss functions
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, MultiLabelMarginLoss


# Auto model selection
MODEL_FOR_SEMANTIC_EMBEDDING = {
    "roberta": "RobertaForSemanticEmbedding",
    "bert": "BertForSemanticEmbedding",
}

MODEL_FOR_WORD2VEC = {
    "bert": "BertForWord2Vec",
}

MODEL_TO_CONFIG = {
    "roberta": RobertaConfig,
    "bert": BertConfig,
}


class AutoModelForSemanticEmbedding:
    """
    Class for choosing the right model class automatically.
    Loosely based on AutoModel classes in HuggingFace.
    """
    
    @staticmethod
    def from_pretrained(*args, **kwargs):
        # Check what type of model it is
        for key in MODEL_TO_CONFIG.keys():
            if type(kwargs['config']) == MODEL_TO_CONFIG[key]:
                class_name = getattr(sys.modules[__name__], MODEL_FOR_SEMANTIC_EMBEDDING[key])
                return class_name.from_pretrained(*args, **kwargs)
        
        # If none of the models were chosen
        raise("This model type is not supported. Please choose one of {}".format(MODEL_FOR_SEMANTIC_EMBEDDING.keys()))


class AutoModelForWord2vec:
    """
    Class for choosing the right model class automatically.
    Loosely based on AutoModel classes in HuggingFace.
    """
    
    @staticmethod
    def from_pretrained(*args, **kwargs):
        # Check what type of model it is
        for key in MODEL_TO_CONFIG.keys():
            if type(kwargs['config']) == MODEL_TO_CONFIG[key]:
                class_name = getattr(sys.modules[__name__], MODEL_FOR_WORD2VEC[key])
                return class_name.from_pretrained(*args, **kwargs)
        
        # If none of the models were chosen
        raise("This model type is not supported. Please choose one of {}".format(MODEL_FOR_WORD2VEC.keys()))


def margin_loss_formatting(labels, device=None):
    """
    Format the labels such that it can be used by MultiLabelMarginLoss
    Convert multilabel one-hot labels to a list of positive classes
    """

    # Create a tensor with element values as column number
    # E.g., [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    arange_tensor = torch.arange(labels.shape[1], device=device).repeat(labels.shape[0], 1)

    # Create a tensor with all -1's
    # E.g., [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    ones_tensor = -1 * torch.ones(labels.shape, device=device, dtype=torch.long)

    formatted_labels = torch.where(labels.bool(), arange_tensor, ones_tensor)

    # Sort each row of formatted labels such that all the -1s appear in the end
    formatted_labels, _ = torch.sort(formatted_labels, dim=1, descending=True)

    return formatted_labels


def semantic_model_forward_pass(
                        config=None,
                        input_loader=None,
                        input_cls_repr=None,
                        outputs=None,
                        labels=None,
                        represented_labels=None,
                        label_representations=None,
                        num_labels=None,
                        device=None):
    """
    Common forward pass for RoBERTa and BERT models
    """

    # Normalize the label representations if required
    if config.normalize_label_embeddings:
        label_representations = nn.functional.normalize(label_representations, dim=1)

    # Compute the logits
    logits = input_cls_repr @ label_representations.T  # (bs, n_class)

    # Compute the loss
    # Code copied from RobertaForSequenceClassification
    loss = None
    if labels is not None:
        if config.problem_type is None:
            if num_labels == 1:
                config.problem_type = "regression"
            elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                config.problem_type = "single_label_classification"
            else:
                config.problem_type = "multi_label_classification"

        if config.problem_type == "regression":
            loss_fct = MSELoss()
            if num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        elif config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        elif config.problem_type == "multi_label_classification":
            # Check if it is hinge loss or not
            if config.label_loss_type == 'hinge':
                # MultiLabelMarginLoss needs the labels to be formatted in a specific way
                formatted_labels = margin_loss_formatting(labels, device=device)

                # MultiLabelMarginLoss doesn't have a margin argument. HACK to allow margin usage
                loss_fct = MultiLabelMarginLoss()

                # Convert logits to logsigmoid
                logsigmoid_logits = nn.functional.logsigmoid(logits)

                negative_logsigmoid_logits = nn.functional.logsigmoid(-logits)

                # HACK: Use hack to include a margin parameter in the loss function
                # NOTE: MultiLabelMarginLoss doesn't have a margin parameter
                loss = config.hinge_margin_value * loss_fct(1./config.hinge_margin_value * logsigmoid_logits, formatted_labels)
            elif config.label_loss_type == 'focal':
                pos_weights = config.relative_weight_positive_samples * torch.ones([represented_labels.shape[0]], device=device)
                loss_fct = BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')

                # Compute log(p) or log(1-p) depending on the label
                log_pt = -loss_fct(logits, labels)
                pt = torch.exp(log_pt)

                loss = - ((1 - pt) ** config.focal_loss_gamma) * log_pt

                # Take the mean
                loss = torch.mean(loss)
            elif config.label_loss_type == 'bce':
                # Use standard BCE loss
                pos_weights = config.relative_weight_positive_samples * torch.ones([represented_labels.shape[0]], device=device)
                loss_fct = BCEWithLogitsLoss(pos_weight=pos_weights)
                loss = loss_fct(logits, labels)
            else:
                raise('Choose one of the available loss functions.')

    if 'return_dict' in input_loader.keys() and not input_loader['return_dict']:
        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

    return SequenceClassifierOutput(
        loss=loss,
        logits=logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


class RobertaForSemanticEmbedding(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        # self.label_model is initialized by the training script

        # Projection layer for label embedding model
        if not self.config.share_label_model:
            self.label_projection = nn.Linear(config.hidden_size, config.label_model_hidden_size)

        # Initialize weights and apply final processing
        self.post_init()        

    def forward(self, input_loader=None, label_loader=None):
        """
        :input_loader contains the inputs to be fed to self.roberta
        :label_loader contains the inputs to be fed to 
        """

        # STEP 1: Store the labels
        # During training, some classes might be held-out
        # Mask those classes so that they are not treated as negatives
        labels = input_loader.pop('labels')
        represented_labels = label_loader.pop('represented_labels')[0]
        labels = labels[:, represented_labels]

        # STEP 2: Forward pass through the input model
        outputs = self.roberta(**input_loader)
        sequence_output = outputs[0]
        input_cls_repr = sequence_output[:, 0, :]

        # STEP 3: Forward pass through the label model
        # The label loader adds an extra dimension at the beginning of the tensors
        # Remove them
        for key in label_loader.keys():
            label_loader[key] = torch.squeeze(label_loader[key], 0)
        with torch.set_grad_enabled(not self.config.freeze_label_model):
            if self.config.share_label_model:
                label_representations = self.roberta(**label_loader).pooler_output # (n_class, d_model)
            else:
                label_representations = self.label_model(**label_loader).pooler_output # (n_class, d_model)
                label_representations = label_representations @ self.label_projection.weight

        # Call function for forward pass
        return semantic_model_forward_pass(
            config=self.config,
            input_loader=input_loader,
            input_cls_repr=input_cls_repr,
            outputs=outputs,
            labels=labels,
            represented_labels=represented_labels,
            label_representations=label_representations,
            num_labels=None,
            device=self.device
        )


class BertForSemanticEmbedding(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # Create a placeholder for the label_model
        self.label_model = BertModel(config)
        # self.label_model = AutoModel.from_pretrained(
        #     config.label_model_name_or_path,
        #     cache_dir=config.cache_dir
        # )

        # Projection layer for label embedding model
        if not self.config.share_label_model:
            self.label_projection = nn.Linear(config.hidden_size, config.label_model_hidden_size)      

        # Initialize weights and apply final processing
        self.post_init()    

    def forward(self, input_loader=None, label_loader=None):
        """
        :input_loader contains the inputs to be fed to self.roberta
        :label_loader contains the inputs to be fed to the label model self.label_model
        """

        # STEP 1: Store the labels
        # During training, some classes might be held-out
        # Mask those classes so that they are not treated as negatives
        labels = input_loader.pop('labels')
        represented_labels = label_loader.pop('represented_labels')[0]
        labels = labels[:, represented_labels]

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
        with torch.set_grad_enabled(not self.config.freeze_label_model):
            label_representations = self.label_model(**label_loader)[1] # (n_class, d_model)
            if self.config.share_label_model:
                label_representations = self.bert(**label_loader)[1] # (n_class, d_model)
            else:
                label_representations = self.label_model(**label_loader)[1] # (n_class, d_model)
                label_representations = label_representations @ self.label_projection.weight


        # Call function for forward pass
        return semantic_model_forward_pass(
            config=self.config,
            input_loader=input_loader,
            input_cls_repr=input_cls_repr,
            outputs=outputs,
            labels=labels,
            represented_labels=represented_labels,
            label_representations=label_representations,
            num_labels=None,
            device=self.device
        )


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
        self.post_init()    

    def forward(self, input_loader=None, label_loader=None):
        """
        :input_loader contains the inputs to be fed to self.roberta
        :label_loader contains the inputs to be fed to the label model self.label_model
        """

        # STEP 1: Store the labels
        # During training, some classes might be held-out
        # Mask those classes so that they are not treated as negatives
        labels = input_loader.pop('labels')
        represented_labels = label_loader.pop('represented_labels')[0]
        labels = labels[:, represented_labels]

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

        label_representations = label_loader['embeddings'] @ self.label_projection.weight.T

        if self.config.use_gile:
            label_representations = torch.tanh(label_representations)

        # Call function for forward pass
        return semantic_model_forward_pass(
            config=self.config,
            input_loader=input_loader,
            input_cls_repr=input_cls_repr,
            outputs=outputs,
            labels=labels,
            represented_labels=represented_labels,
            label_representations=label_representations,
            num_labels=None,
            device=self.device
        )