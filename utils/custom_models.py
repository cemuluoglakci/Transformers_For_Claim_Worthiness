import torch
from torch import nn
from transformers import (AutoModel)


class TransformerClassifier(nn.Module):

    def __init__(self, wandb_config, class_number=1):

        super(TransformerClassifier, self).__init__()
        
        self.transformer = AutoModel.from_pretrained(wandb_config.model_name
            ,hidden_act  = wandb_config.hidden_act
            ,hidden_dropout_prob = wandb_config.transformer_dropout
            ,attention_probs_dropout_prob = wandb_config.attention_dropout
            ,layer_norm_eps = wandb_config.layer_norm_eps
            ,position_embedding_type = wandb_config.position_embedding_type
            )
        hidden_size = None
        try:
            hidden_size = self.transformer.config.to_dict()["hidden_size"]
        except:
            hidden_size = self.transformer.config.to_dict()["word_embed_proj_dim"]
            
        self.classifier = ClassificationHead(hidden_size, wandb_config.classifier_dropout, class_number)

    def forward(self, input_id, verbose = False):
        x = self.transformer(input_id)[0]
        x = self.classifier(x)
        return x

    # def predict(input_id)


class ClassificationHead(nn.Module):

    def __init__(self, hidden_size, classifier_dropout, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(classifier_dropout)
        self.fc = nn.Linear(hidden_size, num_labels, bias=True)

    def forward(self, last_hidden_states):
        x = last_hidden_states[:, 0, :]  # take <s> / [CLS] or equivalent token.
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x