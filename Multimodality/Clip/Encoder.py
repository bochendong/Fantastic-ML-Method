import timm
import torch
from torch import nn
from transformers import DistilBertModel

from Utils import read_settings

config = read_settings()

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name=config['model_name'], 
                 pretrained=config['pretrained'], trainable=config['trainable']):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    
class TextEncoder(nn.Module):
    def __init__(self, model_name=config["text_encoder_model"], 
                 pretrained=config['pretrained'], trainable=config['trainable']):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]