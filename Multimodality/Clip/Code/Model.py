from torch import nn
import torch.nn.functional as F

from Loss import cross_entropy
from Utils import read_settings
from ProjectionHead import ProjectionHead
from Encoder import ImageEncoder, TextEncoder

config = read_settings()

class CLIPModel(nn.Module):
    def __init__(
        self, temperature=config["temperature"],
        image_embedding=config["image_embedding"], text_embedding=config["text_embedding"]):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature

        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T

        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )

        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')

        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
