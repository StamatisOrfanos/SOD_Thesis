import torch
from torch import nn

class AttentionQueryEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Parameters:
            input_dim (int): The combined dimensionality of the query embeddings and bounding box features.
            output_dim (int): The desired output dimensionality, which is typically the original size of the query embeddings.
        """
        super().__init__()
        self.integration_layer = nn.Linear(input_dim, output_dim)

    def forward(self, query_embeddings, bounding_box_features):
        """
        Parameters:
            query_embeddings (Tensor): The original query embeddings.
            bounding_box_features (Tensor): The bounding box features to be integrated with the query embeddings.
        """
        combined_features = torch.cat([query_embeddings, bounding_box_features], dim=0)
        return self.integration_layer(combined_features)
