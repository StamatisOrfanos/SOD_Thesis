import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        Initializes the Transformer Encoder.

        Parameters:
        d_model (int): The number of expected features in the input (required).
        nhead (int): The number of heads in the multiheadattention models (required).
        dim_feedforward (int): The dimension of the feedforward network model (default=2048).
        dropout (float): The dropout value (default=0.1).
        """
        super(TransformerEncoder, self).__init__()

        # Self-attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None, query_embed=None):
        """
        Passes the input, and the query through the transformer encoder.

        Parameters:
        src: The sequence to the encoder (required).
        mask: The mask for the src sequence (default=None).
        query_embed: The query embeddings for decoder (default=None).

        Returns:
        The encoded output.
        """
        # Self-attention with residual connection and normalization
        # query_embed is used as the query to the attention mechanism
        src2 = self.self_attn(query_embed, src, src, attn_mask=mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)

        # Feedforward network with residual connection and normalization
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)

        return src

# Example usage:
# d_model is the size of the embeddings (feature dimensions)
# nhead is the number of heads in the multi-head attention mechanisms
# transformer_encoder = TransformerEncoder(d_model=512, nhead=8)

# Given input feature maps `image_features` and optional `query_embed` for decoder queries,
# you would apply the encoder like this:
# encoded_features = transformer_encoder(image_features, mask=None, query_embed=query_embed)

# Note: The actual dimensions and architecture details need to be adjusted based on the specific model requirements.
