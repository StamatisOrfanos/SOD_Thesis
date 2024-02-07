# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from typing import Optional
from torch import nn, Tensor
from torch.nn import functional as F



class SelfAttentionLayer(nn.Module):
    """
    Parameters:
    - d_model (int): The number of expected features in the input (and output) tensor, the embedding dimension.
    - nhead (int): The number of heads in the multihead attention models.
    - dropout (float): The dropout value, used in the attention to prevent overfitting.
    - activation (str): The activation function for the feedforward layer.

    This layer is responsible for computing self-attention, where the query, key, and value are all the same tensor.
    It also applies dropout and layer normalization as part of the standard Transformer architecture.
    """
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu"):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.reset_parameters()
    
    
    # Initialize parameters using xavier_uniform method for better convergence
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    def with_positional_embedding(self, tensor, position: Optional[Tensor]):
        return tensor if position is None else tensor + position


    def forward(self, target_tensor, target_tensor_mask: Optional[Tensor] = None, target_tensor_key_padding_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        # Implement the positional embedding in the self-attention layer
        query = key = self.with_positional_embedding(target_tensor, query_pos)
        target_tensor2 = self.self_attention(query, key, value=target_tensor, attn_mask=target_tensor_mask, key_padding_mask=target_tensor_key_padding_mask)[0]        
        
        # Apply normalization after adding the residual in the self-attention layer
        target_tensor = target_tensor + self.dropout(target_tensor2) 
        target_tensor = self.norm(target_tensor) 
        
        return target_tensor
    
     
class MaskedAttentionLayer(nn.Module):
    """
    Parameters:
    - d_model (int): The number of expected features in the input tensor, typically the embedding dimension.
    - nhead (int): The number of heads in the multihead attention model.
    - dropout (float): The dropout value, used in the attention to prevent overfitting.
    - activation (str): The activation function for the feedforward layer.

    This layer is responsible for computing cross-attention, where the query comes from one tensor (target_tensor) and the key/value come from another tensor (memory).
    It applies dropout and layer normalization as part of the standard Transformer architecture.
    """
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu"):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.reset_parameters()
    
    
    # Initialize parameters using xavier_uniform method for better convergence
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def with_positional_embedding(self, tensor, position: Optional[Tensor]):
        return tensor if position is None else tensor + position


    def forward(self, target_tensor, memory, memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        # Apply positional embeddings to the target and memory tensors
        target_tensor_query = self.with_pos_embed(target_tensor, query_pos)
        memory_key = self.with_pos_embed(memory, pos)

        # Perform cross-attention between target and memory tensors
        target_tensor2 = self.multihead_attention(query=target_tensor_query, key=memory_key, value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]

        # Apply residual connection followed by dropout and normalization
        target_tensor = target_tensor + self.dropout(target_tensor2)
        target_tensor = self.norm(target_tensor)
        
        return target_tensor
   

   
   
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")