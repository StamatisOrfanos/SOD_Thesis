import logging
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

# Simple Feed Forward Layer, part of t
class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def with_positional_embedding(self, tensor, position: Optional[Tensor]):
        return tensor if position is None else tensor + position


    def forward(self, target_tensor):
        target_tensor_2 = self.linear2(self.dropout(self.activation(self.linear1(target_tensor))))
        target_tensor = target_tensor + self.dropout(target_tensor_2)
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