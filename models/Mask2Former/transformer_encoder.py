# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from torch import nn
from torch.nn import functional as F
from models.Mask2Former.attention_layers import MaskedAttentionLayer, SelfAttentionLayer
from models.Mask2Former.ffn_layer import FFNLayer, MLP
from position_embedding_sine import PositionEmbeddingSine

import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0, activation="relu"):
        super().__init__()
        self.self_attn          = SelfAttentionLayer(d_model, nhead, dropout=dropout, activation=activation)        
        self.cross_attn         = MaskedAttentionLayer(d_model, nhead, dropout=dropout, activation=activation)
        self.ffn                = FFNLayer(d_model, dim_feedforward, dropout=dropout, activation=activation)
        self.position_embedding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)



    def forward(self, src, mask, query_embed, pos_embed):
        # Add position embedding to the source features
        src_with_pos = self.position_embedding(src, mask)
        
        # Apply self-attention and then add position embedding to the query features
        src = self.self_attn(src_with_pos, target_tensor_mask=mask)
        query_with_pos = self.position_embedding(query_embed, mask)
        
        # Apply cross-attention with query embedding and apply FFN
        src = self.cross_attn(src_with_pos, src, memory_mask=mask, pos=pos_embed, query_pos=query_with_pos)
        src = self.ffn(src)
        
        return src

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed
