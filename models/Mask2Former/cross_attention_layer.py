from typing import Optional
from torch import nn, Tensor
from torch.nn import functional as F




class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()
    
    
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward_post(self, tgt, memory, memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, 
                     pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), 
                                   key=self.with_pos_embed(memory, pos), 
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        
        
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,  memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        
        tgt2 = self.norm(tgt) 
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory, memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        
        
        
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")