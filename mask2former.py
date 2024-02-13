# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .position_encoding import PositionEmbeddingSine

class SelfAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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

    def forward_post(self, tgt, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, tgt_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask, tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask, tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
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

    def forward_post(self, tgt, memory, memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory, memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos), key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt, memory, memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask, memory_key_padding_mask, pos, query_pos)


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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(self, in_channels, num_classes, mask_classification=True, hidden_dim=256, num_queries=100, nheads=8, 
                 dim_feedforward=2048, dec_layers=10, pre_norm=False, mask_dim=256,enforce_input_project=False):
        super().__init__()

        # positional encoding
        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm, )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=0.0, normalize_before=pre_norm, )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=0.0, normalize_before=pre_norm, )
            )

        # learnable query features learnable query p.e.
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification: self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)



    def forward(self, x, mask_features, mask=None):
        # x: List of multi-scale feature maps from the backbone or previous layer (each element corresponds to a different scale).
        # mask_features: Features to be used for mask prediction, not explicitly used in this snippet.
        # mask: Optional argument, not used in this function but can be used for additional operations like applying masks to features.

        # Assert that the number of feature maps matches the expected number of feature levels.
        assert len(x) == self.num_feature_levels
        
        src = [] # List to store projected feature maps for each scale.
        pos = [] # List to store positional encodings for each scale.
        size_list = [] # List to store sizes (H, W) of feature maps for each scale.

        # Loop through each feature level.
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:]) # Store the size of the feature map.
            # Generate positional encodings for the feature map and flatten it from NxCxHxW to HWxNxC for processing.
            pos.append(self.pe_layer(x[i], None).flatten(2))
            # Project the feature map to the desired dimensionality (hidden_dim), add level embeddings, and flatten.
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # Permute the flattened positional encodings and source feature maps for transformer processing.
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape # bs: batch size, inferred from the shape of the first source feature map.

        # Initialize query embeddings and replicate them for the batch size.
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1) # Initial output features for the queries.

        predictions_class = [] # List to store class predictions at each layer.
        predictions_mask = [] # List to store mask predictions at each layer.

        # Forward pass through prediction heads (not shown in this snippet) to generate initial predictions.
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # Iterate through each transformer layer.
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels # Determine the current feature level index.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False # Update attention mask.
            
            # Apply cross-attention using the current level's source feature maps and positional encodings.
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # No masking on padded regions here.
                pos=pos[level_index], query_pos=query_embed
            )

            # Apply self-attention.
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # Apply feedforward network layer.
            output = self.transformer_ffn_layers[i](
                output
            )

            # Generate predictions for this layer.
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        # Collect the final predictions.
        out = {
            'pred_logits': predictions_class[-1], # Class predictions.
            'pred_masks': predictions_mask[-1], # Mask predictions.
            # Auxiliary outputs for intermediate layers to support training stability and performance.
            'aux_outputs': self._set_aux_loss(predictions_class if self.mask_classification else None, predictions_mask)
        }
        return out


    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # Normalize the output features from the decoder using Layer Normalization.
        decoder_output = self.decoder_norm(output)
        
        # Transpose the decoder output to match the expected input shape for the subsequent operations.
        # This changes the shape from [sequence length, batch size, features] to [batch size, sequence length, features].
        decoder_output = decoder_output.transpose(0, 1)
        
        # Pass the transposed decoder output through a linear layer to predict class logits.
        outputs_class = self.class_embedding(decoder_output)
        
        # Similarly, pass the transposed decoder output through another linear layer to get mask embeddings.
        mask_embed = self.mask_embedding(decoder_output)
        
        # Perform a tensor operation to generate mask predictions. This operation projects the mask embeddings onto the mask features.
        # "bqc,bchw->bqhw" is the einsum operation indicating batch (b), queries (q), channels (c), height (h), and width (w).
        # It effectively combines mask embeddings (bqc) with mask features (bchw) to produce mask predictions (bqhw).
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # Interpolate the output masks to match the target size for attention masks. This is used for higher-resolution prediction.
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        
        # Apply sigmoid to the interpolated attention mask, flatten it, repeat it for each attention head, and then flatten the first two dimensions.
        # The threshold (< 0.5) determines which positions are allowed to attend: values below 0.5 after sigmoid are set to `True` (meaning they cannot attend).
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        
        # Detach the attention mask from the computation graph to prevent gradients from flowing into it.
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]