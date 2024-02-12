# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_
from models.Mask2Former.ffn_layer import MLP
from position_embedding_sine import PositionEmbeddingSine
from models.Mask2Former.transformer_encoder import TransformerEncoderLayer



class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int, num_queries: int, nheads: int, dim_feedforward: int, dec_layers: int, mask_dim: int):
        """
        Parameters:
            in_channels (int): Number of channels in the input feature maps.
            num_classes (int): Number of classes for classification (including background).
            hidden_dim (int): Dimension of the feature vectors used in the Transformer.
            num_queries (int): Number of query slots used in the Transformer decoder.
            nheads (int): Number of attention heads in the multi-head attention layers.
            dim_feedforward (int): Dimensionality of the hidden layers in the feedforward network.
            dec_layers (int): Number of Transformer encoder layers to use in the decoder.
            mask_dim (int): Dimension of the mask feature vectors.
        """
        super().__init__()
        self.mask_classification = True

        # Positional Encoding
        self.positional_embedding_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        
        # Initialize Transformer encoder layers in the decoder
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                activation="relu"
            )
            for _ in range(dec_layers)
        ])

        # Learnable query features and learnable query p.e.
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
             
        # Learnable embeddings for the decoder queries (provide a fixed number of "slots" for the decoder to focus on different parts of the input)
        self.query_features = nn.Embedding(num_queries, hidden_dim)
        self.query_embeddings = nn.Embedding(num_queries, hidden_dim)

        # Define the amount of multi-scale features and create the corresponding level embedding (we use 4 scales)
        # and the projection layers to align the channel dimensions if necessary.
        self.num_feature_levels = 4
        self.level_embedding = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        self.reset_parameters(in_channels, hidden_dim)

        # Output layers 
        self.class_embedding = nn.Linear(hidden_dim, num_classes + 1)      # Number of classes plus the background
        self.mask_embedding  = MLP(hidden_dim, hidden_dim, mask_dim, 3)    # Mask dimensions output



    # Reset the parameters of the model using the xavier initialization
    def reset_parameters(self, in_channels, hidden_dim):
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim:
                self.input_proj.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())




    def forward(self, feature_map_list, mask_features_list, mask=None):
        """
        Parameters:
            feature_map_list (list): List of multi-scale feature maps from the backbone or previous layer (each element corresponds to a different scale).
            mask_features (list): Features to be used for mask prediction, not explicitly used in this snippet.
            mask: Optional argument, not used in this function but can be used for additional operations like applying masks to features.
        """
        # Assert that the number of feature maps matches the expected number of feature levels.
        assert len(feature_map_list) == self.num_feature_levels
        
        src = []       # List to store projected feature maps for each scale.
        pos = []       # List to store positional encodings for each scale.
        size_list = [] # List to store sizes (H, W) of feature maps for each scale.

        # Loop through each feature level.
        for i in range(self.num_feature_levels):
            size_list.append(feature_map_list[i].shape[-2:])   # Store the size of the feature map.
            
            # Generate positional encodings for the feature map and flatten it from NxCxHxW to HWxNxC for processing.
            pos.append(self.pe_layer(feature_map_list[i], None).flatten(2))
            
            # Project the feature map to the desired dimensionality (hidden_dim), add level embeddings, and flatten.
            src.append(self.input_proj[i](feature_map_list[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # Permute the flattened positional encodings and source feature maps for transformer processing.
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)


        _, bs, _ = src[0].shape # bs: batch size, inferred from the shape of the first source feature map.


        # Initialize query embeddings and replicate them for the batch size.
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1) # Initial output features for the queries.

        predictions_class = [] # List to store class predictions at each layer.
        predictions_mask = []  # List to store mask predictions at each layer.

        # Forward pass through prediction heads (not shown in this snippet) to generate initial predictions.
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features_list, attn_mask_target_size=size_list[0])
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
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features_list, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
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
        # this is a workaround to make torchscript happy, as torchscript doesn't support dictionary with non-homogeneous values, such as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]