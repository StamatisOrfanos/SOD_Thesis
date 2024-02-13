# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
from typing import List, Dict, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from models.Mask2Former.ffn_layer import MLP
from models.Mask2Former.position_embedding_sine import PositionEmbeddingSine
from models.Mask2Former.transformer_encoder import TransformerEncoderLayer



class Mask2Former(nn.Module):
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
        self.num_feature_levels = 5
        self.scale_level_embedding = nn.Embedding(self.num_feature_levels, hidden_dim)
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
            mask_features_list (list): Features to be used for mask prediction, not explicitly used in this snippet.
            mask: Optional argument, not used in this function but can be used for additional operations like applying masks to features.
        """
        # Assert that the number of feature maps matches the expected number of feature levels.
        assert len(feature_map_list) == self.num_feature_levels
        
        srcs = []
        mask_transformed = []
        pos_embeds = []
        
        
        # Project and positionally encode each level of features
        for level, (feature, mask) in enumerate(zip(feature_map_list, mask_features_list)):
            src = self.input_proj[level](feature)
            mask_transformed.append(mask) 
            pos_embed = self.positional_embedding_layer(src, mask)
            pos_embeds.append(pos_embed)
            srcs.append(src)
            
        # Prepare query embeddings
        query_embed = self.query_embeddings.weight
        query_features = self.query_features.weight.unsqueeze(1).repeat(1, srcs[0].size(1), 1)
        
        # Pass through the Transformer encoder layers
        for layer in self.transformer_encoder_layers:
            query_features = layer(query_features, mask_transformed, query_embed, pos_embeds)

        # Normalize the output features from the Transformer
        query_features = self.decoder_norm(query_features)

        # Pass to prediction heads
        logits, mask_embeddings = self.forward_prediction_heads(query_features)

        return logits, mask_embeddings


    def forward_prediction_heads(self, query_features):
        """
        Parameters:
            query_features (Tensor): The output features from the Transformer encoder.            
        """
        # Classification for each query
        logits = self.class_embedding(query_features)

        # Mask embeddings for each query
        mask_embeddings = self.mask_embedding(query_features)
        
        return logits, mask_embeddings
       

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class: List[torch.Tensor], outputs_seg_masks: List[torch.Tensor]) -> List[Dict[str, Union[torch.Tensor, List[torch.Tensor]]]]:
        """
        This method is designed to work around limitations with TorchScript and dictionaries containing non-homogeneous values by creating a list of 
        dictionaries, each containing either the predicted logits and masks for classification tasks or just the masks for segmentation tasks, depending 
        on whether mask classification is enabled.

        Parameters:
            outputs_class (List[torch.Tensor]): A list of tensors representing the class predictions at each decoder layer.
            outputs_seg_masks (List[torch.Tensor]): A list of tensors representing the predicted segmentation masks at each decoder layer.

        """
        # Validate input lengths
        assert len(outputs_class) == len(outputs_seg_masks), "Outputs for class and segmentation masks must have the same length."

        if self.mask_classification:
            # Include both logits and masks for auxiliary losses if mask classification is enabled
            aux_losses = [{"pred_logits": logits, "pred_masks": masks}
                          for logits, masks in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        else:
            # Include only masks for auxiliary losses if mask classification is not enabled
            aux_losses = [{"pred_masks": masks} for masks in outputs_seg_masks[:-1]]

        return aux_losses