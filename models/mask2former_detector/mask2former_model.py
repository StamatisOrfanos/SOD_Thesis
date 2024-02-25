from typing import List, Dict, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from models.mask2former_detector.ffn_layer import MLP
from models.mask2former_detector.position_embedding_sine import PositionEmbeddingSine
from models.mask2former_detector.transformer_decoder_block import TransformerDecoderLayer


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
            TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=nheads,
                dim_feedforward=dim_feedforward,
                dropout=0.0,
                activation="relu"
            )
            for _ in range(dec_layers)
        ])

        # Learnable query features and learnable query p.e.
        self.num_queries = num_queries
             
        # Learnable embeddings for the decoder queries (provide a fixed number of "slots" for the decoder to focus on different parts of the input)
        self.query_features = nn.Embedding(num_queries, hidden_dim)
        self.query_embeddings = nn.Embedding(num_queries, hidden_dim)

        # Define the amount of multi-scale features and create the corresponding level embedding (we use 5 scales from the EFPN)
        # and the projection layers to align the channel dimensions if necessary.
        self.num_feature_levels = 5
        self.scale_level_embedding = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        
        self.reset_parameters(in_channels, hidden_dim)

        # Output layers [Number of classes plus the background, Mask dimensions output]
        self.class_embedding = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embedding  = MLP(hidden_dim, hidden_dim, mask_dim, 3)



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
            mask_features (list): Features to be used for mask prediction.
            mask: Optional argument, not used in this function but can be used for additional operations like applying masks to features.
        """
        assert len(feature_map_list) == self.num_feature_levels
        
        
        # Lists to store    src : [projected feature maps for each scale]
        # positional_embeddings : [positional encodings for each scale]
        # feature_maps_size_list: [sizes (H, W) of feature maps for each scale]
        src, positional_embeddings, feature_maps_size_list = self.generate_info_per_feature_map(feature_map_list)

        # Initialize query embeddings and replicate them for the batch size and create the initial output features for the queries.
        _, batch_size, _ = src[0].shape 
        query_embed      = self.query_embeddings.weight.unsqueeze(1).repeat(1, batch_size, 1)
        output           = self.query_features.weight.unsqueeze(1).repeat(1, batch_size, 1)         
        
        # List of predictions for each class and the corresponding mask of each small object for each feature map
        predictions_class, predictions_mask = self.class_mask_predictions(output, src, positional_embeddings, feature_maps_size_list, mask_features_list, query_embed)

        # Collect the final class and mask predictions, along with auxiliary outputs for intermediate layers to support training stability and performance.
        result = {
            'pred_logits': predictions_class[-1],
            'pred_masks' : predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(predictions_class if self.mask_classification else None, predictions_mask)
        }
        
        return result


    def generate_info_per_feature_map(self, feature_map_list):
        """
        Parameters:
            feature_map_list (list): List of multi-scale feature maps from the backbone or previous layer
        """
        src = []
        positional_embeddings = []
        feature_maps_size_list = []
        
        for i in range(self.num_feature_levels):                
            feature_maps_size_list.append(feature_map_list[i].shape[-2:]) 
            
            # Generate positional encodings, flatten it from NxCxHxW to HWxNxC for processing.
            positional_embeddings.append(self.positional_embedding_layer(feature_map_list[i], None).flatten(2)) 
            
            # Project feature map to the desired dimensionality (hidden_dim), add level embeddings, and flatten.
            src.append(self.input_proj[i](feature_map_list[i]).flatten(2) + self.scale_level_embedding.weight[i][None, :, None])
            
            # Permute the flattened positional encodings and source feature maps for transformer processing.
            positional_embeddings[-1] = positional_embeddings[-1].permute(2, 0, 1) 
            src[-1] = src[-1].permute(2, 0, 1)
            
        return src, positional_embeddings, feature_maps_size_list




    def class_mask_predictions(self, output, src, positional_embeddings, feature_maps_size_list, mask_features_list, query_embed):
        """
        Parameters:
            output (tensor): 
            src (list): Projected feature maps for each scale to dimensionality 
            positional_embeddings (list): [positional encodings for each scale]
            feature_maps_size_list (list): [sizes (H, W) of feature maps for each scale]
            mask_features_list (list): Features to be used for mask prediction.
            query_embed (tensor): 
        """
        # List to store class predictions at each layer. List to store mask predictions at each layer.
        predictions_class = [] 
        predictions_mask  = [] 

        # Forward pass through prediction heads to generate initial predictions.
        outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(output, mask_features_list, feature_maps_size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        
        for i in range(self.num_layers):
            
            # Determine the current feature level index and Update attention mask.
            level_index = i % self.num_feature_levels
            attention_mask[torch.where(attention_mask.sum(-1) == attention_mask.shape[-1])] = False
            
            # Apply the TransformerEncoder Layer that includes the forward functions for [Mask-Attention, Self-Attention, Feed-Forward]
            output = self.transformer_encoder_layers[i](output, src, level_index, attention_mask, positional_embeddings, query_embed)
            
            # Generate predictions for this layer.
            outputs_class, outputs_mask, attention_mask = self.forward_prediction_heads(output, mask_features_list, feature_maps_size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
    
        return predictions_class, predictions_mask



    def forward_prediction_heads(self, output, mask_features_list, attention_mask_target_size):
        """
        Parameters:
            output (tensor): 
            mask_features_list (list): Features to be used for mask prediction.
            attention_mask_target_size (list): Feature maps size list
        """
        # Transpose the decoder output to match the expected input shape for the subsequent operations after Normalizing.
        # This changes the shape from [sequence length, batch size, features] to [batch size, sequence length, features].        
        decoder_output = output
        decoder_output = decoder_output.transpose(0, 1)
        
        # Pass the transposed decoder output through a linear layer to predict class logits.
        # Similarly, pass the transposed decoder output through another linear layer to get mask embeddings.
        outputs_class  = self.class_embedding(decoder_output)
        mask_embedding = self.mask_embedding(decoder_output)
        
        
        # Perform a tensor operation to generate the mask predictions. Project the mask embeddings onto the mask features.
        # "bqc,bchw->bqhw" is the einsum operation indicating: batch (b), queries (q), channels (c), height (h) and width (w).
        # It effectively combines mask embeddings (bqc) with mask features (bchw) to produce mask predictions (bqhw).
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embedding, mask_features_list)

        # Interpolate the output masks to match the target size for attention masks. This is used for higher-resolution prediction.
        attention_mask = F.interpolate(outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False)
        
        # Apply sigmoid to the interpolated attention mask, flatten it, repeat it for each attention head, and then flatten the first two dimensions.
        # The threshold (< 0.5) determines which positions are allowed to attend: values below 0.5 after sigmoid are set to `True` (meaning they cannot attend).
        attention_mask = (attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        
        # Detach the attention mask from the computation graph to prevent gradients from flowing into it.
        attention_mask = attention_mask.detach()

        return outputs_class, outputs_mask, attention_mask




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
    
    
    def instance_segmentation(self, feature_map_list, mask_features_list):
        outputs = self.forward(feature_map_list, mask_features_list)

        # Calculate class confidence
        pred_logits = outputs['pred_logits']  # Shape: [batch_size, num_queries, num_classes + 1]
        class_confidences = F.softmax(pred_logits, dim=-1)  # Convert logits to probabilities
        max_class_confidences, _ = class_confidences.max(dim=-1)  # Shape: [batch_size, num_queries]

        # Calculate mask confidence
        pred_masks = outputs['pred_masks']  # Shape: [batch_size, num_queries, H, W]
        mask_confidences = pred_masks.mean(dim=[-2, -1])  # Average foreground probability

        # Combine confidences
        final_confidences = max_class_confidences * mask_confidences  # Element-wise multiplication

        # Update outputs with final confidences
        outputs['final_confidences'] = final_confidences

        return outputs
