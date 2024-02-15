from typing import Optional
from torch import nn, Tensor
from torch.nn import functional as F


class FFNLayer(nn.Module):
    """
    Parameters:
    - d_model (int): The number of expected features in the input and output tensor, the dimensionality of the embedding.
    - dim_feedforward (int): The dimension of the feedforward network model, which is the size of the hidden layer.
    - dropout (float): The dropout value, which is the probability of an element to be zeroed. Helps in regularizing the model.
    - activation (str): The activation function of the intermediate layer.    
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = get_activation_fn(activation)
        self._reset_parameters()
    
    
    def _reset_parameters(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)


    def with_positional_embedding(self, tensor, position: Optional[Tensor]):
        return tensor if position is None else tensor + position


    def forward(self, target_tensor):
        target_tensor_2 = self.linear2(self.dropout(self.activation(self.linear1(target_tensor))))
        target_tensor = target_tensor + self.dropout(target_tensor_2)
        target_tensor = self.norm(target_tensor)
        return target_tensor
    

class MLP(nn.Module):
    """ Very simple multi-layer perceptron"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
       
def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")