import torch 
import torch.nn as nn

class FFN(nn.Module):

    def __init__(self, input_features, scale_factor = 4):
        super().__init__()

        self.layers = nn.Sequential(nn.Linear(in_features =input_features, out_features = input_features * scale_factor),
                                    nn.ReLU(),
                                    nn.Linear(in_features = input_features * scale_factor, out_features = input_features )
                                    )



    def forward(self, X):
        return self.layers(X)




class residual_norm(nn.Module):
    def __init__(self, layer: nn.Module):

        self.F = layer


    def forward(self, x):
        return x + self.F(x)


