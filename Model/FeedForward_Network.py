import torch 
import torch.nn as nn
from torchinfo import summary

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
    def __init__(self, embed_dim = 512):

        super().__init__()

        self.norm = nn.LayerNorm(normalized_shape = embed_dim)


    def forward(self, x, f_x):
        y = x + f_x
        return self.norm(y)




if __name__ == "__main__":

    tens = torch.randn((30, 100, 512))

    layer = FFN(input_features = 512)

    residblock = residual_norm()


    print(summary(residblock, input_data = (tens,tens)))
    print(summary(layer, input_data  = tens ))
