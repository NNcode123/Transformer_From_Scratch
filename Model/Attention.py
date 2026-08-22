import torch
import torch.nn as nn
from torchinfo import summary
from .ScaledDotAttention import ScaledDotAttention


class MultiHeadAttention(nn.Module):


    def __init__(self, total_embed_dim , num_heads):

        """
        takes a tensor of shape (batch, seq_len, total_embed_dim)

        applies proj_Q, proj_K, proj_V to that tensor (or tensors of equivalent shape)

        which have shape (total_embed_dim, total_embed_dim)

        then tnesor rearranged into shape (batch, num_heads, seq_len, total_embed_dim//h)

        then runs caled dotbroduct attention on this ... 

        now scaled dot product will take these tensors and then do softmax(QK^T)V
        
        """

        super().__init__()


        self.proj_Q = nn.Linear(in_features = total_embed_dim, out_features = total_embed_dim)
        self.proj_K =  nn.Linear(in_features = total_embed_dim, out_features = total_embed_dim  )
        self.proj_V = nn.Linear(in_features = total_embed_dim, out_features = total_embed_dim)
        self.proj_out = nn.Linear(in_features =  total_embed_dim, out_features = total_embed_dim)
        self.scaled_attn = ScaledDotAttention()

        self.num_heads = num_heads

        assert total_embed_dim % num_heads == 0, "total_embed_dim must be divisible by num_heads"

        self.head_embed_size = (total_embed_dim)//num_heads






    def forward(self, value, key, query, mask = None):

        key_proj = self.proj_K(key)

        query_proj = self.proj_Q(query)

        value_proj = self.proj_V(value)

        key_proj  = key_proj.unflatten(-1, [self.num_heads, self.head_embed_size ]).permute(0,2,1,3)

        query_proj  = query_proj.unflatten(-1, [self.num_heads, self.head_embed_size ]).permute(0,2,1,3)
        
        value_proj  = value_proj.unflatten(-1, [self.num_heads, self.head_embed_size ]).permute(0,2,1,3)

        output_tens = self.scaled_attn(key_proj, query_proj, value_proj, mask)


        return self.proj_out(output_tens.transpose(1,2).flatten(-2,-1))




if __name__ == "__main__":

    batch = 64
    seq_len = 12
    embed_dim = 512

    X = torch.randn(batch,seq_len,embed_dim)


    mattention = MultiHeadAttention(512, 8)

    summary(mattention, input_data=(
        torch.randn(batch, seq_len, embed_dim),  # query
        torch.randn(batch, seq_len, embed_dim),  # key
        torch.randn(batch, seq_len, embed_dim),  # value
    ))





