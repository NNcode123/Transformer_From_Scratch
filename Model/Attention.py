import torch
import torch.nn as nn
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

        self.num_heads = num_heads

        assert total_embed_dim % num_heads == 0, "total_embed_dim must be divisible by num_heads"

        self.head_embed_size = (total_embed_dim)//num_heads






    def forward(self, key, query, value):

        key_proj = self.proj_K(key)

        query_proj = self.proj_Q(query)

        value_proj = self.prov_V(query)

        key_proj  = key_proj.unflatten(-1, [self.num_heads, self.head_embed_size ]).permute(0,2,1,3)

        query_proj  = query_proj.unflatten(-1, [self.num_heads, self.head_embed_size ]).permute(0,2,1,3)
        
        value_proj  = value_proj.unflatten(-1, [self.num_heads, self.head_embed_size ]).permute(0,2,1,3)

        output_tens: torch.Tensor = ScaledDotAttention(key_proj, query_proj, value_proj)


        return output_tens.transpose(1,2).flatten(-2,-1)







