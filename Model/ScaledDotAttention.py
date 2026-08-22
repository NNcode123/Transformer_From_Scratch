import torch
from torch import nn
from typing import Union, Optional

class ScaledDotAttention(nn.Module):

    def __init__(self):

        super().__init__()




    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values:torch.Tensor, mask: Union[torch.Tensor, None] = None):

        """

        The tensors will have the following shape:

        queries: (batch, num_heads, seq_len, embed_size)
        keys: (batch, num_heads, seq_len, embed_size)
        values: (batch, num_heads, seq_len, embed_size_v)

        Then I = q @ k^T/sqrt(d_k) is (batch,num_heads, seq_len, seq_len)

        then ignoring the batch,num_head dims (those are for paralleization purposes), the resulting matrix is a matrix where 

        matrix_ij determines how much token_i cares about token_j . 

        before taking_softmax, use the (batch, seq_len) mask to ignore invalid positions okay add a 1 dim to create (batch,1,1,seq_len)

        now take softmax over dimension 1 of matrix_ij which will give us the following 

        it will make every row of the matrix a valid probaility distribution 



        taking soft_max over dim -1 gives us the same shape. 

        the mask will have shape: (seq_len, embed_size)

        Finally softmax(I)@V will have shape (batch, num_heads, seq_len, embed_size)
        
        
        """

        d_k  = queries.size(-1)

        res = queries @ keys.transpose(-1,-2)

        size_diff = queries.ndim  - mask.ndim if mask != None else 0
    
        for _ in range(size_diff):
            mask = mask.unsqueeze(1) if mask != None else None
            

        scaled_res:torch.Tensor = res/(d_k**0.5)

        if mask != None:

            scaled_res.masked_fill((mask == 0), float("-inf"))


        scaled_res = torch.softmax(scaled_res, dim = -1)

        return scaled_res @ values










