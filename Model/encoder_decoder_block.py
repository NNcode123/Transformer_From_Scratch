import torch.nn as nn
import torch
from .Attention import MultiHeadAttention
from .FeedForward_Network import FFN, residual_norm

from torchinfo import summary



class Encoder(nn.Module):
    def __init__(self, embed_dim = 512, num_heads = 8):


         super().__init__()

         self.att_layer: MultiHeadAttention = MultiHeadAttention(total_embed_dim  =embed_dim, num_heads = num_heads)

         self.att_resid = residual_norm(embed_dim = embed_dim)

         self.ffn = FFN(input_features  = embed_dim, scale_factor = 2)

         self.ffd_resid = residual_norm(embed_dim = embed_dim )


    def forward(self, x, mask = None):
        
        att_x =  self.att_layer( x, x, x, mask)
        x  = self.att_resid(x, att_x)
        ffn_x = self.ffn(x)
        return self.ffd_resid(x,ffn_x)




class Decoder(nn.Module):
    def __init__(self, embed_dim = 512, num_heads = 8):

        super().__init__()

        self.masked_att_layer: MultiHeadAttention = MultiHeadAttention(total_embed_dim  =embed_dim, num_heads = num_heads)
        
        self.masked_att_resid = residual_norm(embed_dim = embed_dim)

        self.att_layer = MultiHeadAttention(total_embed_dim = embed_dim, num_heads = num_heads )

        self.att_resid = residual_norm(embed_dim = embed_dim)

        self.ffn = FFN(input_features  = embed_dim, scale_factor = 2)
                
        self.ffd_resid = residual_norm(embed_dim = embed_dim )


    def forward(self,  x, enc_values_keys, src_mask, tgt_mask ):

        masked_att_x = self.masked_att_layer(x,x,x,tgt_mask)

        masked_res_x = self.masked_att_resid(x, masked_att_x)

        second_att_x = self.att_layer(enc_values_keys, enc_values_keys, masked_res_x, src_mask)

        second_res_x = self.att_resid(second_att_x, masked_res_x)

        return self.ffd_resid(second_res_x, self.ffn(second_res_x))






        




class EncoderDecoder(nn.Module):


    def __init__(self, num_blocks = 6, embed_dim  =512, num_heads = 8):

        super().__init__()

        self.encoders = nn.ModuleList([Encoder(embed_dim, num_heads) for _ in range(num_blocks)])

        self.decoders = nn.ModuleList([Decoder(embed_dim, num_heads) for _ in range(num_blocks)])


    def forward(self, src_embed, tgt_embed, src_mask, tgt_mask):

        for encoder in self.encoders:
            src_embed = encoder(src_embed, src_mask)


        for decoder in self.decoders:
            tgt_embed = decoder(tgt_embed, src_embed, src_mask, tgt_mask)


        return tgt_embed

 
if __name__ == "__main__":

    encBlock = Encoder()

    enc = EncoderDecoder()

    pos_input = torch.randn((64, 112))

    pos_embed = torch.randn((64,112,512))


    """
    print(summary(enc, input_data =  (pos_embed, pos_embed, 
                                      torch.ones((64,112)),
                                      torch.triu(torch.ones((64,112)))

                                 )
                                      )
    """
   
    print(summary(encBlock, input_data = pos_embed))



