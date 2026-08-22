import torch
import torch.nn as nn
from .encoder_decoder_block import EncoderDecoder
from .embedding import TokenEmbedding, PositionalEncoding
from .embedding import TokenEmbedding
from torchinfo import summary

class Transformer(nn.Module):

    def __init__(self,vocab_size = 10000, num_blocks = 6, embed_dim = 512, num_heads = 8):
        super().__init__()

        self.src_factory = nn.Sequential(TokenEmbedding(vocab_size ,embed_dim), PositionalEncoding(max_seq_len = 256, embed_dim = embed_dim))

        self.tgt_factory = nn.Sequential(TokenEmbedding(vocab_size, embed_dim), PositionalEncoding(max_seq_len = 256, embed_dim = embed_dim))

        self.enc_dec_blocks = EncoderDecoder(num_blocks,embed_dim, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(in_features = embed_dim, out_features = vocab_size ),
            nn.Softmax(dim = -1)
        )

    def forward(self, src_input, tgt_input, src_mask, tgt_mask):
        src_embed, tgt_embed = self.src_factory(src_input), self.tgt_factory(tgt_input)

        tgt_embed = self.enc_dec_blocks(src_embed,tgt_embed,src_mask,tgt_mask)

        return self.classifier(tgt_embed)




if __name__ == "__main__":
     transformer = Transformer()
     print(summary(transformer, input_data =(pos_input, pos_input, 
                                          torch.ones((64,112)),
                                          torch.triu(torch.ones((64,112)))
    
                                     ) ))