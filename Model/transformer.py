import torch
import torch.nn as nn
from .encoder_decoder_block import EncoderDecoder
from .embedding import TokenEmbedding, PositionalEncoding
from torchinfo import summary

class Transformer(nn.Module):

    def __init__(self,vocab_size = 10000, num_blocks = 6, embed_dim = 512, num_heads = 8):
        super().__init__()

        self.src_factory = nn.Sequential(TokenEmbedding(vocab_size ,embed_dim), PositionalEncoding(max_seq_len = 256, embed_dim = embed_dim))

        self.tgt_factory = nn.Sequential(TokenEmbedding(vocab_size, embed_dim), PositionalEncoding(max_seq_len = 256, embed_dim = embed_dim))

        self.enc_dec_blocks = EncoderDecoder(num_blocks,embed_dim, num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(in_features = embed_dim, out_features = vocab_size )
        )

    def forward(self, src_input, tgt_input, src_mask, tgt_mask):
        src_embed, tgt_embed = self.src_factory(src_input), self.tgt_factory(tgt_input)

        tgt_embed = self.enc_dec_blocks(src_embed,tgt_embed,src_mask,tgt_mask)

        return self.classifier(tgt_embed)

    
    @torch.no_grad()
    def generate(self, input_tensor, input_mask):

        encode_embed = self.enc_dec_blocks.encode(input_tensor, input_mask )

        input_mask = self.src_factory(input_mask)

        decode_seq = torch.ones(input_mask.size(0),1)


        for iter in range(encode_embed.size(1)):

            decode_mask = torch.triu(torch.ones_like(input_mask)) & input_mask

            output_embed = self.tgt_factory(decode_seq)

            output_embed = self.enc_dec_blocks.decode(output_embed, encode_embed, input_mask, decode_mask)

            next_token_batch = torch.argmax(self.classifier(output_embed[:,-1,:]), axis = 1)       

            decode_seq = torch.concat([decode_seq, next_token_batch], dim  = 1)


        return decode_seq




if __name__ == "__main__":
     transformer = Transformer()

     pos_input = torch.randn((64, 112))

     print(summary(transformer, input_data =(pos_input, pos_input, 
                                          torch.ones((64,112)),
                                          torch.triu(torch.ones((64,112)))
    
                                     ) ))


     