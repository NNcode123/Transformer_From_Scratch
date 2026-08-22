import torch
import torch.nn as nn
import random
from torchinfo import summary


class TokenEmbedding(nn.Module):


    def __init__(self, vocab_size = 10000, embed_dim = 512):

        super().__init__()

        self.embed_table = nn.Parameter(torch.randn(vocab_size, embed_dim))



    def forward(self, x: torch.Tensor):

        num_batch = x.size(0)

        x = x.flatten()

        return self.embed_table[ x.int(), : ].unflatten(0, [num_batch, -1])


      



class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len = 256, embed_dim = 512):

        super().__init__()

        enc_tensor = torch.zeros((max_seq_len,embed_dim ))

        enc = torch.sin(torch.arange(max_seq_len).unsqueeze(1)/(1000**(torch.arange(0,embed_dim,2)/embed_dim)).unsqueeze(0))

        enc_tensor[:,::2], enc_tensor[:,1::2] = enc,enc

        self.register_buffer("enc", enc_tensor)



    def forward(self, x):

        with torch.no_grad():

            print(f"shape of embed_tensor x: {x.shape}")

            seq_len = x.size(1)

            return x + self.enc[:seq_len]








if __name__ == "__main__":

    y = torch.Tensor([[random.randint(1,10000-1) for _ in range(128)] for _ in range(64)])

    embed = TokenEmbedding()

    pos_enc = PositionalEncoding()

    summary(embed, input_data = y)

    summary(pos_enc, input_data = embed(y))

    



