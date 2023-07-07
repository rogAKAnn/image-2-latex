import torch
from torch import nn, Tensor

class Attention(nn.Module):
    def __init__(self, enc_dim: int = 512, dec_dim: int = 512, attn_dim: int = 512):
        super().__init__()
        self.dec_attn = nn.Linear(dec_dim, attn_dim, bias=False)
        self.enc_attn = nn.Linear(enc_dim, attn_dim, bias=False)
        self.full_attn = nn.Linear(attn_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h: Tensor, V: Tensor):
        """
            input:
                h: (b, dec_dim) hidden state vector of decoder
                V: (b, w * h, enc_dim) encoder matrix representation
            output:
                context: (b, enc_dim)
        """

        attn_1 = self.dec_attn(h) # attn_1: (b, attn_dim)
        attn_2 = self.enc_attn(V) # attn_2: (b, w * h, attn_dim)
                
        # attn: (b, w*h)
        attn = self.full_attn(torch.tanh(attn_1.unsqueeze(1) + attn_2)).squeeze(2)
        
        # alpha: (b, w*h) -> each sum elements sum = 1
        alpha = self.softmax(attn)
        
        # (b, w*h, 1) * (b, w*h, dec_dim) = (b, w*h, dec_dim).sum(dim=1) -> (b, dec_dim)
        context = (alpha.unsqueeze(2) * V).sum(dim=1)
        return context
