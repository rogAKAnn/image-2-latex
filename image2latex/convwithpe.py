
import torch
from torch import nn, Tensor

class ConvEncoderWithPE(nn.Module):
    def __init__(self, enc_dim:int, drop_out: float=0.1):
        super(ConvEncoderWithPE, self).__init__()
        self.fe = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, stride=2, padding=0),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(2, stride=2, padding=0),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.MaxPool2d(kernel_size=(1,2), stride=(1,2), padding=0),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
                nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=1),
                nn.BatchNorm2d(512)
        )
        self.dropout = nn.Dropout(0.1)
        self.enc_dim = enc_dim
        self.batch_norm = nn.BatchNorm2d(512)
        
        self.div_term = []
        for i in range(0, self.enc_dim+4, 4):
            self.div_term += [i] * 2
        self.div_term = self.div_term * 2
        self.div_term = list(map(lambda x: float(x)/float(512), self.div_term))
        self.div_term = torch.tensor(self.div_term)
        self.div_term = torch.pow(10000, self.div_term)
        
        self.half_enc_dim = int(self.enc_dim / 2)
            
        
    def forward(self, x: Tensor):
        """x: Tensor (bs, h, w, c)"""
        """Return tensor size (bs, -1, c)"""
        fc = self.fe(x)
        
        fc = fc.permute(0,3,2,1)
        
        bs, h, w, c = fc.size()
        
        pe = torch.zeros(h, w, c)
        
        x_pos = torch.arange(0, w).unsqueeze(1)
        y_pos = torch.arange(0, h).unsqueeze(1)
        y_pos = y_pos.repeat(1, w).unsqueeze(-1)
    
        pe[:,:,0:self.half_enc_dim] = x_pos
        pe[:,:,self.half_enc_dim:] = y_pos
        
        pe = pe / self.enc_dim

        pe[:,:,0::2] = torch.sin(pe[:,:,0::2])
        pe[:,:,1::2] = torch.cos(pe[:,:,1::2])
        
#         pe =pe.unsqueeze(0)
        
#         print(fc.size())
#         fc = self.batch_norm((fc+pe).reshape(bs, c, h, w))

        fc = fc + pe
        
        return self.dropout(fc).reshape(bs, -1, c)     