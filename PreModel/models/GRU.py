import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.channels = configs.enc_in
        self.hiddensize = configs.d_model
        # self.hiddensize = 32
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.grucell = nn.GRUCell(input_size=self.channels, hidden_size=self.hiddensize, bias=True)
        self.prel = nn.Linear(self.hiddensize, self.pred_len, bias=True)
    
    def forward(self, x):
        # print(x.shape)

        x = x.reshape(-1, self.seq_len)

        hiddenstate = torch.zeros((x.shape[0], self.hiddensize)).to(x.device)

        for i in range(self.seq_len):
            hiddenstate = self.__step__(x[:, i:i+1], hiddenstate)
            # hiddenstate = self.bn(hiddenstate)

        return self.prel(hiddenstate).reshape(-1, self.pred_len, 1)


    def __step__(self, data,  hiddenstate):
        # data = data.reshape(-1, 1)
        # mask = mask.reshape(-1, 1)
        hiddenstate = self.grucell(data, hiddenstate)
        return hiddenstate