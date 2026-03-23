import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from dataset import DrivingCaptures

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, eps=1e-5) -> None:
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(input_dim, embedding_dim, bias=False)
        

    def forward(self, x):
        out = self.fc1(x)
        return out

class Decoder(nn.Module):
    def __init__(self, embedding_dim, input_dim, eps=1e-5):
        super(Decoder, self).__init__()
        self.embedding = embedding_dim
        self.input_dim = input_dim
        self.fc1 = nn.Linear(embedding_dim, input_dim, bias=False)

        
    def forward(self, x):
        out = self.fc1(x)
        return out


class Autoencoder(nn.Module):
    def __init__(self, input_dim, embedding_dim) -> None:
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, embedding_dim)
        self.decoder = Decoder(embedding_dim, input_dim)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



if __name__ == "__main__":
    a = torch.randn([3, 728]).cuda()
    ae = Autoencoder(728, 256).cuda()
    out = ae(a)
    print(out.shape)