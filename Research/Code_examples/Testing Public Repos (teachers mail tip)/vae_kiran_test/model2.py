
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math



class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(VAE, self).init()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        