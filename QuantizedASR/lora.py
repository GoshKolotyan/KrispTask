
import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=16, alpha=32):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        #LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, original_layer.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_layer.out_features, rank))
        
    def forward(self, x):
        #computation + LoRA adaptation
        original_out = self.original_layer(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return original_out + (self.alpha / self.rank) * lora_out
