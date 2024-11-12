# This file holds NN layer definitions and other NN model utilities used in the models.py file

import torch
from torch import nn
from torch.nn import functional as F
import math 

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super(TimeEmbedding, self).__init__()
        # introduce a slight bottle-neck
        self.lin_1 = nn.Linear(n_embed, n_embed // 2)
        self.lin_2 = nn.Linear(n_embed // 2, n_embed)

        # init the weights using xavier-intialization
        torch.nn.init.xavier_uniform_(self.lin_1.weight)
        torch.nn.init.xavier_uniform_(self.lin_2.weight)

    def forward(self, time):
        x = self.lin_1(time)
        x = F.silu(x)
        x = self.lin_2(x)

        return x

class UNET_Block(nn.Module):
    def __init__(self, in_size, out_size, time_size):
        super(UNET_Block, self).__init__()

        # linear layers
        self.lin = nn.Linear(in_size, out_size)
        self.time_lin = nn.Linear(time_size, in_size)
        
        # layer-norm normalization layers
        self.layer_norm = nn.LayerNorm(in_size)

        # init the weights using xavier-intialization
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.time_lin.weight)

    def forward(self, x, time):
        # following the norm -> acitvation -> weights flow from diffusion model literature
        # normalize the outputs
        time = F.silu(time)
        time = self.time_lin(time)

        # merge, the time features and data features normalize, activation functions
        x = F.silu(x)
        x = x + time
        x = self.layer_norm(x)
        x = F.silu(x)

        # use the linear layer for the merged features
        x = self.lin(x)
        x = F.silu(x)

        return x
    
class UNET_Basic_Conditioning_Block(nn.Module):
    def __init__(self, in_size, out_size, context_size):
        super(UNET_Block, self).__init__()

        # linear layers
        self.lin = nn.Linear(in_size, out_size)
        self.context_lin = nn.Linear(context_size, in_size)
        
        # layer-norm normalization layers
        self.layer_norm = nn.LayerNorm(in_size)

        # init the weights using xavier-intialization
        torch.nn.init.xavier_uniform_(self.lin.weight)
        torch.nn.init.xavier_uniform_(self.context_lin.weight)

    def forward(self, x, context):
        
        # following the norm -> acitvation -> weights flow from diffusion model literature
        # normalize the outputs
        context = F.silu(context)
        context = self.context_lin(context)

        # merge, the time features and data features nromalize, activation functions
        #     NOTE: this comes after a UNetBlock, so x has been activated
        x = x + context
        x = self.layer_norm(x)
        x = F.silu(x)

        # use the linear layer for the merged features
        x = self.lin(x)
        x = F.silu(x)

        return x

# performs cross attention with conditioning sample from VAE
class UNET_Attention_Block(nn.Module):
    def __init__(self, n_heads, n_embed, d_context, in_proj_bias=False, out_proj_bias=True):
        super(UNET_Attention_Block, self).__init__()

        # linear layers
        self.q_proj = nn.Linear(n_embed, n_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_context, n_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_context, n_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(n_embed, n_embed, bias=out_proj_bias)

        self.n_heads = n_heads
        self.d_head = n_embed // n_heads
        
        # layer-norm normalization layers
        self.layer_norm = nn.LayerNorm(n_embed)

        # init the weights using xavier-intialization
        torch.nn.init.xavier_uniform_(self.q_proj.weight)
        torch.nn.init.xavier_uniform_(self.k_proj.weight)
        torch.nn.init.xavier_uniform_(self.v_proj.weight)
        torch.nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, y):
        # x - latents # (Batch_Size, n_embed)
        # y - context (Batch_Size, d_context)
        input_shape = x.shape
        batch_size, d_embed = input_shape
        
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, self.n_heads, self.d_head)

        # (Batch_Size, n_embed) -> (Batch_Size, n_embed)
        q = self.q_proj(x)
        # (Batch_Size, d_context) -> (Batch_Size, n_embed)
        k = self.k_proj(y)
        # (Batch_Size, d_context) -> (Batch_Size, n_embed)
        v = self.v_proj(y)

        # Split into multiple heads...
        # (Batch_Size, n_embed) -> (Batch_Size, n_heads, d_head)
        q = q.view(interim_shape)
        # (Batch_Size, n_embed) -> (Batch_Size, n_heads, d_head)
        k = k.view(interim_shape)
        # (Batch_Size, n_embed) -> (Batch_Size, n_heads, d_head)
        v = v.view(interim_shape)

        # (Batch_Size, n_heads, d_head) @ (Batch_Size, d_head, n_heads) -> (Batch_Size, n_heads, n_heads)
        weight = torch.matmul(q, k.transpose(-1,-2))

        # (Batch_Size, n_heads, n_heads)
        weight /= math.sqrt(self.d_head)

        # (Batch_Size, n_heads, n_heads)
        weight = F.softmax(weight, dim=-1)

        # (Batch_Size, n_heads, n_heads) @ (Batch_Size, n_heads, d_head) -> (Batch_Size, n_heads, d_heads)
        output = torch.matmul(weight, v)

        # (Batch_Size, n_heads, d_heads) -> (Batch_size, n_embed)
        output = output.view(input_shape)

        # (Batch_size, n_embed) -> (Batch_size, n_embed)
        output = self.out_proj(output)

        return output