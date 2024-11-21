import torch
from torch import nn
from torch.nn import functional as F
import math
from modules import TimeEmbedding, UNET_Attention_Block, UNET_Block

# Cross-entropy loss
class Discriminator(nn.Module):
    def __init__(self, options):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(options['dim_a'], options['discrim_h1'])
        self.fc2 = nn.Linear(options['discrim_h1'], options['discrim_h2'])
        self.output = nn.Linear(options['discrim_h2'], options['num_c'])

        self.layer_norm = nn.LayerNorm(options['discrim_h2'])

        # init the weights using xavier-intialization
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.output.weight)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        x = F.silu(x)
        x = self.output(x)
        return x


# Define UNet

# UNET_Block(self, in_size, out_size, time_size)
# UNET_Attention_Block(self, n_heads, n_embed, d_context, in_proj_bias=True, out_proj_bias=True)

class UNet(nn.Module):
    def __init__(self, options):
        super(UNet, self).__init__()
        
        self.options = options
        
        # Down-Sample blocks
        self.input_block = UNET_Block(options["dim_a"], options["diff_out_1"], options["time_size"])
        self.attention_1 = UNET_Attention_Block(options["n_heads"], options["diff_out_1"], options["context_size"])
        self.block_2 = UNET_Block(options["diff_out_1"], options["diff_out_2"], options["time_size"])
        self.attention_2 = UNET_Attention_Block(options["n_heads"], options["diff_out_2"], options["context_size"])

        # bottle-neck layer
        self.bottle_neck_layer = nn.Linear(options["diff_out_2"], options["diff_out_2"])
        self.bottle_neck_norm = nn.LayerNorm(options["diff_out_2"])

        # Up-Sample blocks
        self.block_3 = UNET_Block(options["diff_out_2"], options["diff_out_1"], options["time_size"])
        self.attention_3 = UNET_Attention_Block(options["n_heads"], options["diff_out_1"], options["context_size"])
        self.output_block = UNET_Block(options["diff_out_1"], options["dim_a"], options["time_size"])
        self.attention_4 = UNET_Attention_Block(options["n_heads"], options["dim_a"], options["context_size"])
        self.skip_2_norm = nn.LayerNorm(options["diff_out_2"])
        self.skip_1_norm = nn.LayerNorm(options["diff_out_1"])

        self.b1_dropout = nn.Dropout(options["dropout"])
        self.b2_dropout = nn.Dropout(options["dropout"])
        self.b3_dropout = nn.Dropout(options["dropout"])
        self.b4_dropout = nn.Dropout(options["dropout"])
        self.bottle_neck_dropout = nn.Dropout(options["dropout"])

        # init some weights using xavier-intialization
        torch.nn.init.xavier_uniform_(self.bottle_neck_layer.weight)

    # UNET_Block.forward(self, x, time)
    # UNET_Attention_Block(self, x, context)
    #     note UNET_Block ends with a F.silu activation, UNET_Attention does NOT
    #     norm -> acitvation -> weights
    def forward(self, x, time, context):
        # Down-sample blocks
        # (Batch_Size, options["dim_a"]) (Batch_Size, options["time_size"]) 
        #       -> (Batch_Size, options["diff_out_1"])
        x = self.input_block(x, time)
        x = self.b1_dropout(x)
        # (Batch_Size, options["diff_out_1"]) (Batch_Size, options["context_size"]) 
        #       -> (Batch_Size, options["diff_out_1"])
        x = self.attention_1(x, context)
        # (Batch_Size, options["diff_out_1"])
        skip_1 = x
        # (Batch_Size, options["diff_out_1"]) (Batch_Size, options["time_size"]) 
        #       -> (Batch_Size, options["diff_out_2"])
        x = self.block_2(x, time)
        x = self.b2_dropout(x)
        # (Batch_Size, options["diff_out_2"]) (Batch_Size, options["context_size"]) 
        #       -> (Batch_Size, options["diff_out_2"])
        x = self.attention_2(x, context)
        # (Batch_Size, options["diff_out_2"])
        skip_2 = x

        # bottle neck
        # (Batch_Size, options["diff_out_2"]) -> (Batch_Size, options["diff_out_2"])
        x = self.bottle_neck_norm(x)
        x = F.silu(x)
        x = self.bottle_neck_layer(x)
        x = self.bottle_neck_dropout(x)

        # up-sample blocks
        #     add skip-connection -> normalize -> activation function -> weights

        # (Batch_Size, options["diff_out_2"]) + (Batch_Size, options["diff_out_2"])
        #    -> (Batch_Size, options["diff_out_2"])
        x = self.skip_2_norm(x+skip_2)
        x = F.silu(x)
        # (Batch_Size, options["diff_out_2"]) (Batch_Size, options["time_size"]) 
        #       -> (Batch_Size, options["diff_out_1"])
        x = self.block_3(x, time)
        x = self.b3_dropout(x)
        # (Batch_Size, options["diff_out_1"]) (Batch_Size, options["context_size"]) 
        #       -> (Batch_Size, options["diff_out_1"])
        x = self.attention_3(x, context)
        
        # (Batch_Size, options["diff_out_1"]) + (Batch_Size, options["diff_out_1"])
        #    -> (Batch_Size, options["diff_out_1"])
        x = self.skip_1_norm(x+skip_1)
        x = F.silu(x)
        # (Batch_Size, options["diff_out_1"]) (Batch_Size, options["time_size"]) 
        #       -> (Batch_Size, options["dim_a"])
        x = self.output_block(x, time)
        x = self.b4_dropout(x)
        # (Batch_Size, options["dim_a"]) (Batch_Size, options["context_size"]) 
        #       -> (Batch_Size, options["dim_a"])
        x = self.attention_4(x, context)

        return x
        

# Define Diffusion Model
#     this inlcudes the scheduler...
class Diffusion(nn.Module):
    def __init__(self, options):
        super(Diffusion, self).__init__()
        self.time_embedding = TimeEmbedding(options["time_size"])
        self.unet = UNet(options)
        self.options = options

    def forward(self, latent, context, timestep):
        # latent: (Batch_Size, options["dim_a"])
        # context: (Batch_Size, options["context_size"])
        # time: (Batch_Size, options["time_size"])
        # get the time-pos-embedding for this time-step
        time_pose_embed = self.get_time_pos_embedding(timestep)

        # print(time_pose_embed.device)
        time_pose_embed = time_pose_embed.to(latent.device)
        # print(latent.device)
        # print(time_pose_embed.device)
        # print(next(self.time_embedding.parameters()).device)

        time_pose_embed = self.time_embedding(time_pose_embed)
        
        output = self.unet(latent, time_pose_embed, context)

        return output
    
    def sample_timestep_noise(self, latent, timestep):
        # sample random noise for this step
        if timestep > 1:
            noise = torch.randn(latent.shape).to(latent.device)
        else:
            noise = 0

        return noise
    
    def get_time_pos_embedding(self, timestep):
        # print(timestep)
        # print(timestep.shape)
        half_dim = self.options["time_size"]//2
        # definition from "Attention is all you need" paper https://arxiv.org/abs/1706.03762
        freqs = torch.pow(10000, -torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim)
        x = None
        if len(timestep.shape) < 1:
            x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
        else:
            x = torch.tensor(timestep[:,None], dtype=torch.float32) * freqs[None]
        # x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
        # x = torch.tensor(temp, dtype=torch.float32)[:, None] * freqs[None]
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

    def get_pred_denoised_latent(self, latent, context, time, alpha_bar):
        # get noise prediction from UNet
        e_hat = self.forward(latent, context, time)
        # perform denoising step
        pre_scale = 1 / math.sqrt(alpha_bar)
        e_scale = math.sqrt(1-alpha_bar)
        no_noise_pred = pre_scale * (latent - e_scale * e_hat)

        return no_noise_pred

    def get_pred_denoised_latent_no_forward(self, e_hat, latent, alpha_bar):
        # perform denoising step
        pre_scale = 1 / math.sqrt(alpha_bar)
        e_scale = math.sqrt(1-alpha_bar)
        no_noise_pred = pre_scale * (latent - e_scale * e_hat)

        return no_noise_pred

    def denoise_step(self, latent, context, time, alpha, alpha_bar, beta):
        new_latent = None
        with torch.no_grad():
            noise = self.sample_timestep_noise(latent, time)

            # get noise prediction from UNet
            e_hat = self.forward(latent, context, time)
            # perform denoising step
            pre_scale = 1 / math.sqrt(alpha)
            e_scale = (1-alpha) / math.sqrt(1-alpha_bar)
            noise_scale = math.sqrt(beta)
            new_latent = pre_scale * (latent - e_scale * e_hat) + noise_scale*noise
        return new_latent
    

    def denoise_step_train(self, latent, context, time, alpha, alpha_bar, beta):
        new_latent = None
        noise = self.sample_timestep_noise(latent, time)

        # get noise prediction from UNet
        e_hat = self.forward(latent, context, time)
        # perform denoising step
        pre_scale = 1 / math.sqrt(alpha)
        e_scale = (1-alpha) / math.sqrt(1-alpha_bar)
        noise_scale = math.sqrt(beta)
        new_latent = pre_scale * (latent - e_scale * e_hat) + noise_scale*noise
        
        return new_latent
    

    def denoise_step_no_forward(self, e_hat, latent, time, alpha, alpha_bar, beta):
        noise = self.sample_timestep_noise(latent, time)
        # perform denoising step
        pre_scale = 1 / math.sqrt(alpha)
        e_scale = (1-alpha) / math.sqrt(1-alpha_bar)
        noise_scale = math.sqrt(beta)
        new_latent = pre_scale * (latent - e_scale * e_hat) + noise_scale*noise
        return new_latent
    
    def guided_denoise_step(self, latent, context, time, alpha, alpha_bar, beta, guide, guide_weight):        
        with torch.no_grad():
            # get noise prediction from UNet
            e_hat = self.forward(latent, context, time)
            
            e_hat_guided = e_hat + guide_weight*guide

            return self.denoise_step_no_forward(e_hat_guided, latent, time, alpha, alpha_bar, beta)