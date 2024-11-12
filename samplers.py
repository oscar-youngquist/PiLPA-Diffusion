import torch
import numpy as np
import sys
import math

class DDPM:
    def __init__(self, generator, num_training_steps=50, num_inference_steps=5, 
                 beta_start=0.00085, beta_end=0.0120, schedule_type='cosine', 
                 snr_min=5.0, covar_scale=10e-3):
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        self.generator = generator
        self.num_training_steps = num_training_steps
        self.num_inference_steps  = 0
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type

        # create the beta schedule
        self.betas = self._get_specified_beta()
        # create alphas 
        self.alphas = 1.0 - self.betas
        # calculate alpha bars \Bar{\alpha}_{t} = \prod_{i=0}^{t} \alpha_{i}
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.timesteps = torch.from_numpy(np.arange(0, self.num_training_steps)[::-1].copy())

        # Calculate the noise-error loss term weight based on the Singal-to-Noise Ratio (SNR) https://arxiv.org/pdf/2303.09556.pdf
        snr = self.alpha_bars / (1.0 - self.alpha_bars)
        self.snr_schedule = torch.minimum(snr, torch.ones_like(snr)*snr_min)

        # Calcualte weighting term for Physics-Informed loss based on the fixed noise covariance https://arxiv.org/pdf/2403.14404
        self.pinn_weights = torch.ones_like(self.alpha_bars)
        for i in range(1, self.num_training_steps):
            self.pinn_weights[i] = 1 / 2*(((1.0 - self.alpha_bars[i-1]) / (1.0 - self.alpha_bars[i])*self.betas[i])/covar_scale)

        # calculate the array of inference timesteps
        self.set_inference_timesteps(num_inference_steps)
    
    def _get_specified_betas(self):
        betas = None
        if self.schedule_type == 'cosine':
            s = 0.008
            steps = self.num_training_steps + 1
            x = torch.linspace(0, self.num_training_steps, steps)
            alphas_cumprod = torch.cos(((x / self.num_training_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0, 0.999)
        elif self.schedule_type == 'quad':
            betas = torch.linspace(self.beta_start ** 0.5, self.beta_end ** 0.5, self.num_training_steps, dtype=torch.float32) ** 2
        elif self.schedule_type == 'linear':
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_training_steps, dtype=torch.float32)
        else:
            print("DDPM._get_specified_betas() %s is an invalid beta-schedule option. Valid options are: cosine, linear, quad")
            sys.exit()

        return betas
    
    def set_inference_timesteps(self, num_inference_steps):
        self.num_inference_steps = num_inference_steps
        inf_step_ratio = self.num_training_steps // self.num_inference_steps
        inference_time_steps = (np.arange(0, self.num_inference_steps) * inf_step_ratio).round()[::-1].copy().astype(np.int64)
        self.inference_time_steps = torch.from_numpy(inference_time_steps)

    def get_beta(self, timestep):
        if timestep < 0 or timestep > self.num_training_steps:
            print("DDPM.get_beta(timestep): timestep {:d} is not valid, exiting".format(timestep))
            sys.exit()

        return self.betas[timestep]

    def get_alpha(self, timestep):
        if timestep < 0 or timestep > self.num_training_steps:
            print("DDPM.get_alpha(timestep):timestep {:d} is not valid, exiting".format(timestep))
            sys.exit()

        return self.alphas[timestep]
    
    def get_alpha_bar(self, timestep):
        if timestep < 0 or timestep > self.num_training_steps:
            print("DDPM.get_alpha_bar(timestep):timestep {:d} is not valid, exiting".format(timestep))
            sys.exit()

        return self.alpha_bars[timestep]
    
    def get_beta_inference(self, timestep):
        if timestep < 0 or timestep > self.num_inference_steps:
            print("DDPM.get_beta_inference(timestep): timestep {:d} is not valid, exiting".format(timestep))
            sys.exit()

        return self.get_beta(self.inference_time_steps[timestep])

    def get_alpha_inference(self, timestep):
        if timestep < 0 or timestep > self.num_inference_steps:
            print("DDPM.get_alpha_inference(timestep):timestep {:d} is not valid, exiting".format(timestep))
            sys.exit()

        return self.get_alpha(self.inference_time_steps[timestep])
    
    def get_alpha_bar_inference(self, timestep):
        if timestep < 0 or timestep > self.num_inference_steps:
            print("DDPM.get_alpha_bar_inference(timestep):timestep {:d} is not valid, exiting".format(timestep))
            sys.exit()
            
        return self.get_alpha_bar(self.inference_time_steps[timestep])
    
    def get_pinn_weight(self, timestep):
        if timestep < 0 or timestep > self.num_training_steps:
            print("DDPM.get_pinn_weight(timestep):timestep {:d} is not valid, exiting".format(timestep))
            sys.exit()

        self.pinn_weights[timestep] 

    def get_minSNR_weight(self, timestep):
        if timestep < 0 or timestep > self.num_training_steps:
            print("DDPM.get_minSNR_weight(timestep):timestep {:d} is not valid, exiting".format(timestep))
            sys.exit()

        return self.snr_schedule[timestep]

    


        
        