import torch
import numpy as np
import sys
import math

class DDPM:
    def __init__(self, options):
        # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
        self.options = options
        self.num_training_steps = options["num_training_steps"]
        self.num_inference_steps  = 0
        self.beta_start = options["beta_start"]
        self.beta_end = options["beta_end"]
        self.schedule_type = options["schedule_type"]
        covar_scale = options["covar_scale"]
        snr_min_value = options["snr_min_value"]

        # create the beta schedule
        self.betas = self._get_specified_betas()
        # create alphas 
        self.alphas = 1.0 - self.betas
        # calculate alpha bars \Bar{\alpha}_{t} = \prod_{i=0}^{t} \alpha_{i}
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)
        self.timesteps = torch.from_numpy(np.arange(0, self.num_training_steps)[::-1].copy())

        # print("Training Time-Steps: ", self.timesteps)

        # Calculate the noise-error loss term weight based on the Signal-to-Noise Ratio (SNR) https://arxiv.org/pdf/2303.09556.pdf
        # snr_{t} = min(\Bar{\alpha}_{t} / (1.0 - \Bar{\alpha}_{t}), snr_min_value)
        snr = self.alpha_bars / (1.0 - self.alpha_bars)
        self.snr_schedule = torch.minimum(snr, torch.ones_like(snr)*snr_min_value)

        # Calcualte weighting term for Physics-Informed loss based on the fixed noise covariance https://arxiv.org/pdf/2403.14404
        self.pinn_weights = torch.ones_like(self.alpha_bars)

        for t in range(1, self.num_training_steps):
            # Sigma_{t} = (1-\Bar{\alpha}_{t-1})/(1-\Bar{\alpha}_T}) * \Beta_{t}
            # \Bar{\Sigma}_{t} = \Simga_{t} / covar_scale
            # pinn_weight = 1 / (2*\Bar{\Sigma}_{t})
            sigma = ((1.0 - self.alpha_bars[t-1]) / (1.0 - self.alpha_bars[t]))*self.betas[t]
            # print("sigma: ", sigma)
            sigma_bar = sigma / covar_scale
            # print("sigma_bar: ", sigma_bar)
            self.pinn_weights[t] = 1 / (2*sigma_bar)
            # print("self.pinn_weights[t]: ", self.pinn_weights[t])
        self.pinn_weights[0] = self.pinn_weights[1]



        # calculate the array of inference timesteps
        self.set_inference_timesteps(options["num_inference_steps"])
    
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

        # print("Inference Time-Steps: ", self.inference_time_steps)

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

        return self.pinn_weights[timestep] 

    def get_minSNR_weight(self, timestep):
        if timestep < 0 or timestep > self.num_training_steps:
            print("DDPM.get_minSNR_weight(timestep):timestep {:d} is not valid, exiting".format(timestep))
            sys.exit()

        return self.snr_schedule[timestep]

    


        
        