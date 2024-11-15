import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
import matplotlib.pyplot as plt
import json
import os 
import utils
from vae_context_encoder_models import VAE_Conditioning_Model
from samplers import DDPM
from diffusion import Diffusion, Discriminator
from torchsummary import summary


class Train_PICD():

    def __init__(self, options):
        if sys.platform == 'win32':
            NUM_WORKERS = 0 # Windows does not support multiprocessing
        else:
            NUM_WORKERS = 2
        print('running on ' + sys.platform + ', setting ' + str(NUM_WORKERS) + ' workers')

        # Extract some global values
        self.options = options
        self.dim_a = options["dim_a"]
        self.features = options["features"]
        self.label = options["label"]
        self.labels = options["labels"]
        self.dataset = options["dataset"]
        self.lr = options["learning_rate"]
        self.lr_discrim = options["learning_rate_dis"]
        self.model_save_freq = options["model_save_freq"]
        self.epochs = options['num_epochs']
        self.gamma = options["gamma"]
        self.discrim_save_freq = options['frequency_h']
        self.sn = options['SN']
        self.train_data_path = options['train_path']
        self.test_data_path = options['test_path']
        self.output_path_base = options['output_path']
        self.device = options['device']
        self.display_progress = options['display_progress']
        self.model_name = options['model_name']
        self.vae_warm_up = options['vae_warm_up']
        self.ema_warm_up = options['ema_warm_up']
        self.ema_decay = options['ema_decay']
        self.log_file_path = options['tensorboard_folder']
        self.vae_kl_weight = options['kl_weight']
        self.residual_weight = options['res_weight']
        self.logger = SummaryWriter(self.log_file_path)

        print("\n***********Model output path: ", self.output_path_base)
        print("\t Loading Data....")
        RawData = utils.load_data(self.train_data_path)
        Data = utils.format_data(RawData, features=self.features, output=self.label, body_offset=options["body_offset"])
        self.Data = Data

        RawDataTest = utils.load_data(self.test_data_path) # expnames='(baseline_)([0-9]*|no)wind'
        self.TestData = utils.format_data(RawDataTest, features=self.features, output=self.label, body_offset=options["body_offset"])

        # Update options dict based on the shape of the data
        self.options['dim_x'] = Data[0].X.shape[1]
        self.options['dim_y'] = Data[0].Y.shape[1]
        self.options['num_c'] = len(Data)

        self.num_train_classes = options["num_c"]
        self.num_test_classes = len(self.TestData)

        # Make the dataloaders globally accessible
        self.Trainloader = []
        self.Adaptloader = []
        self.num_batches = 100000000000000
        for i in range(self.num_train_classes ):
            fullset = utils.MyDataset(Data[i].X, Data[i].Y, Data[i].C)
            
            l = len(Data[i].X)
            if options['shuffle']:
                trainset, adaptset = random_split(fullset, [int(2/3*l), l-int(2/3*l)])
            else:
                trainset = utils.MyDataset(Data[i].X[:int(2/3*l)], Data[i].Y[:int(2/3*l)], Data[i].C) 
                adaptset = utils.MyDataset(Data[i].X[int(2/3*l):], Data[i].Y[int(2/3*l):], Data[i].C)

            trainloader = DataLoader(trainset, batch_size=options['phi_shot'], shuffle=options['shuffle'], num_workers=NUM_WORKERS)
            adaptloader = DataLoader(adaptset, batch_size=options['K_shot'], shuffle=options['shuffle'], num_workers=NUM_WORKERS)

            if len(trainloader) < self.num_batches:
                self.num_batches = len(trainloader)

            self.Trainloader.append(trainloader) # for training model
            self.Adaptloader.append(adaptloader) # for LS on M

        
        print("\t Done Loading Data. Num Batches: {:d}".format(self.num_batches))
        print("\t Building Models....")
        # Build Networks
        self.vae_encoder = VAE_Conditioning_Model(options)
        self.discriminator = Discriminator(options)
        self.diff_model = Diffusion(options)

        # Make copies for EMA
        self.vae_encoder_ema = VAE_Conditioning_Model(options)
        self.discriminator_ema = Discriminator(options)
        self.diff_model_ema = Diffusion(options)
        print("\t Done Building Models")
        
        # build DDPM scheduler
        self.scheduler = DDPM(options)

        # build network-specific loss functions
        #     note - VAE has built-in loss functions
        self.diff_loss = nn.MSELoss(reduction='none')
        self.discrim_loss = nn.CrossEntropyLoss(reduction='none')

        # create optimizers
        self.vae_encoder_optimizer = optim.Adam(self.vae_encoder.encoder.parameters(), lr=self.lr, eps=1e-08)
        self.vae_decoder_optimizer = optim.Adam(self.vae_encoder.decoder.parameters(), lr=self.lr, eps=1e-08)
        self.diffusion_optimizer = optim.Adam(self.diff_model.parameters(), lr=self.lr, eps=1e-08)
        self.discriminator_optimizer = optim.Adam(self.diff_model.parameters(), lr=self.lr_discrim, eps=1e-08)

        self.histogram_adder(-1)

        num_params_vae = self.count_parameters(self.vae_encoder.encoder)
        num_params_diff = self.count_parameters(self.diff_model)

        print("Number of parameters in VAE encoder: {:d}, # params in diff-model: {:d}".format(num_params_vae, num_params_diff))
        # print("\tDone building training class")


    # dump current config file to a json
    def save_config(self, output_path):
        with open(output_path, "w+") as f:
            json.dump(self.options, f)
            f.close()

    def add_network_graphs():
        pass

    def count_parameters(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        return total_params

    # Perform Expoential Moving Average or "Temporal Averaging" to model weights https://arxiv.org/pdf/1412.6980 
    def ema_accumulate(self, model1, model2, decay=0.9999):
        par1 = dict(model1.named_parameters())
        par2 = dict(model2.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))

    # generate interactive graph of 

    # Add histrogram of current weight values to tesnorboard log
    #     update for each model specifically
    def histogram_adder(self, curr_epoch):
        # iterate over parameters and add their values to a histogram 
        for name, params in self.vae_encoder_ema.named_parameters():
            _name = "vae_weights/" + name
            self.logger.add_histogram(_name, params, curr_epoch)

        # iterate over parameters and add their values to a histogram 
        for name, params in self.diff_model_ema.named_parameters():
            _name = "diff_weights/" + name
            self.logger.add_histogram(_name, params, curr_epoch)

        # iterate over parameters and add their values to a histogram 
        for name, params in self.discriminator_ema.named_parameters():
            _name = "discrim_weights/" + name
            self.logger.add_histogram(_name, params, curr_epoch)

    def pretrain_vae_encoder(self):
        for warm_up_epoch in range(0, self.vae_warm_up):
             # randomly iterate over the condition-specific datasets self.num_batches
            for batch_idx in range(0, self.num_batches):
                # print("\tStarting Batch: {:d}".format(batch_idx))
                # Randomize the order in which we iterate over the condition-specific datasets
                arr = np.arange(self.num_train_classes)
                np.random.shuffle(arr)
                for i in arr:
                    data = None
                    with torch.no_grad():
                        trainloader = self.Trainloader[i]
                        data = next(iter(trainloader))

                    # print("\tTraining VAE and Diffusion Model....")
                    #     set models to training mode
                    self.vae_encoder.train()
                    #     move models to GPU
                    self.vae_encoder.to(self.device)

                    #     zero the optimizers
                    self.vae_decoder_optimizer.zero_grad()
                    self.vae_encoder_optimizer.zero_grad()

                    inputs = data['input'].to(torch.float32).to(self.device)     # B x dim_x

                    # encode the robot contexts
                    context, means, logvars = self.vae_encoder.encode(inputs)
                    #    decode the robot context (used later)
                    decoded_context = self.vae_encoder.decode(context)


                    #    VAE losses
                    vae_recon_loss = self.vae_encoder.compute_vae_recon_loss(inputs, decoded_context)
                    vae_kl_loss = self.vae_encoder.compute_kld_loss(means, logvars)
                    #    TODO: do we need to add weights to VAE components? probably need to scale the VAE losses down...
                    #    VAE losses ~10e5, diff-loss ~10e1....
                    vae_enc_loss = vae_recon_loss + self.vae_kl_weight * vae_kl_loss
                    vae_dec_loss = vae_recon_loss

                    # update model parameters
                    #    do backwards passes
                    vae_dec_loss.backward(retain_graph=True)
                    vae_enc_loss.backward()

                    self.vae_encoder_optimizer.step()
                    self.vae_decoder_optimizer.step()

    
    def concat_basis_bias(self, basis):
        # single vs. batch processing
        if len(basis.shape) == 1:
            return torch.cat([basis, torch.ones(1).to(basis.device)])
        else:
            return torch.cat([basis, torch.ones(basis.shape[0], 1).to(basis.device)], dim=-1)

    def inference_denoise_loop(self, batch, context, model):
        # print("\tStarting inference_denoise_loop()....")
        latent = self.sample_noise(batch)

        for i, inf_timestep in enumerate(self.scheduler.inference_time_steps):
            alpha_t = self.scheduler.get_alpha(inf_timestep).to(self.device)
            alpha_bar_t = self.scheduler.get_alpha_bar(inf_timestep).to(self.device)
            beta_t = self.scheduler.get_beta(inf_timestep).to(self.device)

            # print("Denoising step {:d} - alpha: {:.4f}, alpha_bar: {:.4f}, beta: {:.4f}".format(i, alpha_t, alpha_bar_t, beta_t))

            latent = model.denoise_step(latent, context, inf_timestep, alpha_t, alpha_bar_t, beta_t)

        # print("\tReturning from inference_denoise_loop()")
        return latent

    def sample_noise(self, batch):
        noise = torch.randn((batch.shape[0], self.dim_a), dtype=batch.dtype, device=batch.device)
        return noise

    def create_noisy_latents(self, latents, noise, timesteps):
        noisy_latents = []
        for t in range(0, len(timesteps)):
            a_bar_t = self.scheduler.get_alpha(timesteps[t])
            noisy_latent = torch.sqrt(a_bar_t) * latents[t] + torch.sqrt(1-a_bar_t) * noise[t]
            noisy_latents.append(noisy_latent)

        return torch.stack(noisy_latents,dim=0)
    
    def train_model(self):
        step = 0
        training_averages = {"vae_recon":[], "vae_kl":[], "vae_encoder":[], "vae_decoder":[], "diff_discrim":[], 
                                        "diff_discrim_weighted":[], "diff":[], "diff_weighted":[],
                                        "diff_total":[], "discrim":[], "res_error":[], "res_error_w":[]}
        validation_averages = {"vae_recon":[], "vae_kl":[], "vae_encoder":[], "vae_decoder":[], "diff_discrim":[], 
                                "diff_discrim_weighted":[], "diff":[], "diff_weighted":[],
                                "diff_total":[], "diff_residual_loss":[]}
        # Iterate over the desired number of epochs
        for epoch in range(0, self.epochs):
            print("Starting Epoch: {:d}".format(epoch))

            epoch_average_losses_train = {"vae_recon":[], "vae_kl":[], "vae_encoder":[], "vae_decoder":[], "diff_discrim":[], 
                                        "diff_discrim_weighted":[], "diff":[], "diff_weighted":[],
                                        "diff_total":[], "discrim":[], "res_error":[], "res_error_w":[]}

            # randomly iterate over the condition-specific datasets self.num_batches
            for batch_idx in range(0, self.num_batches):
                # print("\tStarting Batch: {:d}".format(batch_idx))
                # Randomize the order in which we iterate over the condition-specific datasets
                arr = np.arange(self.num_train_classes)
                np.random.shuffle(arr)
                for i in arr:
                    kshot_data = None
                    data = None
                    with torch.no_grad():
                        adaptloader = self.Adaptloader[i]
                        kshot_data = next(iter(adaptloader))
                        trainloader = self.Trainloader[i]
                        data = next(iter(trainloader))

                    ###
                    #  Perform K-shot adaptation of mixing parameters M
                    ###
                    # print("\tAdapting M parameters....")
                    X_kshot = kshot_data['input'].to(torch.float32).to(self.device)    # K x dim_x
                    Y_kshot = kshot_data['output'].to(torch.float32).to(self.device)   # K x dim_y
                    
                    # encode the context with the VAE context decoder
                    #     push VAE encoder to GPU
                    self.vae_encoder.encoder.to(self.device)
                    #     set the VAE encoder to eval to not track gradients 
                    self.vae_encoder.eval()
                    #     all of these variables are on-device
                    context_kshot, _ , _ = self.vae_encoder.encode(X_kshot)
                    #     remove VAE from GPU
                    self.vae_encoder.cpu()

                    # perform a full inference loop for the diffusion model
                    #     push diffusion model to GPU
                    self.diff_model.to(self.device)
                    # self.diff_model.time_embedding.to(self.device)
                    # self.diff_model.unet.to(self.device)
                    self.diff_model.eval()
                    # self.diff_model.time_embedding.eval()
                    # self.diff_model.unet.eval()
                    latent = self.inference_denoise_loop(X_kshot, context_kshot, self.diff_model)
                    
                    # perform least-sqaures based adaptation
                    #    concatonate the bias to the predicted basis functions - currently not using a bias term
                    # latent = self.concat_basis_bias(latent)                        # K x dim_a
                    latent_T = latent.transpose(0,1)                                 # dim_a x K
                    M = torch.inverse(torch.mm(latent_T, latent))                    # dim_a x dim_a
                    M_star = torch.mm(torch.mm(M, latent_T), Y_kshot)                # dim_a x dim_y
                    #     normalize mixing coefficents 
                    if torch.norm(M_star, 'fro') > self.gamma:
                        M_star = M_star / torch.norm(M_star, 'fro') * self.gamma

                    # TODO: add updating matricies for PINN losses...

                    #    push data off of GPU
                    #    push diffusion model off of GPU
                    self.diff_model.cpu()
                    X_kshot.cpu()
                    Y_kshot.cpu()
                    M.cpu()
                    # print("\tDone Adapting M parameters")

                    ###
                    #  Perform least-sqaures approximation of ground-truth latents (Z^{T}) for diffusion training
                    ###
                    # print("\tApproximating Z^0 via least-sqaures....")
                    labels = data['output'].to(torch.float32).to(self.device)    # B x dim_y
                    M_star_T = M_star.transpose(0,1)                             # dim_y x dim_a
                    temp_1 = torch.mm(labels, M_star_T)                          # B x dim_a
                    temp_2 = torch.inverse(torch.mm(M_star, M_star_T))           # dim_a x dim_a
                    latents_T = torch.mm(temp_1, temp_2)                         # B x dim_a

                    # move temp values to cpu
                    temp_1.cpu()
                    temp_2.cpu()
                    # print("\tDone Approximating Z^0 via least-sqaures")

                    ###
                    #  Train VAE encoder and diffusion model
                    ###
                    # print("\tTraining VAE and Diffusion Model....")
                    #     set models to training mode
                    self.vae_encoder.train()
                    self.diff_model.train()
                    self.discriminator.train()
                    #     move models to GPU
                    self.vae_encoder.to(self.device)
                    self.diff_model.to(self.device)
                    self.discriminator.to(self.device)

                    #     zero the optimizers
                    self.vae_decoder_optimizer.zero_grad()
                    self.vae_encoder_optimizer.zero_grad()
                    self.diffusion_optimizer.zero_grad()

                    inputs = data['input'].to(torch.float32).to(self.device)     # B x dim_x

                    # encode the robot contexts
                    context, means, logvars = self.vae_encoder.encode(inputs)
                    #    decode the robot context (used later)
                    decoded_context = self.vae_encoder.decode(context) 

                    # perform diffusion training
                    #    push diffusion model to 
                    #    sample noise vector
                    noise = self.sample_noise(latents_T)
                    #    sample noise time-steps
                    ts = torch.randint(0, self.scheduler.num_training_steps, [inputs.shape[0]])
                    #    create the noisy latent values to be denoised
                    noisy_latents = self.create_noisy_latents(latents_T, noise, ts)
                    #    use diff model to predict noise
                    e_hat = self.diff_model(noisy_latents, context, ts)

                    # calculate losses
                    #    diffusion losses
                    diff_loss = self.diff_loss(e_hat, noise)
                    diff_loss = torch.mean(diff_loss, dim=1)
                    # print(diff_loss)
                    #    weight the losses via minSNR ratio
                    snr_weights = torch.ones((inputs.shape[0], 1), dtype=inputs.dtype, device=self.device)
                    for idx, t in enumerate(ts):
                        snr_weights[idx] = self.scheduler.get_minSNR_weight(t)

                    # print(snr_weights)
                    weighted_diff_loss = snr_weights * diff_loss

                    #    diffusion latent discriminator losses
                    #        next the denoised latents
                    new_latents = []
                    pinn_weights = []
                    for idx, t in enumerate(ts):
                        # pull out the scheduler params
                        alpha_bar_t = self.scheduler.get_alpha_bar(t)
                        # calcualte the completely denoised latent using the predicted e_hat 
                        new_latent = self.diff_model.get_pred_denoised_latent_no_forward(e_hat[idx], noisy_latents[idx], alpha_bar_t)
                        new_latents.append(new_latent)

                        # pull out the PINN loss weights (used in multiple places...)
                        # print(self.scheduler.get_pinn_weight(t))
                        pinn_weights.append(self.scheduler.get_pinn_weight(t))

                    new_latents = torch.stack(new_latents, dim=0)
                    pinn_weights = torch.stack(pinn_weights, dim=0).to(self.device)

                    # get the discriminator's predictions
                    c_labels = data['c'].type(torch.long).to(self.device)
                    discrim_preds = self.discriminator(new_latents)
                    diff_discrim_loss = self.discrim_loss(discrim_preds, c_labels)

                    #    apply the PINN weight to the discriminator losses
                    weighted_discrim_loss = pinn_weights * diff_discrim_loss

                    # Calculate Residual Targets
                    train_residuals = torch.mm(new_latents, M_star)
                    res_error = torch.mean(F.mse_loss(train_residuals, labels, reduction='none'), dim=1)
                    weighted_res_pred_errors = self.residual_weight * res_error * pinn_weights

                    # calculate the total doiffusion model loss (diffusion model loss and discriminator regularization)
                    #  
                    diff_loss_total = weighted_diff_loss + weighted_res_pred_errors - weighted_discrim_loss
                    diff_loss_total = torch.mean(diff_loss_total)

                    #    VAE losses
                    vae_recon_loss = self.vae_encoder.compute_vae_recon_loss(inputs, decoded_context)
                    vae_kl_loss = self.vae_kl_weight * self.vae_encoder.compute_kld_loss(means, logvars)
                    #    TODO: do we need to add weights to VAE components? probably need to scale the VAE losses down...
                    #    VAE losses ~10e5, diff-loss ~10e1....
                    vae_enc_loss = vae_recon_loss + vae_kl_loss + diff_loss_total
                    vae_dec_loss = vae_recon_loss

                    #    TODO: Add PINN losses...

                    # update model parameters
                    #    do backwards passes
                    vae_dec_loss.backward(retain_graph=True)
                    vae_enc_loss.backward(retain_graph=True)
                    diff_loss_total.backward()

                    self.vae_encoder_optimizer.step()
                    self.vae_decoder_optimizer.step()
                    self.diffusion_optimizer.step()

                    # print("\tDone Training VAE and diffusion model")                

                    ###
                    #   Train discriminator
                    ###
                    # print("\tTraining Discriminator Network....")
                    self.discriminator.train().to(self.device)
                    if np.random.rand() <= 1.0 / self.discrim_save_freq:
                        self.discriminator_optimizer.zero_grad()
                        noise = self.sample_noise(latents_T)
                        #    sample noise time-steps
                        ts = torch.randint(0, self.scheduler.num_training_steps, [inputs.shape[0]])
                        #    create the noisy latent values to be denoised
                        noisy_latents = self.create_noisy_latents(latents_T, noise, ts)
                        #    use diff model to predict noise
                        #        detach from comp-graph of diffusion model.. these are for discriminator
                        e_hat = self.diff_model(noisy_latents, context, ts).detach()

                        new_latents = []
                        # pinn_weights = []
                        for idx, t in enumerate(ts):
                            # pull out the scheduler params
                            alpha_bar_t = self.scheduler.get_alpha_bar(t)
                            # calcualte the completely denoised latent using the predicted e_hat 
                            new_latent = self.diff_model.get_pred_denoised_latent_no_forward(e_hat[idx], noisy_latents[idx], alpha_bar_t)
                            new_latents.append(new_latent)

                            # # pull out the PINN loss weights (used in multiple places...)
                            # pinn_weights[i] = self.scheduler.get_pinn_weight(t)
                        
                        new_latents = torch.stack(new_latents, dim=0)
                        # pinn_weights = torch.stack(pinn_weights, dim=0)
                        discrim_preds = self.discriminator(new_latents)
                        discrim_loss = self.discrim_loss(discrim_preds, c_labels)
                        discrim_loss = torch.mean(discrim_loss)

                        discrim_loss.backward()
                        self.discriminator_optimizer.step()
                        # print("\tDone Training Discriminator Network")
                    # end discriminator training statement


                    ###
                    #  Spectral normalization of VAE and diffusion model
                    ###
                    if self.sn > 0:
                        # print("\tSpectural Normalization....")
                        #    VAE
                        self.vae_encoder.cpu()
                        for param in self.vae_encoder.parameters():
                            W = param.detach().numpy()
                            if W.ndim > 1:
                                s = np.linalg.norm(W,2)
                                if s > self.sn:
                                    param.data = (param / s) * self.sn
                        
                        # diffusion model
                        self.diff_model.cpu()
                        for param in self.diff_model.parameters():
                            W = param.detach().numpy()
                            if W.ndim > 1:
                                s = np.linalg.norm(W,2)
                                if s > self.sn:
                                    param.data = (param / s) * self.sn
                        
                        # print("\tDone with Spectural Normalization")

                    
                    ###
                    #  Perform EMA (Temporal Averaging) Updates
                    ###
                    # print("\tPerforming EMA....")
                    self.vae_encoder_ema.to(self.device)
                    self.vae_encoder.to(self.device)
                    self.diff_model_ema.to(self.device)
                    self.diff_model.to(self.device)
                    self.discriminator_ema.to(self.device)
                    self.discriminator.to(self.device)
                    decay = 0 if step < self.ema_warm_up else self.ema_decay
                    #     VAE
                    self.ema_accumulate(self.vae_encoder_ema, self.vae_encoder, decay)
                    #     diffusion model
                    self.ema_accumulate(self.diff_model_ema, self.diff_model, decay)
                    #     discriminator
                    self.ema_accumulate(self.discriminator_ema, self.discriminator, decay)
                    # print("\tDone Performing EMA")

                    # TODO: Need to perform Spectral Normalization on EMA model??

                    ###
                    #  Log BATCH-Wise training results to logger
                    ###
                    # {"vae_recon":[], "vae_kl":[], "vae_encoder":[], "vae_decoder":[], "diff_discrim":[], 
                    # "diff_discrim_weighted":[], "diff":[], "diff_weighted":[],
                    # "diff_total":[], "discrim":[]}
                    # print("\tLogging Batch {:d} Results....".format(step))
                        
                    #     VAE Reconstruction loss
                    self.logger.add_scalar("train/step/vae_recon",
                                            vae_recon_loss.item(),
                                            step)
                    epoch_average_losses_train["vae_recon"].append(vae_recon_loss.item())
                    #     VAE KL loss
                    self.logger.add_scalar("train/step/vae_kl",
                                            vae_kl_loss.item(),
                                            step)
                    epoch_average_losses_train["vae_kl"].append(vae_kl_loss.item())
                    #     VAE encoder loss
                    self.logger.add_scalar("train/step/vae_encoder",
                                            vae_enc_loss.item(),
                                            step)
                    epoch_average_losses_train["vae_encoder"].append(vae_enc_loss.item())
                    #     VAE decoder loss
                    self.logger.add_scalar("train/step/vae_decoder",
                                            vae_dec_loss.item(),
                                            step)
                    epoch_average_losses_train["vae_decoder"].append(vae_enc_loss.item())
                    #     Diff discrim loss
                    diff_discrim_loss = torch.mean(diff_discrim_loss)
                    self.logger.add_scalar("train/step/diff_discrim",
                                            diff_discrim_loss.item(),
                                            step)
                    epoch_average_losses_train["diff_discrim"].append(diff_discrim_loss.item())
                    #     Weighted diff discrim loss
                    weighted_discrim_loss = torch.mean(weighted_discrim_loss)
                    self.logger.add_scalar("train/step/diff_discrim_weighted",
                                            weighted_discrim_loss.item(),
                                            step)
                    epoch_average_losses_train["diff_discrim_weighted"].append(weighted_discrim_loss.item())
                    #     diff loss
                    diff_loss = torch.mean(diff_loss)
                    self.logger.add_scalar("train/step/diff",
                                            diff_loss.item(),
                                            step)
                    epoch_average_losses_train["diff"].append(diff_loss.item())
                    #     weighted diff loss
                    weighted_diff_loss = torch.mean(weighted_diff_loss)
                    self.logger.add_scalar("train/step/diff_weighted",
                                            weighted_diff_loss.item(),
                                            step)
                    epoch_average_losses_train["diff_weighted"].append(weighted_diff_loss.item())
                    #     diff_total loss
                    diff_loss_total = torch.mean(diff_loss_total)
                    self.logger.add_scalar("train/step/diff_total",
                                            diff_loss_total.item(),
                                            step)
                    epoch_average_losses_train["diff_total"].append(diff_loss_total.item())
                    #     discrim loss
                    diff_loss_total = torch.mean(diff_loss_total)
                    self.logger.add_scalar("train/step/discrim",
                                            diff_loss_total.item(),
                                            step)
                    epoch_average_losses_train["discrim"].append(diff_loss_total.item())
                    #     residual error loss
                    res_error = torch.mean(res_error)
                    self.logger.add_scalar("train/step/res_error",
                                            res_error.item(),
                                            step)
                    epoch_average_losses_train["res_error"].append(res_error.item())
                    #
                    weighted_res_pred_errors = torch.mean(weighted_res_pred_errors)
                    self.logger.add_scalar("train/step/res_error_w",
                                            weighted_res_pred_errors.item(),
                                            step)
                    epoch_average_losses_train["res_error_w"].append(weighted_res_pred_errors.item())

                    

                    # print("\tDone Logging Batch Results....")


                    # increment our step counter for each processed batch
                    step += 1
                # end iteration over sub-datasets
            # end iteration over num_batches

            ###
            #  Calculate and log epoch-wise performance metrics
            ###
            # {"vae_recon":[], "vae_kl":[], "vae_encoder":[], "vae_decoder":[], "diff_discrim":[], 
            # "diff_discrim_weighted":[], "diff":[], "diff_weighted":[],
            # "diff_total":[], "discrim":[]}

            #     VAE Reconstruction loss
            self.logger.add_scalar("train_epoch/vae_recon",
                        np.mean(epoch_average_losses_train["vae_recon"]),
                        epoch)
            training_averages["vae_recon"].extend(epoch_average_losses_train["vae_recon"])
            epoch_average_losses_train["vae_recon"] = []
            #     VAE KL loss
            self.logger.add_scalar("train_epoch/vae_kl",
                        np.mean(epoch_average_losses_train["vae_kl"]),
                        epoch)
            training_averages["vae_kl"].extend(epoch_average_losses_train["vae_kl"])
            epoch_average_losses_train["vae_kl"] = []
            #     VAE encoder loss
            self.logger.add_scalar("train_epoch/vae_encoder",
                        np.mean(epoch_average_losses_train["vae_encoder"]),
                        epoch)
            training_averages["vae_encoder"].extend(epoch_average_losses_train["vae_encoder"])
            epoch_average_losses_train["vae_encoder"] = []
            #     VAE decoder loss
            self.logger.add_scalar("train_epoch/vae_decoder",
                        np.mean(epoch_average_losses_train["vae_decoder"]),
                        epoch)
            training_averages["vae_decoder"].extend(epoch_average_losses_train["vae_decoder"])
            epoch_average_losses_train["vae_decoder"] = []
            #     diff discriminator loss
            self.logger.add_scalar("train_epoch/diff_discrim",
                        np.mean(epoch_average_losses_train["diff_discrim"]),
                        epoch)
            training_averages["diff_discrim"].extend(epoch_average_losses_train["diff_discrim"])
            epoch_average_losses_train["diff_discrim"] = []
            #     diff_discrim_weighted loss
            self.logger.add_scalar("train_epoch/diff_discrim_weighted",
                        np.mean(epoch_average_losses_train["diff_discrim_weighted"]),
                        epoch)
            training_averages["diff_discrim_weighted"].extend(epoch_average_losses_train["diff_discrim_weighted"])
            epoch_average_losses_train["diff_discrim_weighted"] = []
            #     diff loss
            self.logger.add_scalar("train_epoch/diff",
                        np.mean(epoch_average_losses_train["diff"]),
                        epoch)
            training_averages["diff"].extend(epoch_average_losses_train["diff"])
            epoch_average_losses_train["diff"] = []
            #     diff_weighted loss
            self.logger.add_scalar("train_epoch/diff_weighted",
                        np.mean(epoch_average_losses_train["diff_weighted"]),
                        epoch)
            training_averages["diff_weighted"].extend(epoch_average_losses_train["diff_weighted"])
            epoch_average_losses_train["diff_weighted"] = []
            #     diff_total loss
            self.logger.add_scalar("train_epoch/diff_total",
                        np.mean(epoch_average_losses_train["diff_total"]),
                        epoch)
            training_averages["diff_total"].extend(epoch_average_losses_train["diff_total"])
            epoch_average_losses_train["diff_total"] = []
            #     discrim loss
            self.logger.add_scalar("train_epoch/discrim",
                        np.mean(epoch_average_losses_train["discrim"]),
                        epoch)
            training_averages["discrim"].extend(epoch_average_losses_train["discrim"])
            epoch_average_losses_train["discrim"] = []
            #     residual error loss
            self.logger.add_scalar("train_epoch/res_error",
                                    np.mean(epoch_average_losses_train["res_error"]),
                                    step)
            training_averages["res_error"].extend(epoch_average_losses_train["res_error"])
            epoch_average_losses_train["res_error"] = []
            #     weighted residual error loss
            self.logger.add_scalar("train_epoch/res_error_w",
                                    np.mean(epoch_average_losses_train["res_error_w"]),
                                    step)
            training_averages["res_error_w"].extend(epoch_average_losses_train["res_error_w"])
            epoch_average_losses_train["res_error_w"] = []
    
            ###
            #  Perform and log validation tests using EMA model
            ###
            # vae_recon_loss, vae_kl_loss, vae_enc_loss, vae_dec_loss, diff_loss, diff_residual_pred_loss
            # validation_averages = {"vae_recon":[], "vae_kl":[], "vae_encoder":[], "vae_decoder":[], "diff_discrim":[], 
            #                     "diff_discrim_weighted":[], "diff":[], "diff_weighted":[],
            #                     "diff_total":[], "diff_residual_loss":[]}
            val_vae_recon = []
            val_vae_kl = []
            # val_vae_enc = []
            # val_vae_dec = []
            val_diff = []
            # val_diff_w = []
            # val_diff_total = []
            # val_diff_dis = []
            # val_diff_dis_w = []
            val_diff_res = []
            with torch.no_grad():
                for j in range(len(self.TestData)):
                    input, output, condition = self.TestData[j].X, self.TestData[j].Y, self.TestData[j].C
                    val_vae_recon_loss, vae_kl_loss, val_diff_loss, diff_residual_pred_loss = self.validation(input, output, condition)
                    
                    val_vae_recon.append(val_vae_recon_loss.item())
                    val_vae_kl.append(vae_kl_loss.item())
                    # val_vae_dec.append(val_vae_enc_loss.item())
                    # val_vae_enc.append(val_vae_dec_loss.item())
                    val_diff.append(val_diff_loss.item())
                    # val_diff_w.append(val_weighted_diff_loss.item())
                    # val_diff_total.append(val_diff_loss_total.item())
                    # val_diff_dis.append(val_diff_discrim_loss.item())
                    # val_diff_dis_w.append(val_weighted_discrim_loss.item())
                    val_diff_res.append(diff_residual_pred_loss.item())
                    
                val_vae_recon = np.mean(val_vae_recon)
                val_vae_kl = np.mean(val_vae_kl)
                # val_vae_dec = np.mean(val_vae_dec)
                # val_vae_enc = np.mean(val_vae_enc)
                val_diff = np.mean(val_diff)
                # val_diff_w = np.mean(val_diff_w)
                # val_diff_total = np.mean(val_diff_total)
                # val_diff_dis = np.mean(val_diff_dis)
                # val_diff_dis_w = np.mean(val_diff_dis_w)
                val_diff_res = np.mean(val_diff_res)

                # validation_averages
                #    VAE recon loss
                self.logger.add_scalar("val/vae_recon",
                                        val_vae_recon,
                                        step)
                validation_averages["vae_recon"].append(val_vae_recon)
                #    VAE KL loss
                self.logger.add_scalar("val/vae_kl",
                                        val_vae_kl,
                                        step)
                validation_averages["vae_kl"].append(val_vae_kl)
                # #    val_vae_enc_loss loss
                # self.logger.add_scalar("val/vae_encoder",
                #                         val_vae_enc,
                #                         step)
                # validation_averages["vae_encoder"].append(val_vae_enc)
                # #    val_vae_dec_loss loss
                # self.logger.add_scalar("val/vae_decoder",
                #                         val_vae_dec,
                #                         step)
                # validation_averages["vae_decoder"].append(val_vae_dec)
                #    val_diff_loss loss
                self.logger.add_scalar("val/diff",
                                        val_diff,
                                        step)
                validation_averages["diff"].append(val_diff)
                #    val_weighted_diff_loss loss
                # self.logger.add_scalar("val/diff_weighted",
                #                         val_diff_w,
                #                         step)
                # validation_averages["diff_weighted"].append(val_diff_w)
                # #    val_diff_discrim_loss loss
                # self.logger.add_scalar("val/diff_discrim",
                #                         val_diff_dis,
                #                         step)
                # validation_averages["diff_discrim"].append(val_diff_dis)
                # #    val_weighted_discrim_loss loss
                # self.logger.add_scalar("val/diff_discrim_weighted",
                #                         val_diff_dis_w,
                #                         step)
                # validation_averages["diff_discrim_weighted"].append(val_diff_dis_w)
                # #    val_weighted_discrim_loss loss
                # self.logger.add_scalar("val/diff_total",
                #                         val_diff_total,
                #                         step)
                # validation_averages["diff_total"].append(val_diff_total)
                #    diff_residual_pred_loss loss
                self.logger.add_scalar("val/diff_residual_loss",
                                        val_diff_res,
                                        step)
                validation_averages["diff_residual_loss"].append(val_diff_res)

            
            # Update the tensorboard weight histograms
            self.histogram_adder(epoch)
            
            ###
            #  Save model
            ###
            if epoch % self.model_save_freq == 0:
                name = self.model_name + '-epoch-' + str(epoch)
                utils.save_models(name, self.vae_encoder, self.diff_model, self.discriminator, self.vae_encoder_ema, self.diff_model_ema, self.discriminator_ema, self.options)
                
        # end loop over epochs
        #    log overall training loss averages to tensorboard
        name = self.model_name + '-epoch-' + str(epoch)
        utils.save_models(name, self.vae_encoder, self.diff_model, self.discriminator, self.vae_encoder_ema, self.diff_model_ema, self.discriminator_ema, self.options)

    # Perform standard VAE -> Diffusion -> Discriminator forward pass on validation set
    #     also perform complete denosining process for validation samples and predict 
    #     resdiaul dynamics 
    def validation(self, input, labels, C):
        # Split condition-specific validation dataset into test and adapt batches
        fullset = utils.MyDataset(input, labels, C)
        l = len(input)
        trainset, adaptset = random_split(fullset, [int(2/3*l), l-int(2/3*l)])

        trainset = trainset[:]
        adaptset = adaptset[:]
          
        # Adapt M_star on adaptation batch
        # print("\tValidation - Adapting M parameters....")
        X_kshot = torch.from_numpy(adaptset['input']).to(torch.float32).to(self.device)     # K x dim_x
        Y_kshot = torch.from_numpy(adaptset['output']).to(torch.float32).to(self.device)    # K x dim_y
        
        # encode the context with the VAE context decoder
        #     push VAE encoder to GPU
        self.vae_encoder_ema.to(self.device)
        #     set the VAE encoder to eval to not track gradients 
        self.vae_encoder_ema.eval()
        #     all of these variables are on-device
        context_kshot, _ , _ = self.vae_encoder_ema.encode(X_kshot)
        #     remove VAE from GPU
        self.vae_encoder_ema.cpu()

        # perform a full inference loop for the diffusion model
        #     push diffusion model to GPU
        self.diff_model_ema.to(self.device)
        # self.diff_model.time_embedding.to(self.device)
        # self.diff_model.unet.to(self.device)
        self.diff_model_ema.eval()
        # self.diff_model.time_embedding.eval()
        # self.diff_model.unet.eval()
        latent = self.inference_denoise_loop(X_kshot, context_kshot, self.diff_model_ema)
        
        # perform least-sqaures based adaptation
        #    concatonate the bias to the predicted basis functions - currently not using a bias term
        # latent = self.concat_basis_bias(latent)                        # K x dim_a
        latent_T = latent.transpose(0,1)                                 # dim_a x K
        M = torch.inverse(torch.mm(latent_T, latent))                    # dim_a x dim_a
        M_star = torch.mm(torch.mm(M, latent_T), Y_kshot)                # dim_a x dim_y
        #     normalize mixing coefficents 
        if torch.norm(M_star, 'fro') > self.gamma:
            M_star = M_star / torch.norm(M_star, 'fro') * self.gamma

        #    push data off of GPU
        #    push diffusion model off of GPU
        self.diff_model.cpu()
        X_kshot.cpu()
        Y_kshot.cpu()
        M.cpu()
        # print("\tDone Adapting M parameters")

        # Calculate Z^T on test batch
        # print("\tValidation - Approximating Z^0 via least-sqaures....")
        labels = torch.from_numpy(trainset['output']).to(torch.float32).to(self.device)    # B x dim_y
        M_star_T = M_star.transpose(0,1)                             # dim_y x dim_a
        temp_1 = torch.mm(labels, M_star_T)                          # B x dim_a
        temp_2 = torch.inverse(torch.mm(M_star, M_star_T))           # dim_a x dim_a
        latents_T = torch.mm(temp_1, temp_2)                         # B x dim_a

        # move temp values to cpu
        temp_1.cpu()
        temp_2.cpu()
        # print("\tValidation - Done Approximating Z^0 via least-sqaures")

        # print("\tValidation - Testing VAE and Diffusion Model....")
        #     set models to training mode
        self.vae_encoder_ema.eval()
        self.diff_model_ema.eval()
        self.discriminator_ema.eval()
        #     move models to GPU
        self.vae_encoder_ema.to(self.device)
        self.diff_model_ema.to(self.device)
        self.discriminator_ema.to(self.device)

        inputs = torch.from_numpy(trainset['input']).to(torch.float32).to(self.device)     # B x dim_x
        
        # Encode / Decode context
        # encode the robot contexts
        context, means, logvars = self.vae_encoder_ema.encode(inputs)
        #    decode the robot context (used later)
        decoded_context = self.vae_encoder_ema.decode(context) 

        # perform diffusion training
        #    push diffusion model to 
        #    sample noise vector
        noise = self.sample_noise(latents_T)
        #    sample noise time-steps
        ts = torch.randint(0, self.scheduler.num_training_steps, [inputs.shape[0]])
        #    create the noisy latent values to be denoised
        noisy_latents = self.create_noisy_latents(latents_T, noise, ts)
        #    use diff model to predict noise
        e_hat = self.diff_model_ema(noisy_latents, context, ts)

        # calculate losses
        #    diffusion losses
        diff_loss = self.diff_loss(e_hat, noise)
        diff_loss = torch.mean(diff_loss, dim=1)

        # #    weight the losses via minSNR ratio
        # snr_weights = torch.ones((inputs.shape[0], 1), dtype=inputs.dtype, device=self.device)
        # for idx, t in enumerate(ts):
        #     snr_weights[idx] = self.scheduler.get_minSNR_weight(t)

        # # print(snr_weights)
        # weighted_diff_loss = snr_weights * diff_loss

        # #    diffusion latent discriminator losses
        # #        next the denoised latents
        # new_latents = []
        # pinn_weights = []
        # for idx, t in enumerate(ts):
        #     # pull out the scheduler params
        #     alpha_bar_t = self.scheduler.get_alpha_bar(t)
        #     # calcualte the completely denoised latent using the predicted e_hat 
        #     new_latent = self.diff_model_ema.get_pred_denoised_latent_no_forward(e_hat[idx], noisy_latents[idx], alpha_bar_t)
        #     new_latents.append(new_latent)

        #     # pull out the PINN loss weights (used in multiple places...)
        #     # print(self.scheduler.get_pinn_weight(t))
        #     pinn_weights.append(self.scheduler.get_pinn_weight(t))

        # new_latents = torch.stack(new_latents, dim=0)
        # pinn_weights = torch.stack(pinn_weights, dim=0).to(self.device)

        # # get the discriminator's predictions
        # # c_labels = torch.from_numpy(trainset['c']).type(torch.long).to(self.device)
        # c_labels = torch.zeros((new_latents.shape[0], self.num_test_classes))
        # for k in range(new_latents.shape[0]):
        #     c_labels[k][trainset['c']] = 1
        # c_labels = c_labels.type(torch.float32).to(self.device)
        # discrim_preds = self.discriminator_ema(new_latents)
        # diff_discrim_loss = self.discrim_loss(discrim_preds, c_labels)

        # #    apply the PINN weight to the discriminator losses
        # weighted_discrim_loss = pinn_weights * diff_discrim_loss

        # # calculate the total doiffusion model loss (diffusion model loss and discriminator regularization)
        # #  
        # diff_loss_total = weighted_diff_loss - weighted_discrim_loss
        # diff_loss_total = torch.mean(diff_loss_total)

        #    VAE losses
        vae_recon_loss = self.vae_encoder_ema.compute_vae_recon_loss(inputs, decoded_context)
        vae_kl_loss = self.vae_kl_weight * self.vae_encoder_ema.compute_kld_loss(means, logvars)
        #    TODO: do we need to add weights to VAE components? probably need to scale the VAE losses down...
        #    VAE losses ~10e5, diff-loss ~10e1....
        # vae_enc_loss = vae_recon_loss + vae_kl_loss + diff_loss_total
        # vae_dec_loss = vae_recon_loss

        diff_loss = torch.mean(diff_loss)
        # weighted_diff_loss = torch.mean(weighted_diff_loss)
        # diff_discrim_loss = torch.mean(diff_discrim_loss)
        # weighted_discrim_loss = torch.mean(weighted_discrim_loss)

        # perform inference denosing on test validation data
        #    calcualted residual dynamics using M_Star
        # batch, context, model
        denoised_latents = self.inference_denoise_loop(inputs, context, self.diff_model_ema)
        residual_preds = torch.mm(denoised_latents, M_star)

        diff_residual_pred_loss = F.mse_loss(residual_preds, labels)

        return vae_recon_loss, vae_kl_loss, diff_loss, diff_residual_pred_loss


