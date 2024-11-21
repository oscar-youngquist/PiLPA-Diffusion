import torch
from torch import nn
from torch.nn import functional as F


# Cross-entropy loss
class VAE_Discriminator(nn.Module):
    def __init__(self, options):
        super(VAE_Discriminator, self).__init__()
        self.fc1 = nn.Linear(options['context_size'], options['vae_discrim_h1'])
        self.output = nn.Linear(options['vae_discrim_h1'], options['num_c'])

        # init the weights using xavier-intialization
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.output.weight)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.silu(x)
        x = self.output(x)
        return x


# Classes used to encode current robot state and velocity commands into a conditioning value for conditional diffusion
class VAE_Context_Encoder(nn.Module):
    def __init__(self, options):
        super(VAE_Context_Encoder, self).__init__()
        self.options = options

        # network is structured via:
        #  input-lyaer -> hidden-1 -> hidden-2 -> latent_dim (mu, sigma)
        self.input_layer = nn.Linear(options["dim_x"], options["enc_hidden_1_in"])
        self.hidden_layer_1 = nn.Linear(options["enc_hidden_1_in"], options["enc_hidden_2_in"])
        self.hidden_layer_2 = nn.Linear(options["enc_hidden_2_in"], options["enc_hidden_2_out"])

        # dropout layers
        self.dropout_in = nn.Dropout(options["dropout"])
        self.dropout_h1 = nn.Dropout(options["dropout"])
        self.dropout_h2 = nn.Dropout(options["dropout"])
        self.dropout_reparam = nn.Dropout(options["dropout"])

        self.mu_layer = nn.Linear(options["enc_hidden_2_out"], options["context_size"])
        self.sigma_layer = nn.Linear(options["enc_hidden_2_out"], options["context_size"])

        # leayer normalization before the mu and sigma layers
        self.layer_norm = nn.LayerNorm(options["enc_hidden_2_out"])

        # inout layer norm layer
        self.input_norm = nn.LayerNorm(options['dim_x'])

        # init the weights using xavier-intialization
        torch.nn.init.xavier_uniform_(self.input_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer_1.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer_2.weight)
        torch.nn.init.xavier_uniform_(self.mu_layer.weight)
        torch.nn.init.xavier_uniform_(self.sigma_layer.weight)

    
    def _reparam_trick(self, x):
        x = self.dropout_reparam(x)
        mean = self.mu_layer(x)
        logvar = self.sigma_layer(x)

        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8.
        logvar = torch.clamp(logvar, -30, 20)
        variance = logvar.exp()
        stdev = variance.sqrt()

        # sample noise
        epsilon = torch.randn(stdev.shape, device=stdev.device)

        # calculate the "sample" from the latent distribution
        x = mean + stdev * epsilon

        return x, mean, logvar
    
    def forward(self, x):
        x = self.input_norm(x)
        #(Batch_Size, options["dim_x"]) -> (Batch_Size, options["enc_hidden_1_in"])
        x = self.input_layer(x)
        x = self.dropout_in(x)
        x = F.silu(x)
        #(Batch_Size, options["enc_hidden_1_in"]) -> (Batch_Size, options["enc_hidden_2_in"])
        x = self.hidden_layer_1(x)
        x = self.dropout_h1(x)
        x = F.silu(x)
        #(Batch_Size, options["enc_hidden_2_in"]) -> (Batch_Size, options["enc_hidden_2_out"])
        x = self.hidden_layer_2(x)
        x = self.dropout_h2(x)
        x = F.silu(x)
        # Normalization - no size change
        x = self.layer_norm(x)
        #(Batch_Size, options["enc_hidden_2_out"]) -> (Batch_Size, options["dim_a"])
        x, mean, logvar = self._reparam_trick(x)
        return x, mean, logvar


class VAE_Context_Decoder(nn.Module):
    def __init__(self, options):
        super(VAE_Context_Decoder, self).__init__()
        self.options = options

        # network is structured via:
        #  input-layer -> hidden-1 -> hidden-2 -> latent_dim (mu, sigma)
        self.input_layer = nn.Linear(options["context_size"], options["dec_hidden_1_in"])
        self.hidden_layer_1 = nn.Linear(options["dec_hidden_1_in"], options["dec_hidden_2_in"])
        self.hidden_layer_2 = nn.Linear(options["dec_hidden_2_in"], options["dec_hidden_2_out"])
        self.decoded_layer = nn.Linear(options["dec_hidden_2_out"], options["dim_x"])
        
        # dropout layers
        self.dropout_in = nn.Dropout(options["dropout"])
        self.dropout_h1 = nn.Dropout(options["dropout"])
        self.dropout_h2 = nn.Dropout(options["dropout"])
        
        # layer normalization before the decoding layer
        self.layer_norm = nn.LayerNorm(options["dec_hidden_2_out"])

        # init the weights using xavier-intialization
        torch.nn.init.xavier_uniform_(self.input_layer.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer_1.weight)
        torch.nn.init.xavier_uniform_(self.hidden_layer_2.weight)
        torch.nn.init.xavier_uniform_(self.decoded_layer.weight)
    
    def forward(self, x):
        #(Batch_Size, options["dim_a"]) -> (Batch_Size, options["dec_hidden_1_in"])
        x = self.input_layer(x)
        x = self.dropout_in(x)
        x = F.silu(x)
        #(Batch_Size, options["dec_hidden_1_in"]) -> (Batch_Size, options["dec_hidden_2_in"])
        x = self.hidden_layer_1(x)
        x = self.dropout_h1(x)
        x = F.silu(x)
        #(Batch_Size, options["dec_hidden_2_in"]) -> (Batch_Size, options["dec_hidden_2_out"])
        x = self.hidden_layer_2(x)
        x = self.dropout_h2(x)
        x = F.silu(x)
        # Normalization - no size change
        x = self.layer_norm(x)
        #(Batch_Size, options["dec_hidden_2_out"]) -> (Batch_Size, options["dim_x"])
        x = self.decoded_layer(x)
        return x

# model used to predict the next-time step's quadruped robot body state from the
#     VAE latent state. Encourages encoding dynamics indo the conitioning
#     latent space.
# class VAE_Context_Next_Step_Predictor(nn.Module):


# Container class for VAE robot context encoder and decoder, encapsulates calculating the losses
class VAE_Conditioning_Model(nn.Module):
    def __init__(self, options):
        super(VAE_Conditioning_Model, self).__init__()
        self.options = options

        self.encoder = VAE_Context_Encoder(options)
        self.decoder = VAE_Context_Decoder(options)

    def encode(self, x):
        z, mean, logvar = self.encoder(x)
        return z, mean, logvar

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat
    
    def forward(self, x):
        z, mean, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, mean, logvar
    
    # adding a mean to scale both of these to be in roughly the same order of magnitude as the diffusion loss
    def compute_kld_loss(self, mean, logvar):
        kld = torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()))
        return kld
    
    def compute_vae_recon_loss(self, x, x_hat):
        recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        return recon_loss