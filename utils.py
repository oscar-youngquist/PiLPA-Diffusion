import os, re
from typing import List, Dict
from ast import literal_eval
from collections import namedtuple
import torch 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vae_context_encoder_models import VAE_Conditioning_Model
from diffusion import Diffusion, Discriminator
import math 

folder = './data/experiment'
filename_fields = ['condition']

def save_models(name, vae_encoder, diff, discrim, vae_encoder_ema, diff_ema, discrim_ema, options):
    _output_path = os.path.join(options['output_path'], "model")
    if not os.path.isdir(_output_path):
        os.makedirs(_output_path)
    torch.save({
        'vae_encoder':vae_encoder.state_dict(),
        'diffusion':diff.state_dict(),
        'discrim':discrim.state_dict(),
        'vae_encoder_ema':vae_encoder_ema.state_dict(),
        'diffusion_ema':diff_ema.state_dict(),
        'discrim_ema':discrim_ema.state_dict(),
        'options':options
        }, os.path.join(_output_path, (name + '.pth')))
    
def load_models(model_path):
    model = torch.load(model_path)
    options = model['options']

    vae_encoder = VAE_Conditioning_Model(options=options)
    diff = Diffusion(options=options)
    discrim = Discriminator(options=options)

    vae_encoder_ema = VAE_Conditioning_Model(options=options)
    diff_ema = Diffusion(options=options)
    discrim_ema = Discriminator(options=options)

    vae_encoder.load_state_dict(model['vae_encoder'])
    diff.load_state_dict(model['diffusion'])
    discrim.load_state_dict(model['discrim'])

    vae_encoder_ema.load_state_dict(model['vae_encoder_ema'])
    diff_ema.load_state_dict(model['diffusion_ema'])
    discrim_ema.load_state_dict(model['discrim_ema'])

    vae_encoder.eval()
    diff.eval()
    discrim.eval()

    vae_encoder_ema.eval()
    diff_ema.eval()
    discrim_ema.eval()

    return vae_encoder, diff, discrim, vae_encoder_ema, diff_ema, discrim_ema

class MyDataset(Dataset):

    def __init__(self, inputs, outputs, c):
        self.inputs = inputs
        self.outputs = outputs
        self.c = c

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        Input = self.inputs[idx,]
        output = self.outputs[idx,]
        sample = {'input': Input, 'output': output, 'c': self.c}

        return sample

def load_data(folder : str, expnames = None) -> List[dict]:
    ''' Loads csv files from {folder} and return as list of dictionaries of ndarrays '''
    Data = []

    if expnames is None:
        filenames = os.listdir(folder)
        # print(filenames)
    elif isinstance(expnames, str): # if expnames is a string treat it as a regex expression
        filenames = []
        for filename in os.listdir(folder):
            if re.search(expnames, filename) is not None:
                filenames.append(filename)
    elif isinstance(expnames, list):
        filenames = (expname + '.csv' for expname in expnames)
    else:
        raise NotImplementedError()
    for filename in filenames:
        # Ingore not csv files, assume csv files are in the right format
        if not filename.endswith('.csv'):
            continue

        # Load the csv using a pandas.DataFrame
        df = pd.read_csv(folder + '/' + filename)
        
        # print(df.columns)

        # Lists are loaded as strings by default, convert them back to lists
        for field in df.columns[1:]:
            if isinstance(df[field][0], str):
                df[field] = df[field].apply(literal_eval)

        # Copy all the data to a dictionary, and make things np.ndarrays
        Data.append({})
        for field in df.columns[1:]:
            Data[-1][field] = np.array(df[field].tolist(), dtype=float)

        # Add in some metadata from the filename
        namesplit = filename.split('.')[0]
        for i, field in enumerate(filename_fields):
            Data[-1][field] = namesplit
        # Data[-1]['method'] = namesplit[0]
        # Data[-1]['condition'] = namesplit[1]

    return Data


SubDataset = namedtuple('SubDataset', 'X Y C meta')
feature_len = {}

def format_data(RawData: List[Dict['str', np.ndarray]], features: 'list[str]' = ['v', 'q', 'pwm'], output: str = 'fa', body_offset = 6):
    ''' Returns a list of SubDataset's collated from RawData.

        RawData: list of dictionaries with keys of type str. For keys corresponding to data fields, the value should be type np.ndarray. 
        features: fields to collate into the SubDataset.X element
        output: field to copy into the SubDataset.Y element
        hover_pwm_ratio: (average pwm at hover for testing data drone) / (average pwm at hover for training data drone)
         '''
    Data = []
    for i, data in enumerate(RawData):
        # Create input array
        X = []
        for feature in features:
            try:
                feature_len[feature] = len(data[feature][0])
                X.append(data[feature])
            except:
                feature_len[feature] = 1
                print(data[feature][:,np.newaxis].shape)
                X.append(data[feature][:,np.newaxis])

        X = np.hstack(X)

        # print(X.shape)

        # Create label array
        Y = []
        for _label in data[output]:
            Y.append(_label[body_offset:])
        
        Y = np.array(Y)

        # print(Y.shape)

        # create array of next-time step PINN loss data

        # Pseudo-label for cross-entropy
        C = i

        # print(data['condition'])

        # Save to dataset
        Data.append(SubDataset(X, Y, C, {'condition': data['condition'], 'steps': data['steps']}))

    return Data

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert roll, pitch, and yaw angles to a quaternion.

    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

    Returns:
        tuple: Quaternion (w, x, y, z)
    """

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr

    return [w, x, y, z]

def plot_subdataset(data, features, labels, output_path, title_prefix=''):
    fig, axs = plt.subplots(4, len(features)+1, figsize=(10,10))
    leg_labels = ["FR", "FL", "RR", "RL"]

    label_idx = 0
    row = 0
    idx = 0
    for col in range(0, len(features)):
        for j in range(0, feature_len[features[col]]):
            axs[row, col].plot(data.meta["steps"], data.X[:, idx], label = f"{features[col]}_{j}", alpha=0.7)
            idx += 1

            if idx % 3 == 0:
                axs[row,col].legend()
                axs[row,col].grid()

                if col == 0:
                    axs[row,col].set_ylabel(leg_labels[label_idx])
                    label_idx += 1
                # axs[row,col].set_xlabel('Control-Steps')
                row += 1

        
        # reset the row counter for each feature
        row = 0

    axis_range = [-50, 30]
    row = 0
    idx = 0
    for j , label in enumerate(labels):
        axs[row, -1].plot(data.meta["steps"], data.Y[:, idx], label = label, alpha=0.7)
        idx += 1
        if idx % 3 == 0:
            axs[row,-1].legend()
            axs[row,-1].set_ylim(axis_range)
            axs[row,-1].grid()
            # axs[row,col].set_xlabel('Control-Steps')
            row += 1

    # for feature, ax in zip(features, axs):
    #     for j in range(feature_len[feature]):
    #         ax.plot(data.meta['steps'], data.X[:, idx], label = f"{feature}_{j}")
    #         idx += 1
    #     ax.legend()
    #     ax.set_xlabel('Control-Steps')
    # ax = axs[-1]
    # ax.plot(data.meta['steps'], data.Y)
    # ax.legend(labels)
    # ax.set_xlabel('Control-Steps')
    fig.suptitle(f"{title_prefix} {data.meta['condition']}: c={data.C}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)