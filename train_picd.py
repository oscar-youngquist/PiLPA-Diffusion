import numpy as np
import pandas as pd
import os
import argparse
from datetime import datetime
import re
import argparse
import utils
from picd_trainer import Train_PICD

def run_training_loop(options):

    # create a few output paths....
    config_output_path                = os.path.join(options["output_path"], "config.json")

    # this will train a model for some number of epochs...
    model_trainer = Train_PICD(options)

    # save out the config path
    model_trainer.save_config(config_output_path)

    # train the model
    model_trainer.train_model()
    
    # scripted_model_name = 'model_cmd_res_cc_a{:d}_{:d}_{:d}_h{:d}_e{:d}.pt'.format(options['dim_a'],options['phi_first_out'],options['phi_second_out'],options['discrim_hidden'],options['num_epochs'])
    # scripted_model_path = os.path.join(options['output_path'], scripted_model_name)
    
    # model_trainer.save_scripted_model(scripted_model_path)


def build_output_path(options):
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y%H:%M:%S")
    date_time = re.sub('[^0-9a-zA-Z]+', '_', date_time)

    date_time += "_{:s}".format(options["exp_name"])

    cwd = os.getcwd()

    output_path_base = os.path.join(cwd, "training_results", options["output_prefix"], date_time)

    if not os.path.exists(output_path_base):
        os.makedirs(output_path_base)

    tensorboard_folder = os.path.join(cwd, "training_results", options["output_prefix"], date_time, "tensorboard")

    if not os.path.exists(tensorboard_folder):
        os.makedirs(tensorboard_folder)

    options['output_path'] = output_path_base
    options['tensorboard_folder'] = tensorboard_folder


# Index(['body_pos', 'body_ori', 'body_velo', 'body_ang_velo', 'body_acc',
#        'body_ang_acc', 'q', 'q_dot', 'q_ddot_est', 'q_ddot_m', 'tau',
#        'tau_cmd', 'fr_cmd', 'contact_m', 'fr_contact', 'tau_residual_m',
#        'tau_residual_full', 'tau_residual_cmd_centered', 'body_rpy', 'body_rp',
#        'body_rp_dot', 'steps', 'tau_residual_cmd'],
#       dtype='object')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PiCD")
    
    parser.add_argument('--train-path', type=str, 
                        default='/home/oyoungquist/Research/PiLPA/PiLPA-Diffusion/data/training_data_cse_small/', 
                        help='Path to training data')
    parser.add_argument('--test-path', type=str, 
                        default='/home/oyoungquist/Research/PiLPA/PiLPA-Diffusion/data/eval_data_cse/', 
                        help='Path to eval data')

    # parser.add_argument('--train-path', type=str, 
    #                     default='/work/pi_hzhang2_umass_edu/oyoungquist_umass_edu/RINA/rina/data/06_24_2024_formal/training_data_corrected/', 
    #                     help='Path to training data')
    # parser.add_argument('--test-path', type=str, 
    #                     default='/work/pi_hzhang2_umass_edu/oyoungquist_umass_edu/RINA/rina/data/06_24_2024_formal/eval_data_corrected/', 
    #                     help='Path to training data')

    # old carried over RINA params
    parser.add_argument('--num-epochs', type=int, default=10000, help='Number of epochs to train (default: 10000)')
    parser.add_argument('--learning-rate', type=float, default=0.00005, help='Learning rate (default: 0.00005)')
    parser.add_argument('--learning-rate-dis', type=float, default=0.00005, help='Learning rate (default: 0.00005)')
    parser.add_argument('--model-save-freq', type=int, default=100, help='Number of epochs between model saves (default: 100)')
    parser.add_argument('--SN', type=float, default=6.0, help='Max single-layer spectural norm (default: 6.0)')
    parser.add_argument('--gamma', type=float, default=10, help='Max magnitude of a (default: 10.0)')
    parser.add_argument('--frequency-h', type=float, default=2.0, help='Phi/Dis. update ratio (default: 2.0)')
    parser.add_argument('--K-shot', type=int, default=32, help='Hidden layer size for discriminator (default: 64)')
    parser.add_argument('--phi-shot', type=int, default=256, help='Training batch-size (default: 64)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Training device (default: cuda:0)')

    # binary decisions
    parser.add_argument('--shuffle', action="store_true", help='Shuffle training data (default: True)')
    parser.add_argument('--no-shuffle', dest="shuffle", action="store_false")
    parser.add_argument('--save-data-plots', action='store_true', help='Save out plots highlighting training data (default: True)')
    parser.add_argument('--no-save-data-plots', dest='save_data_plots', action='store_false')
    parser.add_argument('--display-progress', action="store_true", help='Print out progress (default: True)')
    parser.add_argument('--no-display-progress', dest="display_progress", action="store_false")
    parser.set_defaults(shuffle=True)
    parser.set_defaults(save_data_plots=True)
    parser.set_defaults(display_progress=True)

    # Diffusion / Model Parameters
    #     VAE encoder/decoder
    parser.add_argument('--context-size', type=int, default=10, help='Size of VAE robot-state context vector (default: 10)')
    parser.add_argument('--enc-hidden-1-in', type=int, default=52, help='Input size of first hidden layer for VAE encoder (default: 52)')
    parser.add_argument('--enc-hidden-2-in', type=int, default=36, help='Output size of first hidden layer for VAE encoder (default: 36)')
    parser.add_argument('--enc-hidden-2-out', type=int, default=18, help='Output size of second hidden layer for VAE encoder (default: 18)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability (default: 0.1 - 10\%)')
    parser.add_argument('--dec-hidden-1-in', type=int, default=18, help='Input size of first hidden layer for VAE decoder (default: 18)')
    parser.add_argument('--dec-hidden-2-in', type=int, default=36, help='Output size of first hidden layer for VAE decoder (default: 36)')
    parser.add_argument('--dec-hidden-2-out', type=int, default=52, help='Output size of second hidden layer for VAE decoder (default: 52)')

    #    Diffusion model parameters
    parser.add_argument('--time-size', type=int, default=10, help='Size of the timestep embedding (default: 10)')
    parser.add_argument('--diff-out-1', type=int, default=10, help='First hidden layer size for diffusion UNet (default: 10)')
    parser.add_argument('--diff-out-2', type=int, default=8, help='Second hidden layer size for  diffusion UNet (default: 8)')
    parser.add_argument('--n-heads', type=int, default=2, help='Number of attention head in diff. UNet context conditioning layers (default: 2)')

    #    Diffusion scheduler parameters
    parser.add_argument('--num-training-steps', type=int, default=50, help='Number of training steps for the diffusion model (default: 50)')
    parser.add_argument('--num-inference-steps', type=int, default=5, help='Number of inference steps for the diffusion model (default: 5)')
    parser.add_argument('--schedule-type', type=str, default='cosine', help='Type of beta schedule for diffusion model (default: \'cosine\')')
    # Params "beta_start" and "beta_end" taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L5C8-L5C8
    parser.add_argument('--beta-start', type=float, default=0.00085, help='Beta starting value (default: 0.00085)')
    parser.add_argument('--beta-end', type=float, default=0.0120, help='Beta end value (default: 0.0120)')
    # values taken from https://arxiv.org/pdf/2303.09556.pdf
    parser.add_argument('--snr-min-value', type=float, default=5.0, help='Maximum SNR weight value (default: 5.0)')
    # PINN loss fixed noise covaraiance scale factor https://arxiv.org/pdf/2403.14404
    parser.add_argument('--covar-scale', type=float, default=0.01, help='PINN loss weigth scale factor (default: 0.01)')


    #    training params
    parser.add_argument('--model-name', type=str, help='Name used to create experiment folder (default: None)')
    parser.add_argument('--exp-name', type=str, help='Name used to create experiment folder (default: None)')
    parser.add_argument('--vae-warm-up', type=int, default=0, help='Number of epochs to train the VAE before diffusion training (default: 0)')
    parser.add_argument('--ema-warm-up', type=int, default=10, help='Number of warm up EMA training (default: 10)')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='Decay rate for EMA training (default: 0.9999)')


    # ['body_rp','q','body_rp_dot','q_dot','fr_contact','tau_cmd']
    parser.add_argument('--features', nargs="+", type=str, 
                        default=['body_rpy', 'q', 'body_velo', 'body_ang_velo', 'q_dot','tau_cmd'], 
                        help='Values used an input data (default: [body_rpy,q,body_velo,body_ang_velo,q_dot,tau_cmd])')
    parser.add_argument('--label', type=str, default='tau_residual_cmd_cs', help='Name of training lable (target)')
    parser.add_argument('--dim-a', type=int, default=12, help='Number of basis-functions (defaut: 12)')
    parser.add_argument('--output-prefix', type=str, default='', help='Prefix for output folder (default: '')')


    args = parser.parse_args()

    options = {}

    # # fill in the default values...
    options["dim_a"]           = 12
    options["features"]        = ['body_rpy', 'q', 'body_velo', 'body_ang_velo', 'q_dot','tau_cmd']
    options["label"]           = 'tau_residual_cmd_cs'
    options["labels"]          = ["FR_hip", "FR_knee", "FR_foot", "FL_hip", "FL_knee", "FL_foot",
                                  "RR_hip", "RR_knee", "RR_foot", "RL_hip", "RL_knee", "RL_foot"]
    options["dataset"]         = 'rina'
    options["learning_rate"]   = 5e-4
    options["model_save_freq"] = 5
    options['num_epochs']      = 10
    options["gamma"]           = 10    # max 2-norm of M
    options['frequency_h']     = 2.    # discriminator update frequency
    options['SN']              = 4     # maximum single-layer spectral norm
    options['shuffle']         = True
    options["body_offset"]     = 0
    options['K_shot']          = 32 # number of K-shot for least square on a
    options['phi_shot']        = 256 # batch size for training phi
    options['loss_type']       = 'crossentropy-loss'
    options['display_progress'] = False

    # find keys we need to copy out of args
    args_dict = vars(args)
    config_keys = args_dict.keys()

    print(args_dict)
    print("\n\n")
    print(options)

    # print(config_keys)

    for key in config_keys:
        options[key] = args_dict[key]

    print(options)

    # build the output path
    build_output_path(options)

    # print(options.keys())

    # print(options["output_path"])

    # print(options['features'])

    run_training_loop(options)


# body_rp q body_rp_dot q_dot fr_contact tau_cmd


# python3 train_rina.py --output-prefix cmd_residual_cs_update/corrected/extended_state --num-epochs 15000 --label tau_residual_cmd_cs --discrim-hidden 40 --phi-first-out 100 --phi-second-out 128  --device cuda:0 --phi-shot 2048 --K-shot 1024 --features body_rpy q body_velo body_ang_velo q_dot tau_cmd --no-save-data-plots --learning-rate 0.00069 --alpha 0.1 --dim-a 16 --SN 6 --gamma 10

# python3 train_picd.py --output-prefix debugging_implementation --exp-name debug_tests 