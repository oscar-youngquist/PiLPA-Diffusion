import numpy as np
import torch
import pinocchio
import torch.nn.functional as F
import math
import utils


class WholeBody_PINN_Loss():
    def __init__(self, options) -> None:
        self.options = options
        self.mixing_torso_pos = None
        self.mixing_q_pos = None
        self.mixing_torso_velo = None
        self.device = options['device']
        self.gamma = options['gamma']

        self.model = pinocchio.buildModelFromUrdf(options["urdf_path"], pinocchio.JointModelFreeFlyer())
        print("model name: " + self.model.name)

        # Create data required by the algorithms
        self.data = self.model.createData()

        self.leg_joint_limits = [[-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                                 [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                                 [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721],
                                 [-1.047, 1.047], [-0.663, 2.966], [-0.837, -2.721]]

    
    def update_mixing_parameters(self, latents, position_torso, velocity_torso, q_state):
        # position_torso --> p_x, p_y, p_z, roll, pitch, yaw, (all for the next time-step)
        # velocity_torso --> p_velo (3), \Theta_velo (3) (all for next time-step)
        # q-state        --> leg joint positions (12)
        # print("WholeBody_PINN_Loss.update_mixing_parameters() -- Updating PINN-Loss mixing parameters!")
        # used for all calculations
        latent_T = latents.transpose(0,1)                                        # dim_a x K
        M = torch.inverse(torch.mm(latent_T, latents))                           # dim_a x dim_a

        # extract just the height, roll, pitch, and yaw
        position_torso = position_torso[:,[2,3,4,5]]
        
        # Update matrix for torso position (height, roll, pitch, yaw) 
        self.mixing_torso_pos = torch.mm(torch.mm(M, latent_T), position_torso)  # dim_a x 4
        #     normalize mixing coefficents 
        if torch.norm( self.mixing_torso_pos, 'fro') > self.gamma:
             self.mixing_torso_pos =  self.mixing_torso_pos / torch.norm( self.mixing_torso_pos, 'fro') * self.gamma

        # Update matrix for q_state
        self.mixing_q_pos = torch.mm(torch.mm(M, latent_T), q_state)             # dim_a x 12
        #     normalize mixing coefficents 
        if torch.norm( self.mixing_q_pos, 'fro') > self.gamma:
             self.mixing_q_pos =  self.mixing_q_pos / torch.norm( self.mixing_q_pos, 'fro') * self.gamma

        # Update matrix for torso velocity (z, y, z, roll, pitch, yaw)
        self.mixing_torso_velo = torch.mm(torch.mm(M, latent_T), velocity_torso) # dim_a x 6
        #     normalize mixing coefficents 
        if torch.norm( self.mixing_torso_velo, 'fro') > self.gamma:
             self.mixing_torso_velo =  self.mixing_torso_velo / torch.norm( self.mixing_torso_velo, 'fro') * self.gamma


    def clamp_q_preds(self, q_preds): 
        for i in range(0, len(self.leg_joint_limits)):
            torch.clamp(q_preds[:,i], min=self.leg_joint_limits[i][0], max=self.leg_joint_limits[i][1])
        return q_preds
    
    
    #     NOTE this does not reduce the loss batch-wise
    def calculate_pinn_loss(self, inputs, latents, residuals, position_torso, velocity_torso, ex_tau, ex_fr_tau):
        wb_pinn_loss = None
        
        if self.mixing_torso_pos == None or self.mixing_q_pos == None or self.mixing_torso_velo == None:
            print("WholeBody_PINN_Loss().calculate_pinn_loss() - Error, tried calculating PINN loss without valid mixing parameters")
            exit()

        # position_torso --> p_x, p_y, p_z, roll, pitch, yaw, (all for the next time-step)
        # velocity_torso --> p_velo (3), \Theta_velo (3) (all for next time-step)

        wb_torso_pos = torch.mm(latents, self.mixing_torso_pos)  # p_z, roll, pitch, yaw
        wb_q_pos = torch.mm(latents, self.mixing_q_pos)          # q-state (12-D)
        wb_velo = torch.mm(latents, self.mixing_torso_velo)      # \dot{p} (3-D), \dot{\Theta} (3-D)

        # clamp the leg joint position predictions to the joint limits (help clean up the gradients)
        wb_q_pos = self.clamp_q_preds(wb_q_pos)

        # print("inputs")
        # print(inputs.shape)
        # print(inputs[0])

        # print("wb_q_pos")
        # print(wb_q_pos.shape)
        # print(wb_q_pos[0])

        # iterate over the q-poses and get the gradient w.r.t. time via auto-diff to approximate velocity and accelerations then concat
        q_velo_ad = []
        q_acc_ad = []
        for i in range(0, wb_q_pos.shape[1]):
            _q_velo_ad = torch.autograd.grad(wb_q_pos[:,i], inputs, create_graph=True, grad_outputs=torch.ones_like(wb_q_pos[:,i]))[0]
            _q_acc_ad = torch.autograd.grad(_q_velo_ad, inputs, retain_graph=True, grad_outputs=torch.ones_like(_q_velo_ad))[0]
            q_velo_ad.append(_q_velo_ad)
            q_acc_ad.append(_q_acc_ad)
        q_velo_ad = torch.cat(q_velo_ad, dim=1)
        q_acc_ad = torch.cat(q_acc_ad, dim=1)

        # print("q_velo_ad")
        # print(q_velo_ad.shape)
        # print(q_velo_ad[0])

        # print("q_acc_ad")
        # print(q_acc_ad.shape)
        # print(q_acc_ad[0])

        # print("wb_velo")
        # print(wb_velo.shape)
        # print(wb_velo[0])

        # calculate the torso accelerations of the torso using auto-diff
        torso_acc_ad = []
        for i in range(0, wb_velo.shape[1]):
            _torso_acc_ad = torch.autograd.grad(wb_velo[:,i], inputs, grad_outputs=torch.ones_like(wb_velo[:,i]), retain_graph=True)[0]
            torso_acc_ad.append(_torso_acc_ad)
        torso_acc_ad = torch.cat(torso_acc_ad, dim=1)

        # print("torso_acc_ad")
        # print(torso_acc_ad.shape)
        # print(torso_acc_ad[0])

        # Create numpy version of PyTorch vectors to calculate M and b via pinocchio library
        wb_pos_np = wb_torso_pos.detach().cpu().numpy()
        wb_velo_np = wb_velo.detach().cpu().numpy()
        wb_q_pos_np = wb_q_pos.detach().cpu().numpy()
        q_velo_ad_np = q_velo_ad.detach().cpu().numpy()

        position_target_np = position_torso.detach().cpu().numpy()

        Ws = []
        bs = []
        for i in range(0, inputs.shape[0]):
            # create necessary state-vectors for pinocchio
            #     create the 19x1 wb-state vector [3D-torso position, 4D torso quaternion, 12-D joint positions]
            body_x_y_z = np.array([position_target_np[i][0], position_target_np[i][1], wb_pos_np[i][0]])
            quat_body_ori = np.array(utils.euler_to_quaternion(wb_pos_np[i][1], wb_pos_np[i][2], wb_pos_np[i][3]))
            wb_state_np = np.concatenate((body_x_y_z, quat_body_ori, wb_q_pos_np[i]))

            #     create the 18x1 wb-velocity vectory [3D torso linear velocity, 3d torso angular veloctiy, 12-D joint velocities]
            wb_velo_state_np = np.concatenate((wb_velo_np[i], q_velo_ad_np[i]))

            #     calculate A and b+g
            #  matrix using pinocchio
            aq0 = np.zeros(self.model.nv)
            #     compute dynamic drift -- Coriolis, centrifugal, gravity
            b = pinocchio.rnea(self.model, self.data, wb_state_np, wb_velo_state_np, aq0)   # batch_size x 18
            #     compute mass matrix A
            A = pinocchio.crba(self.model, self.data, wb_state_np)

            # add W and b to lists
            Ws.append(torch.Tensor(A).to(self.device))
            bs.append(torch.Tensor(b).to(self.device))

        # end iteration over batch

        # stack the create dynamics tensors... batch_size x 18 x 18 and batch_size x 18
        M_torch = torch.stack(Ws, dim=0)
        b_torch = torch.stack(bs, dim=0)

        # print("M_torch")
        # print(M_torch.shape)
        # print("b_torch")
        # print(b_torch.shape)

        # print("torso_acc_ad.shape: ", torso_acc_ad.shape)
        # print("q_acc_ad.shape: ", q_acc_ad.shape)

        # build necessary acceleration vector for WB loss calculation
        wb_Q = torch.cat([torso_acc_ad, q_acc_ad], dim=1)    # batch_size x 18

        # print("wb_Q")
        # print(wb_Q.shape)
        # print(wb_Q[0])

        # calculate wb-PINN loss
        #     use the full-body dynamics to calculate physics-informed residauls
        #     M * Q + b - ex_tau - ex_fr_tau - residuals = 0.... (might not be true for the torso....)
        pinn_dynamics = torch.bmm(M_torch, wb_Q.unsqueeze(2)).squeeze() + b_torch

        # used during debugging, remove later...
        zeros_1 = torch.zeros((ex_tau.shape[0], 6), device=self.device)
        zeros_2 = torch.zeros((ex_fr_tau.shape[0], 6), device=self.device)
        ex_tau = torch.cat([zeros_1, ex_tau], dim=1)
        ex_fr_tau = torch.cat([zeros_2, ex_fr_tau], dim=1)
        #     
        zeros = torch.zeros((residuals.shape[0], 6), device=self.device)
        residuals = torch.cat([zeros, residuals], dim=1)

        wb_res = pinn_dynamics - ex_tau - ex_fr_tau - residuals
        #     create a target vector filled with zeros
        target = torch.zeros_like(wb_res, dtype=latents.dtype, device=self.device)
        #     use PyTorch functional to calculate MSE loss 
        wb_pinn_loss = F.mse_loss(wb_res, target, reduction='none')
        wb_pinn_loss = torch.mean(wb_pinn_loss, dim=1)

        #    calculate the loss of the model predicting the next time-step torso velocities
        wb_toros_velo_loss = F.mse_loss(wb_velo, velocity_torso, reduction='none')
        wb_toros_velo_loss = torch.mean(wb_toros_velo_loss, dim=1)

        # print("wb_pinn_loss")
        # print(wb_pinn_loss.shape)
        # print(wb_pinn_loss[0])

        # print("wb_toros_velo_loss")
        # print(wb_toros_velo_loss.shape)
        # print(wb_toros_velo_loss[0])

        return wb_pinn_loss, wb_toros_velo_loss