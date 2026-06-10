#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
from ruamel.yaml import YAML
import pytorch_kinematics as pk

import sys
import os
project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(project_path)
# from plant.hydraulic_model import HydraulicActuatorModel

class MenziActNet():
  def __init__(self, weight_bin_path, urdf_path, nenv = 1, simple_hydraulic_model = False, device = 'cpu'):
    """
      Initializes the ActuatorNet class.
      Args:
        weight_bin_path (str): Path to the binary file containing the weights for the neural network.
        urdf_path (str): Path to the URDF file for the robotic system.
        nenv (int, optional): Number of environments for parallelization. Defaults to 1.
        simple_hydraulic_model (bool, optional): Flag to use a simple hydraulic model. Defaults to False.
        device (str, optional): Device to run the computations on ('cpu' or 'cuda'). Defaults to 'cpu'.
      Attributes:
        no_dof (int): Number of degrees of freedom.
        nenv (int): Number of environments for parallelization.
        use_simple_hydraulic (bool): Flag to use a simple hydraulic model.
        device (str): Device to run the computations on.
        dt (float): Time step for the actuator network.
        heap_kinematics (pk.Chain): Kinematic chain built from the URDF file.
        heapkin_base2eec (pk.SerialChain): Subchain from the base to the end-effector contact.
        architecture (torch.nn.Module): Neural network architecture for the actuator model.
        pos_buffer (torch.Tensor): Buffer for position history.
        vel_buffer (torch.Tensor): Buffer for velocity history.
        stpt_buffer (torch.Tensor): Buffer for setpoint history.
        posMeans (torch.Tensor): Mean values for position normalization.
        posStds (torch.Tensor): Standard deviation values for position normalization.
        velMeans (torch.Tensor): Mean values for velocity normalization.
        velStds (torch.Tensor): Standard deviation values for velocity normalization.
        stptMeans (torch.Tensor): Mean values for setpoint normalization.
        stptStds (torch.Tensor): Standard deviation values for setpoint normalization.
        tmpMean (torch.Tensor): Mean value for temperature normalization.
        tmpStd (torch.Tensor): Standard deviation value for temperature normalization.
        rpmMean (torch.Tensor): Mean value for RPM normalization.
        rpmStd (torch.Tensor): Standard deviation value for RPM normalization.
        pos_idxs (torch.Tensor): Indices for position history.
        vel_idxs (torch.Tensor): Indices for velocity history.
        stpt_idxs (torch.Tensor): Indices for setpoint history.
        pos_limit (torch.Tensor): Limits for position values.
    """

    pos_hist_len = 2
    pos_hist_stride = 10
    vel_hist_len = 10
    vel_hist_stride = 1
    stpt_hist_len = 33
    stpt_hist_stride = 3
    no_dof = 4
    self.no_dof = no_dof
    if not simple_hydraulic_model:
      self.nenv = nenv
    else:
      self.nenv = 1
    self.use_simple_hydraulic = simple_hydraulic_model
    self.device = device

    dim_in = no_dof*(pos_hist_len + 1 + vel_hist_len + 1 + stpt_hist_len + 1) + 2

    # self.actnet_dt = 0.01
    # self.sim_dt = sim_dt
    # self.step_division = int(self.actnet_dt/self.sim_dt + 1e-5)
    self.dt = 0.01

    urdf_path = os.path.join(project_path, "rsc/m545/m545_boom_dipper_tele_pitch.urdf")
    self.heap_kinematics = pk.build_chain_from_urdf(open(urdf_path, 'rb').read())
    # subchain EE_Contact wrt base
    self.heapkin_base2eec = pk.SerialChain(self.heap_kinematics, 'ENDEFFECTOR_CONTACT', 'CABIN').to(dtype=torch.float32, device=self.device)

    if self.use_simple_hydraulic:
      assert False, "Simple hydraulic model not implemented"
      # Use reduced order hydraulic model with nonlinearities
      cfg = YAML().load(open(os.path.join(project_path, "cfg/cfg_online_learning.yaml")))
      cfg["num_parallelization"] = nenv
      self.architecture = torch.compile(HydraulicActuatorModel(cfg, self.device))
      self.architecture.reset()
      
      inMeans_ = torch.zeros(14, device=self.device)
      inStds_ = torch.ones(14, device=self.device)
      self.outMeans_ = torch.zeros(4, device=self.device)
      self.outStds_ = torch.ones(4, device=self.device)

      self.pos_buffer = torch.zeros((self.architecture.n_parallelization, pos_hist_len*pos_hist_stride+1, 4)).to(self.device)
      self.vel_buffer = torch.zeros((self.architecture.n_parallelization, vel_hist_len*vel_hist_stride+11, 4)).to(self.device)
      self.stpt_buffer = torch.zeros((self.architecture.n_parallelization, stpt_hist_len*stpt_hist_stride+1, 4)).to(self.device)

    else:
      # Use actuator network
      self.weight_bin = weight_bin_path
      self.architecture = nn.Sequential(
        nn.Linear(dim_in, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 4)
      ).to(self.device)
      self.architecture.train(False)
      self.architecture.requires_grad_(False)

      no_params = 0
      for ele in self.architecture.state_dict():
        no_params += self.architecture.state_dict()[ele].numel()

      f = open(weight_bin_path, 'rb')
      from array import array
      try:
        dimsarray = array('L')
        dimsarray.fromfile(f, 2)
        weightarray = array('f')
        weightarray.fromfile(f, dimsarray[0])
      except EOFError:
        print("EOFError, make sure the dimensionality of the model is correct")
        f.close()
        return  
      finally:
        f.close()

      state_dict_from_bin = self.architecture.state_dict()
      weight_idx = 0
      for ele in state_dict_from_bin:
        state_dict_from_bin[ele] = torch.from_numpy(np.array(weightarray[weight_idx:weight_idx+state_dict_from_bin[ele].numel()]).reshape(state_dict_from_bin[ele].shape, order='F'))
        weight_idx += state_dict_from_bin[ele].numel()
      self.architecture.load_state_dict(state_dict_from_bin)

      inMeans_ = torch.tensor([
        -8.05776873e-01,
        -2.67923520e-04,
        3.37058329e-02,
        1.26712022e+00,
        -2.39100833e-04,
        1.73984845e-02,
        9.41187465e-01,
        2.47239296e-04,
        -2.38371266e-02,
        6.69505075e-01,
        1.12226883e-03,
        2.20907048e-02,
        5.88260766e+01,
        1.52313138e+03], device=self.device)
      inStds_ = torch.tensor([
        0.30391076,
        0.12098661,
        0.30277164,
        0.42981509,
        0.17843214,
        0.33397835,
        0.56450499,
        0.17670568,
        0.28408635,
        0.86848699,
        0.41701002,
        0.28605379,
        4.29141358,
        26.35360754], device=self.device)

      self.outMeans_ = torch.tensor([3.46894301e-07, -1.30858329e-07, -1.60912608e-07, -1.80913785e-05], device=self.device)
      self.outStds_ = torch.tensor([0.00528237, 0.0069478, 0.01019651, 0.02800822], device=self.device)

      self.pos_buffer = torch.zeros((self.nenv, pos_hist_len*pos_hist_stride+1, 4)).to(self.device)
      self.vel_buffer = torch.zeros((self.nenv, vel_hist_len*vel_hist_stride+11, 4)).to(self.device)
      self.stpt_buffer = torch.zeros((self.nenv, stpt_hist_len*stpt_hist_stride+1, 4)).to(self.device)

    self.posMeans = inMeans_[0:12:3]
    self.posStds = inStds_[0:12:3]
    self.velMeans = inMeans_[1:12:3]
    self.velStds = inStds_[1:12:3]
    self.stptMeans = inMeans_[2:12:3]
    self.stptStds = inStds_[2:12:3]
    self.tmpMean = inMeans_[12]
    self.tmpStd = inStds_[12]
    self.rpmMean = inMeans_[13]
    self.rpmStd = inStds_[13]


    self.pos_idxs = torch.tensor(range(0, pos_hist_len*pos_hist_stride+1, pos_hist_stride))
    self.vel_idxs = torch.tensor(range(0, vel_hist_len*vel_hist_stride+1, vel_hist_stride))
    self.stpt_idxs = torch.tensor(range(0, stpt_hist_len*stpt_hist_stride+1, stpt_hist_stride))

    # self.pos_limit = torch.tensor([[-1.34, 0.44], [0.58, 2.78], [0.0, 1.8], [-0.58, 2.28]], device=self.device)
    self.pos_limit = torch.tensor([[-1.2, -0.5], [0.58, 1.5708], [0.0, 0.2], [0.0, 1.5708]], device=self.device)

  def reset(self):
    self.reset_nominal()
  
  def reset_static_pos(self, pos_init_):
    if pos_init_.shape == (self.no_dof,):
      self.pos_buffer.copy_(((pos_init_-self.posMeans)/self.posStds).repeat(*self.pos_buffer.shape[0:-1], 1))
    elif pos_init_.shape == (self.nenv, self.no_dof):
      self.pos_buffer.copy_(((pos_init_-self.posMeans)/self.posStds).unsqueeze(1).repeat(1, *self.pos_buffer.shape[1:-1], 1))
    else:
      raise ValueError("pos_init_ shape unknown")

    self.vel_buffer.copy_(((torch.zeros((self.no_dof), device=self.device)-self.velMeans)/self.velStds).repeat(*self.vel_buffer.shape[0:-1], 1))
    self.stpt_buffer.copy_(((torch.zeros((self.no_dof), device=self.device)-self.stptMeans)/self.stptStds).repeat(*self.stpt_buffer.shape[0:-1], 1))
    if self.use_simple_hydraulic:
      self.architecture.reset()

  def reset_static_pos_random(self):
    # embed()
    if self.use_simple_hydraulic:
      rand_pos_init = torch.rand(self.architecture.n_parallelization, 4).to(self.device) * (self.pos_limit[:,1] - self.pos_limit[:,0]) + self.pos_limit[:,0]
    else:
      rand_pos_init = torch.rand(self.nenv, 4).to(self.device) * (self.pos_limit[:,1] - self.pos_limit[:,0]) + self.pos_limit[:,0]
    self.pos_buffer = ((rand_pos_init-self.posMeans)/self.posStds).unsqueeze(1).repeat(1, *self.pos_buffer.shape[1:-1], 1)
    self.vel_buffer = (((torch.zeros((self.no_dof), device=self.device)-self.velMeans)/self.velStds).repeat(*self.vel_buffer.shape[0:-1], 1))
    self.stpt_buffer = (((torch.zeros((self.no_dof), device=self.device)-self.stptMeans)/self.stptStds).repeat(*self.stpt_buffer.shape[0:-1], 1))
    if self.use_simple_hydraulic:
      self.architecture.reset()

  def reset_nominal(self):
    self.pos_buffer.copy_(((0.5*(self.pos_limit[:,0]+self.pos_limit[:,1])-self.posMeans)/self.posStds).repeat(*self.pos_buffer.shape[0:-1], 1))
    self.vel_buffer.copy_(((torch.zeros((self.no_dof), device=self.device)-self.velMeans)/self.velStds).repeat(*self.vel_buffer.shape[0:-1], 1))
    self.stpt_buffer.copy_(((torch.zeros((self.no_dof), device=self.device)-self.stptMeans)/self.stptStds).repeat(*self.stpt_buffer.shape[0:-1], 1))
    if self.use_simple_hydraulic:
      self.architecture.reset()

  def advance(self, u: torch.tensor):
    # Sahpe of u tensor should be (n_env, 4)
    if self.use_simple_hydraulic:
      assert u.dim() == 2 and u.shape[0] == self.architecture.n_parallelization and u.shape[1] == 4
    else:
      assert u.dim() == 2 and u.shape[0] == self.nenv and u.shape[1] == 4
    
    # pos_buffer = pos(t), pos(t-1) ...
    # vel_buffer = vel(t), vel(t-1) ...
    # stpt_buffer = stpt(t-1), stpt(t-2) ...
    u_s = (torch.clip(u, -1., 1.) - self.stptMeans)/self.stptStds

    # self.stpt_buffer[:,1:] = self.stpt_buffer.clone()[:,:-1]
    # self.stpt_buffer[:,0] = u_s
    self.stpt_buffer = torch.cat([u_s.unsqueeze(1), self.stpt_buffer[:, :-1]], dim=1)
    # stpt_buffer = stpt(t), stpt(t-2) ...

    if self.use_simple_hydraulic:
      # breakpoint()
      self.architecture.forward(u_s.T)
      vel_abs = self.architecture.vel_history[:,:,0].T
      pos_abs = self.pos_buffer[:,0]*self.posStds + self.posMeans + self.dt*vel_abs
      pos_abs = torch.clip(pos_abs, self.pos_limit[:,0], self.pos_limit[:,1])

      # self.pos_buffer[:,1:] = self.pos_buffer.clone()[:,:-1]
      # self.pos_buffer[:,0] = (pos_abs - self.posMeans)/self.posStds
      self.pos_buffer = torch.cat([(pos_abs.unsqueeze(1) - self.posMeans) / self.posStds, self.pos_buffer[:, :-1]], dim=1)
      # self.vel_buffer[:,1:] = self.vel_buffer.clone()[:,:-1]
      # self.vel_buffer[:,0] = (vel_abs - self.velMeans)/self.velStds
      self.vel_buffer = torch.cat([(vel_abs.unsqueeze(1) - self.velMeans) / self.velStds, self.vel_buffer[:, :-1]], dim=1)

      return pos_abs, vel_abs, None
    else:
      model_in = []
      for i in range(self.no_dof):
        # list of n_env x n_step tensors
        model_in.append(self.pos_buffer[:, self.pos_idxs, i])
        model_in.append(self.vel_buffer[:, self.vel_idxs, i])
        model_in.append(self.stpt_buffer[:, self.stpt_idxs, i])

      # Mean values for tmp and engine rpm
      model_in.append(torch.zeros((self.nenv, 2)).to(self.device))
      model_in = torch.cat(model_in, dim=1).to(self.device)

      # with torch.no_grad():
      # n_envs x 4
      dvels = self.architecture(model_in.float().to(self.device))

      vel_abs = self.vel_buffer[:,0]*self.velStds + self.velMeans + (dvels*self.outStds_ + self.outMeans_)
      pos_abs = self.pos_buffer[:,0]*self.posStds + self.posMeans + self.dt*vel_abs
      pistonvel = self.jointveltocylindervel(vel_abs, pos_abs)
      pos_abs = torch.clip(pos_abs, self.pos_limit[:,0], self.pos_limit[:,1])
      # self.pos_buffer[:,1:] = self.pos_buffer.clone()[:,:-1]
      # self.pos_buffer[:,0] = (pos_abs - self.posMeans)/self.posStds
      self.pos_buffer = torch.cat([(pos_abs.unsqueeze(1) - self.posMeans) / self.posStds, self.pos_buffer[:, :-1]], dim=1)
      # self.vel_buffer[:,1:] = self.vel_buffer.clone()[:,:-1]
      # self.vel_buffer[:,0] = (vel_abs - self.velMeans)/self.velStds
      self.vel_buffer = torch.cat([(vel_abs.unsqueeze(1) - self.velMeans) / self.velStds, self.vel_buffer[:, :-1]], dim=1)
      # pos_buffer = pos(t+1), pos(t) ...
      # vel_buffer = vel(t+1), vel(t) ...
        # embed()
      return pos_abs, vel_abs, pistonvel

  def idSignal(self, mode, t):
    if mode == 0:
      return -1.0 if (t%8.0) > 4.0 else 1.0
    elif mode == 1:
      return -0.5 if (t%8.0) > 4.0 else 0.5
    elif mode == 2:
      return torch.sin(2.0*torch.pi*2.0*t)
    elif mode == 3:
      return torch.sin(2.0*torch.pi*0.5*t)
    else:
      return 0.0
  
  def get_response(self, no_dim=4):
    control_dt = 0.04
    response_steps = 200
    response = torch.zeros((self.no_dof*4, response_steps))
    no_substeps = int(control_dt/self.dt + 1e-5)
    for i in range(no_dim):
      for m in range(4):
        self.reset()
        for step in range(200):
          u = torch.zeros((4,1))
          u[i] = self.idSignal(m, step*control_dt)
          for substep in range(no_substeps):
            pos, vel, cylindervel = self.advance(u)
            # cylindervel = self.jointveltocylindervel(vel, pos)
          response[i*4+m, step] = cylindervel[i]

    self.reset()
    return response
  
  def get_map(self):
    map_points = 21
    map = torch.tensor([
      [-0.18, -0.18, -0.18, -0.153, -0.126, -0.0905, -0.055, -0.036, -0.0135, -0.005, 0.0, 0.004, 0.0165, 0.034, 0.06, 0.09, 0.12, 0.128, 0.136, 0.138, 0.14],
      [-0.295, -0.2825, -0.27, -0.248, -0.226, -0.1705, -0.115, -0.071, -0.034, -0.0085, 0.0, 0.001, 0.011, 0.033, 0.08, 0.1375, 0.195, 0.2325, 0.27, 0.289, 0.308],
      [-0.57, -0.57, -0.57, -0.5325, -0.495, -0.37, -0.245, -0.147, -0.075, -0.021, 0.0, 0.022, 0.125, 0.21, 0.33, 0.43, 0.53, 0.565, 0.6, 0.615, 0.63],
      [-0.33, -0.33, -0.33, -0.3115, -0.293, -0.2225, -0.152, -0.094, -0.053, -0.021, 0.0, 0.012, 0.044, 0.082, 0.129, 0.182, 0.235, 0.2845, 0.334, 0.334, 0.334]])
    return map

  # For all conversion functions below:
  #  Arguments should be like:
  #  {joint, cylinder}{pos, vel} as 2d tensors of shape (n_env, dof)
  #  where nenv is the number of parallel calculations,
  #  and dof=4 is the number of degrees of freedom of the menzi arm
  #  np arrays of the same shape are treated as torch tensors
  #  1d arrays/tensors of length n will are treated as n_env=1
  #  The return values are always 2d tensors of n_env x dof

  def jointveltocylindervel(self, jointvel, jointpos):
    assert type(jointvel) == type(jointpos)
    assert jointvel.shape == jointpos.shape
    # breakpoint()
    if type(jointvel) == np.ndarray:
      jointvel = torch.from_numpy(jointvel)
    if type(jointpos) == np.ndarray:
      jointpos = torch.from_numpy(jointpos)
    input_dim = jointvel.dim()
    
    if input_dim == 2:
      # n_envs x dof
      inputvel = jointvel
      inputpos = jointpos
    elif input_dim == 1:
      # 1 x dof
      inputvel = jointvel.unsqueeze(0)
      inputpos = jointpos.unsqueeze(0)

    cylindervel = torch.zeros_like(inputvel)

    cylindervel[:,0] = inputvel[:,0] * self.ftau_boom(inputpos[:,0])
    cylindervel[:,1] = inputvel[:,1] * self.ftau_dipper(inputpos[:,1])
    cylindervel[:,2] = inputvel[:,2] * self.ftau_tele(inputpos[:,2])
    cylindervel[:,3] = inputvel[:,3] * self.ftau_pitch(inputpos[:,3])
    
    return cylindervel
  
  def cylinderveltojointvel(self, cylindervel, jointpos):
    assert type(cylindervel) == type(jointpos)
    assert cylindervel.shape == jointpos.shape
    if type(cylindervel) == np.ndarray:
      cylindervel = torch.from_numpy(cylindervel)
    if type(jointpos) == np.ndarray:
      jointpos = torch.from_numpy(jointpos)
    input_dim = cylindervel.dim()
    
    if input_dim == 2:
      inputvel = cylindervel
      inputpos = jointpos
    elif input_dim == 1:
      inputvel = cylindervel.unsqueeze(0)
      inputpos = jointpos.unsqueeze(0)

    jointvel = torch.zeros_like(inputvel)
    # embed()
    jointvel[:,0] = inputvel[:,0] / self.ftau_boom(inputpos[:,0])
    jointvel[:,1] = inputvel[:,1] / self.ftau_dipper(inputpos[:,1])
    jointvel[:,2] = inputvel[:,2] / self.ftau_tele(inputpos[:,2])
    jointvel[:,3] = inputvel[:,3] / self.ftau_pitch(inputpos[:,3])
    
    return jointvel
  
  def jointpostocylinderpos(self, jointpos):
    if type(jointpos) == np.ndarray:
      jointpos = torch.from_numpy(jointpos)
    input_dim = jointpos.dim()
    
    if input_dim == 2:
      inputpos = jointpos
    elif input_dim == 1:
      inputpos = jointpos.unsqueeze(0)

    outputpos = torch.zeros_like(inputpos)
    outputpos[:,0] = self.pos_j2c_boom(inputpos[:,0])
    outputpos[:,1] = self.pos_j2c_dipper(inputpos[:,1])
    outputpos[:,2] = inputpos[:,2]
    outputpos[:,3] = self.pos_j2c_pitch(inputpos[:,3])
    return outputpos

  def cylinderpostojointpos(self, cylinderpos):
    if type(cylinderpos) == np.ndarray:
      cylinderpos = torch.from_numpy(cylinderpos)
    input_dim = cylinderpos.dim()

    if input_dim == 2:
      inputpos = cylinderpos
    elif input_dim == 1:
      inputpos = cylinderpos.unsqueeze(0)
    
    outputpos = torch.zeros_like(inputpos)
    outputpos[:,0] = self.pos_c2j_boom(inputpos[:,0])
    outputpos[:,1] = self.pos_c2j_dipper(inputpos[:,1])
    outputpos[:,2] = inputpos[:,2]
    outputpos[:,3] = self.pos_c2j_pitch(inputpos[:,3])
    return outputpos

  def pos_j2c_boom(self, boomjointpos):
    beta0_ = (90 - 18 - 16.3111)*torch.pi/180.
    direction_ = -1.
    beta = beta0_ + direction_ * boomjointpos
    b_ = 1.4446
    a_ = 0.4
    x0_ = 1.129
    return torch.sqrt(b_ * b_ + a_ * a_ - 2.0 * a_ * b_ * torch.cos(beta)) - x0_
  
  def pos_c2j_boom(self, boomcylinderpos):
    beta0_ = (90 - 18 - 16.3111)*torch.pi/180.
    direction_ = -1.
    b_ = 1.4446
    a_ = 0.4
    x0_ = 1.129
    cylinderLength = boomcylinderpos + x0_
    beta = torch.acos((a_ * a_ + b_ * b_ - cylinderLength * cylinderLength) / (2 * a_ * b_))
    return direction_ * (beta - beta0_)

  def pos_j2c_dipper(self, dipperjointpos):
    x0_ = 1.545
    a1_ = 11.05 / 180 * torch.pi
    a2_ = 50.0 / 180 * torch.pi
    a3_ = 8.89 / 180 * torch.pi
    l1_ = 0.15
    l2_ = 0.49
    l3_ = 0.59
    l4_ = 2.13378

    aEFD = a2_ + dipperjointpos + a1_
    lDE = torch.sqrt(l1_ * l1_ + l2_ * l2_ - 2 * l1_ * l2_ * torch.cos(aEFD))
    aDEF = torch.acos((lDE * lDE + l1_ * l1_ - l2_ * l2_) / (2 * lDE * l1_))
    aDEF[dipperjointpos > (torch.pi - (a2_ + a1_))] *= -1.
    aCED = torch.acos((l3_ * l3_ + lDE * lDE - l1_ * l1_) / (2 * l3_ * lDE))
    aECD = torch.acos((l3_ * l3_ + l1_ * l1_ - lDE * lDE) / (2 * l3_ * l1_))
    aBCE = torch.pi - aECD
    lBE = torch.sqrt(l1_ * l1_ + l3_ * l3_ - 2 * l1_ * l3_ * torch.cos(aBCE))
    aBEC = torch.acos((lBE * lBE + l3_ * l3_ - l1_ * l1_) / (2 * lBE * l3_))
    aAEB = torch.pi - a3_ - a2_ - aDEF - aBEC - aCED
    cylinderPosition = torch.sqrt(l4_ * l4_ + lBE * lBE - 2 * l4_ * lBE * torch.cos(aAEB))

    return cylinderPosition - x0_
  
  def pos_c2j_dipper(self, dippercylinderpos):
    x0_ = 1.545
    cylinderLength = dippercylinderpos + x0_
    # From documentation thesis Martin Rudin
    # 5. order interpolation of inverted forward solution (joint angle to piston position)
    jointAngle = -0.048553 * torch.pow(cylinderLength, 5) - 0.70524 * torch.pow(cylinderLength, 4) + 8.4555 * torch.pow(cylinderLength, 3) - 29.499 * torch.pow(cylinderLength, 2) + 45.078 * torch.pow(cylinderLength, 1) - 25.425;
    return jointAngle;

  def pos_j2c_pitch(self, pitchjointpos):
    # from documentation thesis Martin Rudin
    x0_ = 0.5
    a1_ = 0.0
    a2_ = 7.393*torch.pi/180.
    a3_ = 29.4*torch.pi/180.
    lDE_ = 0.272
    lCE_ = 0.35
    lBC_ = 0.38
    lBD_ = 0.56
    lAD_ = 0.877
    aCED = a2_ + pitchjointpos + (torch.pi / 2 - a1_)
    lDC = torch.sqrt(lDE_ * lDE_ + lCE_ * lCE_ - 2 * lDE_ * lCE_ * torch.cos(aCED))
    aDCE = torch.arccos((lDC * lDC + lCE_ * lCE_ - lDE_ * lDE_) / (2 * lDC * lCE_))
    aBCD = torch.arccos((lDC * lDC + lBC_ * lBC_ - lBD_ * lBD_) / (2 * lDC * lBC_))
    aDBC = torch.arccos((lBD_ * lBD_ + lBC_ * lBC_ - lDC * lDC) / (2 * lBD_ * lBC_))
    aBCE = aBCD + aDCE
    aBFC = torch.pi - aDBC - aBCE
    lBF = lBC_ * torch.sin(aBCE) / torch.sin(aBFC)
    lDF = lBF - lBD_
    aCDE = torch.arccos((lDC * lDC + lDE_ * lDE_ - lCE_ * lCE_) / (2 * lDC * lDE_))
    aBDC = torch.arccos((lBD_ * lBD_ + lDC * lDC - lBC_ * lBC_) / (2 * lBD_ * lDC))
    aADB = torch.pi - a2_ - a3_ - aCDE - aBDC
    aADF = torch.pi - aADB
    lAB = torch.sqrt(lAD_ * lAD_ + lBD_ * lBD_ - 2 * lAD_ * lBD_ * torch.cos(aADB))
    lAF = torch.sqrt(lAD_ * lAD_ + lDF * lDF - 2 * lAD_ * lDF * torch.cos(aADF))
    aABF = torch.arccos((lAB * lAB + lBF * lBF - lAF * lAF) / (2 * lAB * lBF))
    aFDC = torch.pi - aDBC
    lCF = torch.sqrt(lDC * lDC + lDF * lDF - 2 * lDC * lDF * torch.cos(aFDC))
    a4 = torch.pi / 2.0 - aABF

    aCED = a2_ + pitchjointpos + (torch.pi / 2 - a1_)
    lDC = torch.sqrt(lDE_ * lDE_ + lCE_ * lCE_ - 2 * lDE_ * lCE_ * torch.cos(aCED))
    aCDE = torch.acos((lDC * lDC + lDE_ * lDE_ - lCE_ * lCE_) / (2 * lDC * lDE_))
    # if (pitchjointpos > (torch.pi / 2 - a2_ + a1_)) {
    #   aCDE = -aCDE;
    # }
    aCDE[pitchjointpos > (torch.pi / 2 - a2_ + a1_)] *= -1.
    aBDC = torch.acos((lBD_ * lBD_ + lDC * lDC - lBC_ * lBC_) / (2 * lBD_ * lDC))
    aADB = torch.pi - a2_ - a3_ - aCDE - aBDC
    lAB = torch.sqrt(lAD_ * lAD_ + lBD_ * lBD_ - 2 * lAD_ * lBD_ * torch.cos(aADB))

    return (lAB - x0_)

  def pos_c2j_pitch(self, pitchcylinderpos):
    x0_ = 0.5
    a1_ = 0.0
    a2_ = 7.393*torch.pi/180.
    a3_ = 29.4*torch.pi/180.
    lDE_ = 0.272
    lCE_ = 0.35
    lBC_ = 0.38
    lBD_ = 0.56
    lAD_ = 0.877

    cylinderPosition = pitchcylinderpos + x0_;
    # from documentation thesis Martin Rudin
    aADB = torch.acos((lAD_ * lAD_ + lBD_ * lBD_ - cylinderPosition * cylinderPosition) / (2 * lAD_ * lBD_))
    aBDE = torch.pi - a3_ - a2_ - aADB
    lBE = torch.sqrt(lBD_ * lBD_ + lDE_ * lDE_ - 2 * lBD_ * lDE_ * torch.cos(aBDE))
    aDEB = torch.acos((lDE_ * lDE_ + lBE * lBE - lBD_ * lBD_) / (2 * lDE_ * lBE))
    aBEC = torch.acos((lBE * lBE + lCE_ * lCE_ - lBC_ * lBC_) / (2 * lBE * lCE_))
    aCED = aDEB + aBEC
    return aCED - a2_ - (torch.pi / 2 - a1_)

  def ftau_boom(self, boompos):
    beta0_ = (90 - 18 - 16.3111)*torch.pi/180.
    direction_ = -1.
    beta = beta0_ + direction_ * boompos
    b_ = 1.4446
    a_ = 0.4
    return (direction_ * b_ * torch.sin(torch.arctan2(a_ * torch.sin(beta), b_ - a_ * torch.cos(beta))))

  def ftau_dipper(self, dipperpos):
    x0_ = 1.545
    a1_ = 11.05 / 180 * torch.pi
    a2_ = 50.0 / 180 * torch.pi
    a3_ = 8.89 / 180 * torch.pi
    l1_ = 0.15
    l2_ = 0.49
    l3_ = 0.59
    l4_ = 2.13378
    # from documentation thesis Martin Rudin
    aEFD = a2_ + dipperpos + a1_
    lDE = torch.sqrt(l1_ * l1_ + l2_ * l2_ - 2 * l1_ * l2_ * torch.cos(aEFD))
    aEDF = torch.arccos((lDE * lDE + l2_ * l2_ - l1_ * l1_) / (2 * lDE * l2_))
    aCDE = torch.arccos((l1_ * l1_ + lDE * lDE - l3_ * l3_) / (2 * l1_ * lDE))
    aECD = torch.arccos((l3_ * l3_ + l1_ * l1_ - lDE * lDE) / (2 * l3_ * l1_))
    aCDF = aCDE + aEDF
    aCGD = torch.pi - aECD - aCDF
    lEG = l1_ * torch.sin(aCDF) / torch.sin(aCGD) - l3_
    aBCE = torch.pi - aECD
    lBE = torch.sqrt(l1_ * l1_ + l3_ * l3_ - 2 * l1_ * l3_ * torch.cos(aBCE))
    aDEF = torch.arccos((lDE * lDE + l1_ * l1_ - l2_ * l2_) / (2 * lDE * l1_))
    # if (dipperpos > (torch.pi - (a2_ + a1_))):
    #   aDEF = -aDEF
    aDEF[dipperpos > (torch.pi - (a2_ + a1_))] *= -1.
    aCED = torch.arccos((l3_ * l3_ + lDE * lDE - l1_ * l1_) / (2 * l3_ * lDE))
    aBEC = torch.arccos((lBE * lBE + l3_ * l3_ - l1_ * l1_) / (2 * lBE * l3_))
    aAEB = torch.pi - a3_ - a2_ - aDEF - aBEC - aCED
    aAEG = torch.pi - aAEB - aBEC
    aGEB = aAEB + aAEG
    lBG = torch.sqrt(lBE * lBE + lEG * lEG - 2 * lBE * lEG * torch.cos(aGEB))
    lAB = torch.sqrt(l4_ * l4_ + lBE * lBE - 2 * l4_ * lBE * torch.cos(aAEB))
    lAG = torch.sqrt(l4_ * l4_ + lEG * lEG - 2 * l4_ * lEG * torch.cos(aAEG))
    lCG = lEG + l3_
    lDG = torch.sqrt(lCG * lCG + l1_ * l1_ - 2 * lCG * l1_ * torch.cos(aECD))
    aABG = torch.arccos((lAB * lAB + lBG * lBG - lAG * lAG) / (2 * lAB * lBG))
    a6 = torch.pi / 2 - aABG

    return 1.0 / (lDG / (lBG * l2_ * torch.cos(a6)))
  
  def ftau_tele(self, telepos):
    return torch.ones_like(telepos)
  
  def ftau_pitch(self, pitchpos):
    # from documentation thesis Martin Rudin
    x0_ = 0.5
    a1_ = 0.0
    a2_ = 7.393*torch.pi/180.
    a3_ = 29.4*torch.pi/180.
    lDE_ = 0.272
    lCE_ = 0.35
    lBC_ = 0.38
    lBD_ = 0.56
    lAD_ = 0.877
    aCED = a2_ + pitchpos + (torch.pi / 2 - a1_)
    lDC = torch.sqrt(lDE_ * lDE_ + lCE_ * lCE_ - 2 * lDE_ * lCE_ * torch.cos(aCED))
    aDCE = torch.arccos((lDC * lDC + lCE_ * lCE_ - lDE_ * lDE_) / (2 * lDC * lCE_))
    aBCD = torch.arccos((lDC * lDC + lBC_ * lBC_ - lBD_ * lBD_) / (2 * lDC * lBC_))
    aDBC = torch.arccos((lBD_ * lBD_ + lBC_ * lBC_ - lDC * lDC) / (2 * lBD_ * lBC_))
    aBCE = aBCD + aDCE
    aBFC = torch.pi - aDBC - aBCE
    lBF = lBC_ * torch.sin(aBCE) / torch.sin(aBFC)
    lDF = lBF - lBD_
    aCDE = torch.arccos((lDC * lDC + lDE_ * lDE_ - lCE_ * lCE_) / (2 * lDC * lDE_))
    aBDC = torch.arccos((lBD_ * lBD_ + lDC * lDC - lBC_ * lBC_) / (2 * lBD_ * lDC))
    aADB = torch.pi - a2_ - a3_ - aCDE - aBDC
    aADF = torch.pi - aADB
    lAB = torch.sqrt(lAD_ * lAD_ + lBD_ * lBD_ - 2 * lAD_ * lBD_ * torch.cos(aADB))
    lAF = torch.sqrt(lAD_ * lAD_ + lDF * lDF - 2 * lAD_ * lDF * torch.cos(aADF))
    aABF = torch.arccos((lAB * lAB + lBF * lBF - lAF * lAF) / (2 * lAB * lBF))
    aFDC = torch.pi - aDBC
    lCF = torch.sqrt(lDC * lDC + lDF * lDF - 2 * lDC * lDF * torch.cos(aFDC))
    a4 = torch.pi / 2.0 - aABF
    # Shovel dogbone torquefactor
    x = 1.0 / (lCF / (lCE_ * lBF * torch.cos(a4)))

    # if pitchpos < -0.361476:
    #   x = -x
    # if pitchpos > 1.441764:
    #   x = x - 2 * (x - 0.227971989)

    x[pitchpos < -0.361476] *= -1.
    x[pitchpos > 1.441764] = x[pitchpos > 1.441764] - 2 * (x[pitchpos > 1.441764] - 0.227971989)

    return x


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--weight_bin', type=str, default='/home/fang/Projects/rl_grading_m545/rsc/actuatorModel/modelWeightsDrawwireAll.bin', help='path to trained')
  args = parser.parse_args()

  model = MenziActNet(args.weight_bin)
  model.reset()

  test_len = 3000
  data = torch.zeros((4*3, test_len))
  for i in range(test_len):
    u_test = torch.tensor([[0.8*(i>200), 0.8*(i>400), 0.8*(i>600), 0.8*(i>800)]])
    pos, vel, pistonvel = model.advance(u_test)
    data[:,i] = torch.cat((pos[0], vel[0], u_test[0]), dim=0)
  
  np.savetxt("mm_model_test.txt", data)
