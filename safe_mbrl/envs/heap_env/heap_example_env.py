import torch
import numpy as np

import pytorch_kinematics as pk

import os, sys
file_path = os.path.dirname(os.path.abspath(__file__))

sys.path.append(file_path)

from plant.actuator_net import MenziActNet
import time

import pinocchio as pin
from custom_meshcat_visualizer import AnimeMeshcatVisualizer

import gymnasium as gym
from matplotlib import pyplot as plt

class HeapEnv(gym.Env):
  def __init__(self, n_envs = 1,
               use_act_net = True,
               n_history_steps= 10,
               n_ref_steps = 15,
               t_step = 0.04,
               t_traj = 6.0,
               cfg = None) -> None:
    """
      Initialize the heap example environment.
      Args:
        n_envs (int, optional): Number of environments. Defaults to 1.
        use_act_net (bool, optional): Flag to use the action network. Defaults to True.
        n_history_steps (int, optional): Number of history steps. Defaults to 10.
        n_ref_steps (int, optional): Number of reference steps. Defaults to 15.
        t_step (float, optional): Time step for the simulation. Must be a multiple of 0.01. Defaults to 0.04.
        t_traj (float, optional): Total trajectory time. Defaults to 6.0.
      Raises:
        AssertionError: If t_step is not a multiple of 0.01.
        AssertionError: If the number of degrees of freedom (DOF) in actnet and kinematics do not match.
      Attributes:
        ref_traj_steps (int): Number of reference trajectory steps.
        step_substeps (int): Number of substeps per step.
        kinematics (pk.SerialChain): Kinematics model.
        act_dim (int): Number of action dimensions (degrees of freedom).
        dof_pos_history (torch.Tensor): Tensor to store the history of DOF positions.
        dof_vel_history (torch.Tensor): Tensor to store the history of DOF velocities.
    """
    
    if torch.backends.mps.is_available():
      self.device = torch.device('mps')
    elif torch.cuda.is_available():
      self.device = torch.device('cuda')
    else:
      self.device = torch.device('cpu')

    self.cfg = cfg
    self.n_envs = n_envs
    self.n_history_steps = n_history_steps
    self.n_ref_steps = n_ref_steps
    self.t_traj = t_traj
    self.t_step = t_step
    assert self.t_step / 0.01 == int(self.t_step / 0.01), "t_step must be a multiple of 0.01"

    # total steps = trajectory time / time per step
    self.ref_traj_steps = int(self.t_traj / self.t_step) + 1
    self.step_substeps = int(self.t_step / 0.01)

    self.m545_rsc_dir = os.path.join(file_path, "rsc/m545/")
    self.m545_urdf_path = os.path.join(self.m545_rsc_dir, "m545_boom_dipper_tele_pitch.urdf")
    m545_model_bin = os.path.join(self.m545_rsc_dir, "modelWeightsDrawwireAll.bin")
    self.actnet = MenziActNet(m545_model_bin, self.m545_urdf_path, nenv=self.n_envs, device=self.device)
    self.kinematics = pk.build_serial_chain_from_urdf(open(self.m545_urdf_path, 'rb').read(), "ENDEFFECTOR_CONTACT", "CABIN").to(device=self.device)
    self.kinematics.print_tree()

    self.mesh_dir = os.path.join(self.m545_rsc_dir, "meshes/")

    assert self.actnet.no_dof == self.kinematics.n_joints, "actnet and kinematics must have the same number of joints"
    self.act_dim = self.actnet.no_dof
    self.act_min = -1
    self.act_max = 1
    
    self.dof_pos_history = torch.zeros(self.n_envs, self.actnet.no_dof, self.n_history_steps, device=self.device)
    self.dof_vel_history = torch.zeros(self.n_envs, self.actnet.no_dof, self.n_history_steps, device=self.device)

    self.observation_space = gym.spaces.Dict({
      'dof_pos': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_envs, self.actnet.no_dof, self.n_history_steps), dtype=np.float32),
      'dof_vel': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_envs, self.actnet.no_dof, self.n_history_steps), dtype=np.float32),
      'ee_pos': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_envs, 3), dtype=np.float32),
      'ee_vel': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_envs, 6), dtype=np.float32),
      'ref_traj_eepos': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_envs, self.n_ref_steps, 3), dtype=np.float32),
    })
    
    self.action_space = gym.spaces.Box(low=self.act_min, high=self.act_max, shape=(self.n_envs, self.act_dim), dtype=np.float32)

    self.is_rendering = False
    self.predicted_trajectories = []
    self.predicted_velocities = []
    self.joint_pos_histories = []
    self.joint_vel_histories = []
    self.reward_histories = []
    self.target_trajectories = []
    self.target_joint_pos_histories = []

  def sample_ref_traj(self, start_q: None | torch.Tensor = None):
    """
      Generates a sample reference trajectory for the environment.

      This method performs the following steps:
      1. Generates a polynomial trajectory for each joint using random initial and final positions.
      2. Scales the generated trajectory to fit within the joint position limits.
      3. Computes the forward kinematics to obtain the end-effector pose for each step in the trajectory.
      4. Extracts the end-effector positions from the computed poses.

      Attributes:
        ref_traj_joint_01 (torch.Tensor): The initial polynomial trajectory for each joint.
        ref_traj_joint (torch.Tensor): The scaled polynomial trajectory within joint limits.
        ref_traj_Tee (torch.Tensor): The transformation matrices representing the end-effector poses.
        ref_traj_eepos (torch.Tensor): The end-effector positions extracted from the transformation matrices.

      Returns:
        None
    """
    lo, hi = self.actnet.pos_limit[:, 0], self.actnet.pos_limit[:, 1]
    
    if start_q is not None:
      start_q = torch.tensor(start_q, device=self.device)
      start01 = ((start_q - lo) / (hi - lo)).clamp(0.0, 1.0)
    else:
      start01 = torch.rand(self.n_envs*self.actnet.no_dof, device=self.device)
      
    self.ref_traj_joint_01 = self.generate_polynomial_traj(start01,
                                                           torch.rand(self.n_envs*self.actnet.no_dof, device=self.device),
                                                           1.,
                                                           0.).reshape(self.n_envs, self.actnet.no_dof, -1).contiguous()
    self.ref_traj_joint = torch.einsum('ijk,j->ijk', self.ref_traj_joint_01, self.actnet.pos_limit[:, 1] - self.actnet.pos_limit[:, 0]) + self.actnet.pos_limit[:, 0].unsqueeze(0).unsqueeze(-1).repeat(self.n_envs, 1, self.ref_traj_steps).contiguous()
    self.ref_traj_Tee = self.kinematics.forward_kinematics(self.ref_traj_joint.transpose(1,2).reshape(self.n_envs*self.ref_traj_steps, -1)).get_matrix().reshape(self.n_envs, self.ref_traj_steps, 4, 4)
    self.ref_traj_eepos = self.ref_traj_Tee[:, :, :3, 3]
  
  def reset(self, seed=None, options=None, start_q=None, reset_model: bool = True):
    """
      Resets the environment to an initial state.

      This method performs the following actions:
      1. Samples a reference trajectory.
      2. Resets the actuator network's static positions using the first set of joint positions from the reference trajectory.
      3. Initializes the history of degrees of freedom (DoF) positions and velocities using the actuator network's buffers, 
        standard deviations, and means.
      4. Computes the end-effector (EE) position using forward kinematics based on the initial DoF positions.
      5. Computes the end-effector velocity using the Jacobian and initial DoF velocities.
      6. Sets the current step counter to zero.
      
      Returns:
        np.ndarray: The initial observation of the environment.
    """
    super().reset(seed=seed)

    self.sample_ref_traj(start_q)
    if reset_model:
      self.actnet.reset_static_pos(self.ref_traj_joint[:,:,0])
      self.dof_pos_history = (self.actnet.pos_buffer[:, 0]*self.actnet.posStds + self.actnet.posMeans).unsqueeze(-1).repeat(1, 1, self.n_history_steps)
      self.dof_vel_history = (self.actnet.vel_buffer[:, 0]*self.actnet.velStds + self.actnet.velMeans).unsqueeze(-1).repeat(1, 1, self.n_history_steps)
    self.ee_pos = self.kinematics.forward_kinematics(self.dof_pos_history[:,:,0]).get_matrix()[:, :3, 3]
    self.ee_vel = self.kinematics.jacobian(self.dof_pos_history[:,:,0]).matmul(self.dof_vel_history[:,:,0].unsqueeze(-1)).squeeze(-1)
    self.accel = (self.ee_vel - torch.zeros_like(self.ee_vel)) / self.t_step
    self.current_step = 0
    self.action = torch.zeros(self.n_envs, self.act_dim, device=self.device)
    return self._get_obs(), self._get_info()

  def step(self, action):
    """
      Perform a simulation step given an action.
      Args:
        action (Tensor): The action to be taken in the environment. shape: (n_envs, act_dim), act_dim = number of degrees of freedom.
      Returns:
        tuple: A tuple containing:
          - obs (Dict): The observation after taking the action. 
          - rwd (float): The reward obtained after taking the action.
          - done (bool): A boolean indicating whether the episode has ended.
    """
    if isinstance(action, np.ndarray):
        action = torch.tensor(action, device=self.device)
    action = action.to(self.device)
    action = torch.clamp(action, self.act_min, self.act_max)
    for substep in range(self.step_substeps):
      pos_joint, vel_joint, vel_piston = self.actnet.advance(action)
    
    self.last_accel = self.accel.clone()
    self.last_ee_pos = self.ee_pos.clone()
    self.last_ee_vel = self.ee_vel.clone()
    self.last_action = self.action.clone()
    
    self.dof_pos_history = torch.cat([pos_joint.unsqueeze(-1), self.dof_pos_history[:, :, :-1]], dim=-1)
    self.dof_vel_history = torch.cat([vel_joint.unsqueeze(-1), self.dof_vel_history[:, :, :-1]], dim=-1)
    self.ee_pos = self.kinematics.forward_kinematics(pos_joint).get_matrix()[:, :3, 3]
    self.ee_vel = self.kinematics.jacobian(pos_joint).matmul(vel_joint.unsqueeze(-1)).squeeze(-1)
    self.accel = (self.ee_vel - self.last_ee_vel) / self.t_step
    self.current_step += 1
    self.action = action

    obs = self._get_obs()
    rwd = self._compute_reward().cpu().numpy()
    terminated = self.current_step >= self.ref_traj_steps - 1
    truncated = False
    info = self._get_info()
    
    return obs, rwd, terminated, truncated, info
    

  def _get_obs(self):
    """(
      Generates and returns a dictionary containing the current observation of the environment.

      The observation dictionary includes:
      - 'dof_pos': History of degrees of freedom positions. shape: (n_envs, act_dim, n_history_steps)
      - 'dof_vel': History of degrees of freedom velocities. shape: (n_envs, act_dim, n_history_steps)
      - 'ee_pos': Current end-effector position. shape: (n_envs, 3)
      - 'ee_vel': Current end-effector velocity. shape: (n_envs, 6)
      - 'ref_traj_eepos': Reference trajectory end-effector positions for the current and future steps. shape: (n_envs, n_ref_steps, 3)

      Returns:
        dict: A dictionary containing the current observation of the environment.
    """
    obs = {}
    obs['dof_pos'] = self.dof_pos_history.cpu().numpy()
    obs['dof_vel'] = self.dof_vel_history.cpu().numpy()
    obs['ee_pos'] = self.ee_pos.cpu().numpy()
    obs['ee_vel'] = self.ee_vel.cpu().numpy()
    if self.current_step+self.n_ref_steps < self.ref_traj_steps:
      obs['ref_traj_eepos'] = self.ref_traj_eepos[:, self.current_step:self.current_step+self.n_ref_steps].cpu().numpy()
    else:
      obs['ref_traj_eepos'] = torch.cat([self.ref_traj_eepos[:, self.current_step:], self.ref_traj_eepos[:, [-1]].repeat(1, self.n_ref_steps - (self.ref_traj_steps - self.current_step), 1)], dim=1).cpu().numpy()
    return obs
  
  def _get_info(self):
    """
      Generates and returns a dictionary containing the current information of the environment.
    """
    info = {}
    return info

  def _compute_reward(self):
    """
      Compute the reward for the current step of the environment.

      The reward is calculated based on the negative squared norm of the difference
      between the current end-effector position and the reference trajectory position
      at the current step. This self.ee_posencourages the end-effector to follow the reference
      trajectory closely.

      self.ee_pos: Current end-effector position. shape: (n_envs, 3)
      self.ref_traj_eepos: Reference trajectory end-effector positions. shape: (n_envs, n_ref_steps, 3)

      Returns:
        reward: The reward for the current step. shape: (n_envs)
    """
    reward = torch.zeros((self.n_envs), device=self.device, dtype=torch.float32)
    reward = -torch.sum((self.ee_pos - self.ref_traj_eepos[:, self.current_step])**2, dim=-1)
    # TODO: How to set up the reward function?
    # The main objective we consider first is endeffector tracking
    # Let's keep it simple for now, maybe add small regularization when necessary
    # reward += ...

    # Penalize large steps
    step_penalty_coef = self.cfg.step_penalty_coef
    step_penalty = torch.sum((self.ee_pos - self.last_ee_pos)**2, dim=-1)

    # Penalize large actions
    action_penalty_coef = self.cfg.action_penalty_coef
    action_penalty = torch.sum((self.action - self.last_action)**2, dim=-1)

    # Penalize large accelerations
    accel_penalty_coef = self.cfg.accel_penalty_coef
    accel_penalty = torch.sum(torch.abs(self.accel - self.last_accel), dim=-1)

    # Penalize change of sign in acceleration
    accel_sign_penalty_coef = self.cfg.accel_sign_penalty_coef
    accel_sign_penalty = torch.sum((self.accel * self.last_accel < 0).float(), dim=-1)

    reward = reward - step_penalty_coef * step_penalty \
      - action_penalty_coef * action_penalty \
      - accel_penalty_coef * accel_penalty \
      - accel_sign_penalty_coef * accel_sign_penalty

    return reward
    

  def generate_polynomial_traj(self, p_start, p_end, p_max, p_min):
   
    """
      Generates smooth polynomial trajectories from p_start to p_end over time t for multiple trajectories.
      The velocity at the start and end are constrained to be zero.

      Args:
        p_start (torch.Tensor): The starting positions of the trajectories. 
        p_end (torch.Tensor): The ending positions of the trajectories. 
        p_max (torch.Tensor): The maximum allowable positions. 
        p_min (torch.Tensor): The minimum allowable positions. 
      Returns:
        torch.Tensor: The generated polynomial trajectories.
    """

    t = torch.linspace(0., self.t_traj, self.ref_traj_steps, device=self.device)
    
    v_start = torch.zeros_like(p_start ,device=self.device)
    v_end = torch.zeros_like(p_end ,device=self.device)

    a0 = p_start
    a1 = v_start * self.t_traj
    a2 = torch.zeros_like(p_start ,device=self.device)

    # TODO: self.t_traj is a scalar, need to make it a tensor
    # TODO: even if self.t_traj is a tensor, A is not a 3x3 matrix?
    t_traj = torch.tensor([self.t_traj], device=self.device)
    A = torch.tensor([[t_traj**3, t_traj**4, t_traj**5],
                      [3*t_traj**2, 4*t_traj**3, 5*t_traj**4],
                      [6*t_traj, 12*t_traj**2, 20*t_traj**3]
                     ], device=self.device).unsqueeze(0).repeat(p_start.shape[0], 1, 1)  # Shape: (batch_size, 3, 3)
    B = torch.stack([
        p_end - (a0 + a1 + a2),
        v_end - (a1 + 2*a2*t_traj),
        torch.zeros_like(p_start ,device=self.device)  # Zero acceleration at the end
    ], dim=-1)  # Shape: (batch_size, 3)

    A_inv = torch.inverse(A)  # Shape: (batch_size, 3, 3)
    coeffs = torch.bmm(A_inv, B.unsqueeze(-1)).squeeze(-1)  # coeffs = (a3, a4, a5)
    a3, a4, a5 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]

    position = (a0.unsqueeze(1) + a1.unsqueeze(1)*t + a2.unsqueeze(1)*t**2 + a3.unsqueeze(1)*t**3 + a4.unsqueeze(1)*t**4 + a5.unsqueeze(1)*t**5)

    position = torch.clamp(position, p_min, p_max)
    return position
  
  def render(self, mode=None, done=None, save_dir=None, file_name=None, best_episode_only=False):
    """
      Render the current environment state and draw the end-effector trajectory.
    """
    assert self.n_envs == 1, "Render is not supported for multiple environments"
    if mode is None:
      if not self.is_rendering:
        self.is_rendering = True
        self.episode_reward = 0
        self.joint_pos_history = [self.dof_pos_history.squeeze(0).cpu().numpy()[:, 0]]
        self.joint_vel_history = [self.dof_vel_history.squeeze(0).cpu().numpy()[:, 0]]
        self.predicted_trajectory = [self.ee_pos.squeeze(0).cpu().numpy()]
        self.predicted_velocity = [self.ee_vel.squeeze(0).cpu().numpy()]
        self.target_trajectory = self.ref_traj_eepos.clone().cpu().numpy().squeeze(0)
        self.target_joint_pos = self.ref_traj_joint.clone().cpu().numpy().squeeze(0)
      else:
        self.predicted_trajectory.append(self.ee_pos.squeeze(0).cpu().numpy())
        self.predicted_velocity.append(self.ee_vel.squeeze(0).cpu().numpy())
        self.joint_pos_history.append(self.dof_pos_history.squeeze(0).cpu().numpy()[:, 0])
        self.joint_vel_history.append(self.dof_vel_history.squeeze(0).cpu().numpy()[:, 0])
        self.episode_reward += self._compute_reward().squeeze(0).cpu().numpy()

      # TODO: self.done may broke render as env may be truncated. Consider passing done as an argument from outside
      if done:
        self.is_rendering = False
        self.predicted_trajectories.append(np.array(self.predicted_trajectory))
        self.predicted_velocities.append(np.array(self.predicted_velocity))
        self.joint_pos_histories.append(np.array(self.joint_pos_history))
        self.joint_vel_histories.append(np.array(self.joint_vel_history))
        self.reward_histories.append(self.episode_reward)
        self.target_trajectories.append(np.array(self.target_trajectory))
        self.target_joint_pos_histories.append(np.array(self.target_joint_pos))
    elif mode == 'plot':
      assert save_dir is not None or file_name is not None, "Please provide save dir and file name"
      self.plot(save_dir, file_name, best_episode_only)
    elif mode == 'visualize':
      assert save_dir is not None or file_name is not None, "Please provide save dir and file name"
      self.visualize(save_dir, file_name)

  def plot(self, save_dir, plot_name, best_episode_only=False):
    """
      Plot the end-effector trajectory and the target trajectory.
    """
    assert not self.is_rendering, "Render is not done. Did you call render() after every step() and reset()?"
    if best_episode_only:
      best_episode = np.argmax(self.reward_histories)
    n_trajs = len(self.predicted_trajectories)
    n_steps = self.ref_traj_steps

    if best_episode_only:
      tracking_error = np.empty((1, n_steps))
      max_vels = np.empty((1, n_steps))
    else:
      tracking_error = np.empty((n_trajs, n_steps))
      max_vels = np.empty((n_trajs))

    fig = plt.figure()
    plt.clf()
    for i_traj in range(n_trajs):
      if best_episode_only and i_traj != best_episode:
        continue
      ee_pos = self.predicted_trajectories[i_traj]
      ee_pos_ref = self.target_trajectories[i_traj]
      tracking_error[i_traj] = np.linalg.norm(ee_pos_ref - ee_pos, axis=1)
      max_vels[i_traj] = np.max(np.linalg.norm(np.diff(ee_pos, axis=0)*25, axis=1))
      plt.plot(ee_pos[:, 0], ee_pos[:, 2], 'k', label='ee_traj')
      plt.plot(ee_pos[0, 0], ee_pos[0, 2], 'k.')
      plt.plot(ee_pos_ref[:, 0], ee_pos_ref[:, 2], 'r', alpha=0.5, label='ee_traj_ref')
      plt.plot(ee_pos_ref[0, 0], ee_pos_ref[0, 2], 'r.')
      if i_traj == 0:
        plt.legend()

    heap_img_path = os.path.expanduser(os.path.join(self.m545_rsc_dir, 'heap_render.png'))
    if os.path.exists(heap_img_path):
      heap_img = plt.imread(heap_img_path)
      img_center = [0.85, 1.95]
      img_scale = 0.0045
      extent = [img_center[0] - img_scale * heap_img.shape[1], img_center[0] + img_scale * heap_img.shape[1], img_center[1] + img_scale * heap_img.shape[0], img_center[1] - img_scale * heap_img.shape[0]]
      plt.imshow(heap_img, aspect='auto', origin='lower', extent=extent, alpha=0.25)
    plt.xlim([0, 8])
    plt.ylim([-1, 7])
    plt.axis('equal')
    plt.grid()

    # Save the plot instead of showing it
    plt.savefig(os.path.join(save_dir, f'{plot_name}.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

    print("Mean err: ", np.mean(np.power(tracking_error, 1)))
    print("MSRE: ", np.sqrt(np.mean(np.power(tracking_error, 2))))
    print("MSE: ", np.mean(np.power(tracking_error, 2)))
    print("Max err: ", np.max(tracking_error))
    print("Max vels: ", np.max(max_vels))
    print("Mean max vels: ", np.mean(max_vels))
  
  def cur_joints(self):
    return (self.dof_pos_history.squeeze(0).cpu().numpy()[:, 0],
            self.dof_vel_history.squeeze(0).cpu().numpy()[:, 0])
  
  def visualize(self, save_dir, video_name):
    """
    Render the current environment state and draw the end-effector trajectory.
    """

    assert not self.is_rendering, "Render is not done. Did you call render() after every step() and reset()?"

    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    import imageio

    # Load the URDF model, including collision and visual models
    self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(str(self.m545_urdf_path), str(self.mesh_dir))
    self.data = self.model.createData()
    base_frame_id = self.model.getFrameId("BASE_inertia")

    # Define the colors for the trajectories
    mesh_color = [0.929, 0.753, 0.106, 1] # Gold color
    target_color = 0xff2d00  # Red
    predicted_color = 0x00ff04 # Green
    planned_color = 0x0000ff # Blue

    # print(type(self.model))
    # print(type(self.collision_model))
    # print(type(self.visual_model))

    # Initialize Meshcat Visualizer
    self.viewer = AnimeMeshcatVisualizer(self.model, self.collision_model, self.visual_model)
    self.viewer.initViewer(open=True)
    self.viewer.loadViewerModel(rootNodeName="heap", color=mesh_color) 
    self.viewer.displayCollisions(False)
    self.viewer.displayVisuals(True)
    self.viewer.displayFrames(True)
    self.viewer.viewer["/Cameras/default"].set_transform(tf.translation_matrix([5, 8, 2]))

    # Create an animation object
    anim = meshcat.animation.Animation()
    frame_idx = 0

    # Find the episode with the highest reward
    best_episode = np.argmax(self.reward_histories)
    best_reward = np.float32(self.reward_histories[best_episode])
    
    # Prepare the target and predicted trajectories
    target_trajectory = self.target_trajectories[best_episode] # shape (n_traj_steps, 3)
    predicted_trajectory = self.predicted_trajectories[best_episode] # shape (n_traj_steps, 3)
    
    predicted_velocity = self.predicted_velocities[best_episode] # shape (n_traj_steps, 6)
    joint_positions = self.joint_pos_histories[best_episode] # shape (n_traj_steps, act_dim)
    joint_velocities = self.joint_vel_histories[best_episode] # shape (n_traj_steps, act_dim)
    target_joint_pos = self.target_joint_pos_histories[best_episode] # shape (act_dim, n_traj_steps)

    # TODO: planned_trajectory = 

    # Extract the initial joint positions from the target trajectory
    try:
      initial_joint_positions = target_joint_pos[:, 0] # Assuming the first set of joint positions corresponds to the start
    except Exception as e:
      print("Error extracting initial joint positions from the target trajectory. Are you sure the environment is initialized and reset?")
      print(e)
      return

    # Set the initial state of the model
    with anim.at_frame(self.viewer.viewer, frame_idx) as frame:
      self.viewer.display(initial_joint_positions, animation_frame=frame)
    frame_idx += 1
    self.viewer.display(initial_joint_positions)
    pin.forwardKinematics(self.model, self.data, initial_joint_positions)
    pin.updateFramePlacements(self.model, self.data)

    # print("Frames in the model:")
    # for idx, frame in enumerate(self.model.frames):
    #     print(f"Frame {idx}: {frame.name}")
    #     base_to_frame_transform = self.data.oMf[idx].homogeneous
    #     print(f"Frame {frame.name}: {base_to_frame_transform}")
    
    # Define the transformation matrix from the base coordinate to the new coordinate
    base_frame_id = self.model.getFrameId("BASE_inertia")
    world_to_base_transform = self.data.oMf[base_frame_id].homogeneous

    # print(f"World to frame transform: {world_to_frame_transform}")
    # Compute the transformation from base to world
    # base_to_world_transform = np.linalg.inv(world_to_frame_transform)
    # Apply the transformation to the viewer
    # self.viewer.viewer["heap"].set_transform(base_to_world_transform)

    # Draw the trajectories in Meshcat using g.Line
    if len(target_trajectory) > 1:
        # Convert to shape (3, N)
        transposed_target_trajectory = target_trajectory.T  
        transposed_predicted_trajectory = predicted_trajectory.T  

         # Visualize the target trajectory
        self.viewer.viewer["target_trajectory"].set_object(
            g.Line(g.PointsGeometry(transposed_target_trajectory), g.MeshBasicMaterial(color=target_color, linewidth=100000))
        )
        self.viewer.viewer["target_trajectory"].set_transform(world_to_base_transform)


        # Visualize the predicted trajectory
        self.viewer.viewer["predicted_trajectory"].set_object(
            g.Line(g.PointsGeometry(transposed_predicted_trajectory), g.MeshBasicMaterial(color=predicted_color, linewidth=100000))
        )
        self.viewer.viewer["predicted_trajectory"].set_transform(world_to_base_transform)

    
    # Draw the robot following the predicted trajectory
    static_frames = []
    for i in range(0, len(predicted_trajectory)):
        q = joint_positions[i, :] 
        v = joint_velocities[i, :] 

        # Add a keyframe to the animation
        with anim.at_frame(self.viewer.viewer, frame_idx) as frame:
            self.viewer.display(q, animation_frame=frame)
        frame_idx += 1

        # Capture the frame
        if self.cfg.save_heap_video:
          self.viewer.display(q)
          static_frame = self.viewer.captureImage()  # Capture the frame
          static_frames.append(static_frame)
    
    # Set the animation to the viewer
    self.viewer.viewer.set_animation(anim, play=False)

    # Save the animation as a video
    if self.cfg.save_heap_video:
      print("Saving the animation as a video... Total frames: ", len(static_frames))
      imageio.mimsave(os.path.join(save_dir, f"{video_name}-{best_reward:.02f}.mp4"), static_frames, fps=30)

    # time.sleep(10)  # Wait for the viewer to load

    print("Rendering is done!")

