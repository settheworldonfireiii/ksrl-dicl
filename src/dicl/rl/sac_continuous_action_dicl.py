import warnings
from typing import List, Dict, Any, Union
import os
import random
import time
from dataclasses import dataclass
import copy
from tqdm import tqdm
import json
import csv

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

from stable_baselines3.common.buffers import ReplayBuffer

import gymnasium as gym
from gymnasium import spaces

from transformers import LlamaForCausalLM, AutoTokenizer

from dicl import dicl

from typing import Any, Callable, Optional, Union


from stable_baselines3.common.type_aliases import ReplayBufferSamples


import pdb

from stable_baselines3.common.vec_env import VecNormalize


from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)




import os
print(os.getcwd())

import tensorflow.compat.v1 as tf
from .tf_models.constructor import construct_shallow_model, construct_shallow_cost_model, construct_model, construct_cost_model

#import ksdp
from .ksdp import *
from .ksdp import utils
from .ksdp import ksd

from .NB_dx_tf_new import neural_bays_dx_tf




def randperm_in_range(n, m):
  """
  Generates a tensor of n unique random integers in the range [0, m-1].

  Args:
    n: The number of random integers to generate.
    m: The upper bound of the range (exclusive).

  Returns:
    A tensor of shape (n,) containing a random permutation of integers from 0 to m-1.
  """
  if n > m:
    raise ValueError("n cannot be greater than m")

  return torch.randint(m, (n,), generator=torch.Generator().manual_seed(torch.seed()))





def merge_and_shuffle_samples(
    data: ReplayBufferSamples,
    data_llm: ReplayBufferSamples
) -> ReplayBufferSamples:
    """
    Concatenate two ReplayBufferSamples and randomly shuffle in unison, all on torch tensors.
    """
    # 1) Concatenate each tensor field along dim=0:
    obs            = torch.cat([data.observations,    data_llm.observations],    dim=0)  # :contentReference[oaicite:0]{index=0}
    next_obs       = torch.cat([data.next_observations, data_llm.next_observations], dim=0)  # :contentReference[oaicite:1]{index=1}
    actions        = torch.cat([data.actions,         data_llm.actions],         dim=0)  # :contentReference[oaicite:2]{index=2}
    rewards        = torch.cat([data.rewards,        data_llm.rewards],        dim=0)  # :contentReference[oaicite:3]{index=3}
    dones          = torch.cat([data.dones,          data_llm.dones],          dim=0)  # :contentReference[oaicite:4]{index=4}

    # infos is a Python list of dicts; just extend it in CPU memory
    #infos = data.infos + data_llm.infos  # :contentReference[oaicite:5]{index=5}

    # 2) Build a random permutation on the same device as your tensors:
    device = obs.device
    maxim  = obs.shape[0]
    total = data.observations.shape[0]
    #perm   = torch.randperm(total, device=device)  # :contentReference[oaicite:6]{index=6}
    perm = randperm_in_range(total, maxim)
    # 3) Re‐index (shuffle) each tensor using that same permutation:
    obs_s      = obs[perm]           # :contentReference[oaicite:7]{index=7}
    next_obs_s = next_obs[perm]      # :contentReference[oaicite:8]{index=8}
    acts_s     = actions[perm]       # :contentReference[oaicite:9]{index=9}
    rews_s     = rewards[perm]       # :contentReference[oaicite:10]{index=10}
    dones_s    = dones[perm]         # :contentReference[oaicite:11]{index=11}
    #infos_s    = [infos[i] for i in perm.tolist()]  # :contentReference[oaicite:12]{index=12}

    # 4) Wrap back into ReplayBufferSamples:
    return ReplayBufferSamples(
        observations=obs_s,
        next_observations=next_obs_s,
        actions=acts_s,
        rewards=rews_s,
        dones=dones_s
        )

    


try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances
        (check out `videos` folder)"""
    path: str = "."
    """the path where the experiment will be saved"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    #learning_starts: int = 1
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Custom
    save_policy_checkpoints: int = 20000
    """frequency of saving policy checkpoints"""
    act_deterministically: bool = False
    """whether to act deterministically"""
    # algo logic
    interact_every: int = 1
    """frequency of policy-environment interactions (update frequency)"""
    # icl
    llm_model: str = "meta-llama/Llama-3.1-8B"
    """the LLM used for in-context learning"""
    method: str = "vicl"
    """the method used for in-context learning"""
    dicl_pca_n_components: int = -1
    """number of components for PCA in DICL"""
    context_length: int = 300
    """length of the context"""
    rescale_factor: float = 7.0
    """factor to rescale the input"""
    up_shift: float = 1.5
    """value to shift the input upwards"""
    llm_learning_starts: int = learning_starts * 2
    """timestep to start learning for the LLM"""
    llm_learning_frequency: int = 32
    """frequency of learning for the LLM"""
    llm_batch_size: int = 25
    """batch size for the LLM"""
    train_only_from_llm: bool = False
    """whether to train only from the LLM"""
    min_episodes_to_start_icl: int = 5
    """minimum number of episodes to start in-context learning"""
    burnin_llm: int = 0
    """number of burn-in steps for the LLM"""
    add_init_burin_steps_to_llm: bool = False
    """whether to add initial burn-in steps to the LLM"""
    llm_percentage_to_keep: int = 100
    """percentage of LLM data to keep"""
    auxiliary_actions: bool = True
    """whether to use auxiliary actions"""


class TruncReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        handle_auxiliary_actions: bool = False,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination
        # are true, see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            dtype=observation_space.dtype,
        )

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = np.zeros(
                (self.buffer_size, self.n_envs, *self.obs_shape),
                dtype=observation_space.dtype,
            )

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.handle_auxiliary_actions = handle_auxiliary_actions
        self.auxiliary_actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype
        )
        self.batch_inds = None
        if psutil is not None:
            total_memory_usage = (
                self.observations.nbytes
                + self.actions.nbytes
                + self.rewards.nbytes
                + self.dones.nbytes
            )

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the "
                    "complete replay buffer {total_memory_usage:.2f}GB > "
                    f"{mem_available:.2f}GB"
                )
    
    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        self.batch_inds = batch_inds
        return self._get_samples(batch_inds, env=env)

    def sample_indices(batch_size):
        self.sample()
        return self.batch_inds
    
    
    def sample_consec(self, batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.arange(1, self.buffer_size, sibatch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.arange(0, self.pos, size=batch_size)
        self.batch_inds = batch_inds
        return self._get_samples(batch_inds, env=env)

        
        
    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(
                next_obs
            ).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array(infos["truncations"]).copy()

        # auxiliary actions
        if self.handle_auxiliary_actions:
            self.auxiliary_actions[self.pos] = np.array(
                infos["auxiliary_actions"]
            ).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
    
    
    def delete(self, pos: List[int], shift: Optional[bool] = None,) -> None:
        to_keep = []
        for i in range(self.pos):
            if i not in pos:
                to_keep.append(i)
        for i in range(len(to_keep)):
            self.observations[i] = self.observations[to_keep[i]]

            self.next_observations[i] = self.next_observations[to_keep[i]]
            self.actions[i] = self.actions[to_keep[i]]
            self.rewards[i] = self.rewards[to_keep[i]]
            self.dones[i] = self.dones[to_keep[i]]
            if self.handle_timeout_termination:
                self.timeouts[i] = self.timeouts[to_keep[i]]
            if self.handle_auxiliary_actions:
                self.auxiliary_actions[i] = self.auxiliary_actions[to_keep[i]]

        for i in range(len(to_keep), self.pos, 1):
            self.observations[i] = np.zeros_like(self.observations[i]) 

            self.next_observations[i] =  np.zeros_like(self.next_observations[i])
            self.actions[i] =  np.zeros_like(self.actions[i])
            self.rewards[i] =  np.zeros_like(self.rewards[i])
            self.dones[i] =  np.zeros_like(self.dones[i])
            if self.handle_timeout_termination:
                self.timeouts[i] =  np.zeros_like(self.timeouts[i])
            if self.handle_auxiliary_actions:
                self.auxiliary_actions[i] =  np.zeros_like(self.auxiliary_actions[i])
        self.pos -= len(pos)



                




class CSVLogger:
    def __init__(self, filename, fieldnames, write_frequency=1):
        self.filename = filename
        self.fieldnames = fieldnames
        self.write_frequency = write_frequency
        self.buffer = []
        self.write_header()
        self.counter = 0

    def write_header(self):
        if not os.path.exists(self.filename):
            with open(self.filename, mode="w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, data):
        self.counter += 1
        self.buffer.append(data)
        if self.counter == self.write_frequency:
            self.flush()
            self.counter = 0

    def flush(self):
        if self.buffer:
            with open(self.filename, mode="a", newline="") as csv_file:
                print(csv_file, " FILE WHERE WE SAVE IT")
                writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
                writer.writerows(self.buffer)
            self.buffer = []


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.action_space.high - env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.action_space.high + env.action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def main():
    erstens = True
    args = tyro.cli(Args)
    tf.disable_v2_behavior()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"{args.path}/runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    csv_logger = CSVLogger(
        f"{args.path}/runs/{run_name}/logs.csv",
        fieldnames=["global_step", "return"],
        write_frequency=1,
    )
    # save args
    with open(f"{args.path}/runs/{run_name}/args.json", "w") as fout:
        json.dump(args.__dict__, fout, indent=4)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"
    actor = Actor(envs)
    #actor.load_state_dict(torch.load('runs/HalfCheetah__test_5p_vicl__494__1743916116/actor_checkpoint_1000000.pth', weights_only=True))
    
    actor = actor.to(device)
    #print(actor.eval())
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = TruncReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
        handle_auxiliary_actions=True,
    )
    rb_llm = TruncReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        handle_auxiliary_actions=False,
    )
    start_time = time.time()

    # ------------------------------ load model and tokenizer --------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model,
        use_fast=False,
    )
    model = LlamaForCausalLM.from_pretrained(
        args.llm_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    # ----------------------------------------------------------------------------------

    # ----------- define n_observations and n_actions -----------
    n_observations = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.shape[0]
    #pdb.set_trace()
    dx_model = construct_shallow_model(obs_dim=n_observations, act_dim=action_shape, hidden_dim=200, num_networks=1, num_elites=1)
    print("BEFORE NEURAL BAYS")
    my_dx = neural_bays_dx_tf(args, dx_model, "dx", n_observations, sigma_n2=1e-3**2,sigma2=1e1**2)


    #print(f'n_observations, whatever it is, {n_observations}')
    # other counters
    started_sampling = False
    step_started_sampling = 0

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    episode_step = 0
    global_step = 0
    pbar = tqdm(total=args.total_timesteps)
    list_model_order = []
    posterior_sigma = []
    while global_step <= args.total_timesteps:
        # SAVE ACTOR CHECKPOINTS
        if global_step % args.save_policy_checkpoints == 0:
            torch.save(
                actor.state_dict(),
                f"{args.path}/runs/{run_name}/actor_checkpoint_{global_step}.pth",
            )

        # ------- This is interaction with environment -------
        if global_step % 1000 == 0:
            #PROBABLY WILL PUT INTO SOME OTHER PLACE
            my_dx.sample()
            


        if global_step % args.interact_every == 0:
            for _ in range(args.interact_every):
                # ALGO LOGIC: put action logic here
                
                if global_step < args.learning_starts:
                    actions = np.array(
                        [
                            envs.single_action_space.sample()
                            for _ in range(envs.num_envs)
                        ]
                    )
                    auxiliary_actions = copy.copy(actions)
                else:
                    actions, _, mean_actions = actor.get_action(
                        torch.Tensor(obs).to(device)
                    )
                    if args.act_deterministically:
                        actions = mean_actions.detach().cpu().numpy()
                    else:
                        actions = actions.detach().cpu().numpy()
                    auxiliary_actions, _, _ = actor.get_action(
                        torch.Tensor(obs).to(device)
                    )
                    auxiliary_actions = auxiliary_actions.detach().cpu().numpy()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)
                infos["truncations"] = truncations
                infos["auxiliary_actions"] = auxiliary_actions
                global_step += 1
                pbar.update(1)

                
                """
                pdb.set_trace()
                xu = torch.cat((torch.tensor(obs.squeeze()).double(), torch.tensor(actions.squeeze()).double()))
                #print("XUXUXU ", xu
                pdb.set_trace()
                shappe = my_dx.add_data(new_x=xu, new_y=torch.tensor(next_obs).squeeze() - torch.tensor(obs).squeeze(), new_r = torch.tensor(rewards).squeeze(0))
                my_dx.train(100)
                """


                # TRY NOT TO MODIFY: record rewards for plotting purposes
                if "final_info" in infos:
                    episode_step = 0
                    for info in infos["final_info"]:
                        print(
                            f"global_step={global_step}, "
                            f"episodic_return={info['episode']['r']}, "
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        # Log to CSV
                        csv_logger.log(
                            {
                                "global_step": global_step,
                                "return": float(info["episode"]["r"]),
                            }
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )
                        writer.add_scalar(
                            "charts/replay_buffer_size", rb.pos, global_step
                        )
                        writer.add_scalar(
                            "charts/llm_replay_buffer_size", rb_llm.pos, global_step
                        )
                        break

                # TRY NOT TO MODIFY: save data to rb; handle `final_observation`
                real_next_obs = next_obs.copy()
                for idx, trunc in enumerate(truncations):
                    if trunc:
                        print(infos)
                        real_next_obs[idx] = infos["final_observation"][idx]
                #pdb.set_trace()
                rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
                if (
                    episode_step < args.burnin_llm
                ) and args.add_init_burin_steps_to_llm:
                    #pdb.set_trace()
                    rb_llm.add(
                        obs, real_next_obs, actions, rewards, terminations, infos
                    )
                episode_step += 1

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs

        # ALGO LOGIC: training.
        ksd_val = 0
        if global_step > args.learning_starts:
            local_step = 0
            #pdb.set_trace()
            for _ in range(args.interact_every):
                # ------- sample from real replay buffer --------
                #  vim /scratch.global/radke149/dicl/lib/python3.9/site-packages/stable_baselines3/common/buffers.py 
                data = rb.sample(args.batch_size)
                #pdb.set_trace()
                # ------- Data Augmentation using LLM -------
                # 1. Generate transformed transition
                # 1.1. Sample sub-trajectory of length 'context_length' from rb
                where_dones = np.where(np.logical_or(rb.dones, rb.timeouts))[0]
                starts = np.concatenate([np.array([0]), where_dones + 1], axis=0)
                endings = np.concatenate([where_dones, np.array([rb.pos - 1])], axis=0)
                episode_lengths = endings - starts
                possible_episodes = starts[
                    np.argwhere(episode_lengths > args.context_length + 1)
                ]
                possible_episodes_endings = endings[
                    np.argwhere(episode_lengths > args.context_length + 1)
                ]
                
                if ((global_step + local_step) % args.llm_learning_frequency == 0) and (
                    len(possible_episodes) >= args.min_episodes_to_start_icl
                ):
                    if not started_sampling:
                        started_sampling = True
                        step_started_sampling = copy.copy((global_step + local_step))
                        with open(
                            f"{args.path}/runs/{run_name}/icl_started.txt", "w"
                        ) as f:
                            f.write(f"icl started at: {(global_step + local_step)}")
                        print(
                            f"---------- icl started at: {(global_step + local_step)} "
                            "-----------"
                        )
                    random_idx = np.random.randint(0, len(possible_episodes))
                    start_episode = int(possible_episodes[random_idx])
                    start_index = int(
                        np.random.randint(
                            start_episode,
                            possible_episodes_endings[random_idx]
                            - args.context_length
                            - 1,
                        )
                    )
                    # 1.2. Do ICL
                    if args.method == "vicl":
                        time_series = rb.observations[
                            start_index : start_index + args.context_length
                        ]
                        time_series = time_series.reshape((args.context_length, -1))
                        DICL = dicl.vICL(
                            n_features=time_series.shape[1],
                            model=model,
                            tokenizer=tokenizer,
                            rescale_factor=args.rescale_factor,
                            up_shift=args.up_shift,
                        )
                    elif args.method == "dicl_s_pca":
                        time_series = rb.observations[
                            start_index : start_index + args.context_length
                        ]
                        time_series = time_series.reshape((args.context_length, -1))
                        DICL = dicl.DICL_PCA(
                            n_components=time_series.shape[1]
                            if args.dicl_pca_n_components == -1
                            else args.dicl_pca_n_components,
                            n_features=time_series.shape[1],
                            model=model,
                            tokenizer=tokenizer,
                            rescale_factor=args.rescale_factor,
                            up_shift=args.up_shift,
                        )
                    elif args.method == "dicl_sa_pca":
                        #print("DICL SA")
                        time_series = np.concatenate(
                            [
                                rb.observations[
                                    start_index : start_index + args.context_length
                                ],
                                rb.actions[
                                    start_index : start_index + args.context_length
                                ],
                            ],
                            axis=-1,
                        )
                        time_series = time_series.reshape((args.context_length, -1))
                        DICL = dicl.DICL_PCA(
                            n_components=time_series.shape[1]
                            if args.dicl_pca_n_components == -1
                            else args.dicl_pca_n_components,
                            n_features=time_series.shape[1],
                            model=model,
                            tokenizer=tokenizer,
                            rescale_factor=args.rescale_factor,
                            up_shift=args.up_shift,
                        )

                    DICL.fit_disentangler(X=time_series)
                    mean, mode, lb, ub = DICL.predict_single_step(X=time_series)

                    # compute threshold on the true error
                    all_groundtruth = rb.next_observations[
                        start_index : start_index + args.context_length - 1
                    ]

                    true_errors = np.linalg.norm(
                        all_groundtruth.squeeze() - mean[:, :n_observations], axis=1
                    )

                    sorted_indices = true_errors.argsort()
                    n_to_keep = int(
                        args.llm_percentage_to_keep * len(true_errors) / 100
                    )

                    for t in range(args.burnin_llm, args.context_length):
                        if t in sorted_indices[:n_to_keep]:
                            # 1.3. New transition created by llm prediction
                            llm_next_obs = mean[t, :n_observations].reshape((1, -1))
                            llm_obs = time_series[t, :n_observations].reshape((1, -1))
                            llm_actions = rb.actions[start_index + t].reshape((1, -1))
                            llm_rewards = rb.rewards[start_index + t].reshape((1,))
                            llm_terminations = np.zeros((1,))

                            # 2. Append transformed transition to augmented rb
                            #pdb.set_trace()
                            rb_llm.add(
                                llm_obs,
                                llm_next_obs,
                                rb.auxiliary_actions[start_index + t]
                                if args.auxiliary_actions
                                else llm_actions,
                                llm_rewards,
                                llm_terminations,
                                {},  # llm_infos,
                            )
                coeff_batches_to_train_on = [1.0]
                batches_to_train_on = [copy.copy(data)]
                #can do some decaying schedule instead
                if (global_step + local_step)%500 == 0:                
                    for i in range(batches_to_train_on[0].observations.shape[0]):
                        #pdb.set_trace()
                        xu = torch.cat((torch.tensor(tf.get_static_value(batches_to_train_on[0].observations[i].squeeze().cpu())).double(), torch.tensor(tf.get_static_value(batches_to_train_on[0].actions[i].squeeze().cpu())).double()))
                        #print("XUXUXU ", xu)
                        #pdb.set_trace()
                        shappe = my_dx.add_data(new_x=xu, new_y=torch.tensor(tf.get_static_value(batches_to_train_on[0].next_observations[i].cpu())).squeeze() - torch.tensor(tf.get_static_value(batches_to_train_on[0].observations[i].cpu())).squeeze(), new_r = torch.tensor(tf.get_static_value(batches_to_train_on[0].rewards[i].cpu())).squeeze(0))
                if (global_step + local_step)%500 == 0:
                    """
                        my_dx.train(100)
                        print("TRAINED LIKE IN AKHMAT!")
                        post_var = my_dx.update_bays_reg()
                        print (np.trace(post_var[0]))

                        posterior_sigma.append(np.trace(post_var[0]))
                        print("POSTERIOR SIGMA ", posterior_sigma)
                        ksd_val = my_dx.get_ksd('ksd')
                    
                    
                    if (global_step+local_step) % 1000 == 0:
                        print("GLOBAL STEP ", global_step)
                        print("LOCAL STEP ", local_step)
                        pdb.set_trace()
                        my_dx.train(100)
                        post_var = my_dx.update_bays_reg()
                        ksd_val = my_dx.get_ksd('ksd')
                        print("KSD VAL", ksd_val)
                    """
                    print("GLOBAL STEP ", global_step)
                    print("LOCAL STEP ", local_step)
                    my_dx.train(100)
                    #pdb.set_trace()
                    #print("TRAINED LIKE IN AKHMAT!")
                
                    post_var = my_dx.update_bays_reg()
                    #print (np.trace(post_var[0]))
                      
                    posterior_sigma.append(np.trace(post_var[0]))
                    #print("POSTERIOR SIGMA ", posterior_sigma)
                    ksd_val = my_dx.get_ksd('ksd')
                    print("KSD VAL", ksd_val)
                    #coeff_batches_to_train_on = [1.0]
                    
                # 3. Sample from rb and transformed_rb to train ActorCritic
                
                if (
                    (global_step + local_step)
                    > args.llm_learning_starts
                    - args.learning_starts + step_started_sampling
                ) and started_sampling:
                    #INCREASE THE LLM BATCH SIZE
                    data_llm = rb_llm.sample(args.llm_batch_size*7)
                    # concatenate data and data_llm
                    if args.train_only_from_llm:
                        # data = data_llm
                        batches_to_train_on = [copy.copy(data_llm)]
                        coeff_batches_to_train_on = [1.0]
                    else:
                        print("BATCHES TO TRAIN ON ", batches_to_train_on)
                        #pdb.set_trace()
                        """
                        batches_to_train_on.append(copy.copy(data_llm))
                        
                        coeff_batches_to_train_on.append(
                            float(args.llm_batch_size / args.batch_size)
                        )
                        """
                        if ((global_step + local_step) % 25 == 0) or erstens:
                              
                            for i in range(data_llm.observations.shape[0]):

                                xu = torch.cat((torch.tensor(tf.get_static_value(data_llm.observations[i].squeeze().cpu())).double(), torch.tensor(tf.get_static_value(data_llm.actions[i].squeeze().cpu())).double()))
                                print("XUXUXU ", xu)
                                #pdb.set_trace()
                                shappe = my_dx.add_data(new_x=xu, new_y=torch.tensor(tf.get_static_value(data_llm.next_observations[i].cpu())).squeeze() - torch.tensor(tf.get_static_value(data_llm.observations[i].cpu())).squeeze(), new_r = torch.tensor(tf.get_static_value(data_llm.rewards[i].cpu())).squeeze(0), real = False)
                                print("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS")
                                #batches_to_train_on = [merge_and_shuffle_samples(data,data_llm)] 
                        # 71
                        if ((global_step + local_step) % 25  == 0) or erstens:
                            print("GLOBAL STEP ", global_step)
                            print("LOCAL STEP ", local_step)
 

                            print("TRAINED LIKE IN AKHMAT!")
                            """
                            post_var = my_dx.update_bays_reg()
                            print (np.trace(post_var[0]))

                            posterior_sigma.append(np.trace(post_var[0]))
                            """
                            my_dx.generate_latent_z(False)
                            print("POSTERIOR SIGMA ", posterior_sigma)
                            ksd_val_s = my_dx.get_ksd('ksd', False)
                            print("KSD VAL FAKE", ksd_val_s)
                            erstens = False
                            
                            indices = my_dx.thin_data_new("ksd", False)
                            #pdb.set_trace()
                            inds = torch.cat(indices, dim=0).cpu().numpy()
                            obs_l = data_llm.observations[inds] 
                            next_obs_l = data_llm.next_observations[inds] 
                            actions_l = data_llm.actions[inds]
                            rewards_l  = data_llm.rewards[inds] 
                            dones_l = data_llm.dones[inds]
                            data_llm = ReplayBufferSamples(
                                observations=obs_l,
                                next_observations=next_obs_l,
                                actions=actions_l,
                                rewards=rewards_l,
                                dones=dones_l
                                )
                            batches_to_train_on.append(copy.copy(data_llm))
                            
                            coeff_batches_to_train_on.append(float(args.llm_batch_size / args.batch_size))


                # --------------------------------------------
                iterator = 0
            
                for data, p in zip(batches_to_train_on, coeff_batches_to_train_on):
                    #print(f'local_step {local_step} global step {global_step} target network frequency {args.target_network_frequency} llm learning frequency {args.llm_learning_frequency} policy frequency {args.policy_frequency}')
                    #pdb.set_trace()
                    iterator += 1
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, next_state_mean = actor.get_action(
                            data.next_observations
                        )
                        

                        #Laplace approximation 
                        def logprob_sum(stat):
                            mean, std = stat
                            return (torch.distributions.Normal(mean, std)).logprob(mean).sum()
                        """
                        next_state_pi = next_state_log_pi.exp()
                        tupleparam = [next_state_mean, next_state_pi]
                        hess = torch.autograd.functional.hessian(logprob_sum, tupleparam)[0][0]
                        cov = torch.inverse(hess + 1e-4 * torch.eye(hess.size(-1)))
                        """
                        qf1_next_target = qf1_target(
                            data.next_observations, next_state_actions
                        )
                        qf2_next_target = qf2_target(
                            data.next_observations, next_state_actions
                        )
                        min_qf_next_target = (
                            torch.min(qf1_next_target, qf2_next_target)
                            - alpha * next_state_log_pi
                        )
                        next_q_value = data.rewards.flatten() + (
                            1 - data.dones.flatten()
                        ) * args.gamma * (min_qf_next_target).view(-1)
                    #MOVE TO #3. Sample from rb ... AND GET KSD AND THIN THERE.
                    """
                    if (global_step+local_step)%1000 == 0:
                            print("GOT SHAPE!")
                            print("GLOBAL_STEP ", global_step)
                            print("LOCAL_STEP ", local_step)
                            my_dx.train(100)
                            print("TRAINED LIKE IN AKHMAT!")
                            post_var = my_dx.update_bays_reg()
                            print (np.trace(post_var[0]))

                            posterior_sigma.append(np.trace(post_var[0]))
                            print("POSTERIOR SIGMA ", posterior_sigma)
                            ksd_val = my_dx.get_ksd('ksd')
                            print("KSD val ", ksd_val)
                            print("ITERATOR ", iterator)
                            print("LOCAL STEP ", local_step)
                            to_del = my_dx.thin_data_new("ksd")
                            rb.delete(to_del)
                    """
        
                    qf1_a_values = qf1(data.observations, data.actions).view(-1)
                    qf2_a_values = qf2(data.observations, data.actions).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    if p < 1.0:
                        qf_loss = p * (qf1_loss + qf2_loss)/(ksd_val_s/10000)
                    else:
                        qf_loss = p * (qf1_loss + qf2_loss)/(ksd_val/1000)
                    print("QF LOSS ", qf_loss) 

                    """
                    KSD = 
                    qf1_a_values = qf1(data.observations + KSD, data.actions).view(-1) here KSD is a vector
                    qf2_a_values = qf2(data.observations + KSD, data.actions).view(-1) here KSD is a vector
                    
                    # OR
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value) + KSD #if added here, KSD is a scalar
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value) + KSD #if added here, KSD is a scalar
                    
                    #OR
                    qf_loss = p * (qf1_loss + qf2_loss) + KSD # if added here, KSD is a scalar

                    """

                    # optimize the model
                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    if (
                        global_step + local_step
                    ) % args.policy_frequency == 0:  # TD 3 Delayed update support
                        for _ in range(
                            args.policy_frequency
                        ):  # compensate for the delay by doing 'actor_update_interval'
                            pi, log_pi, _ = actor.get_action(data.observations)
                            qf1_pi = qf1(data.observations, pi)
                            qf2_pi = qf2(data.observations, pi)
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                            actor_loss = p * ((alpha * log_pi) - min_qf_pi).mean()

                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            actor_optimizer.step()

                            if args.autotune:
                                with torch.no_grad():
                                    _, log_pi, _ = actor.get_action(data.observations)
                                alpha_loss = (
                                    p
                                    * (
                                        -log_alpha.exp() * (log_pi + target_entropy)
                                    ).mean()
                                )

                                a_optimizer.zero_grad()
                                alpha_loss.backward()
                                a_optimizer.step()
                                alpha = log_alpha.exp().item()

                # update the target networks
                if (global_step + local_step) % args.target_network_frequency == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )

                if (global_step + local_step) % 100 == 0:
                    writer.add_scalar(
                        "losses/qf1_values", qf1_a_values.mean().item(), global_step
                    )
                    writer.add_scalar(
                        "losses/qf2_values", qf2_a_values.mean().item(), global_step
                    )
                    writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                    writer.add_scalar(
                        "losses/qf_loss", qf_loss.item() / 2.0, global_step
                    )
                    writer.add_scalar(
                        "losses/actor_loss", actor_loss.item(), global_step
                    )
                    writer.add_scalar("losses/alpha", alpha, global_step)
                    writer.add_scalar(
                        "charts/SPS",
                        int(global_step / (time.time() - start_time)),
                        global_step,
                    )
                    if args.autotune:
                        writer.add_scalar(
                            "losses/alpha_loss", alpha_loss.item(), global_step
                        )
                local_step += 1
        """
        if global_step % 1000 == 0 and xu is not None: 
            list_model_order.append(my_dx.get_shape())
            
            print("GOT SHAPE!")
            my_dx.train(100)
            print("TRAINED LIKE IN AKHMAT!")
            post_var = my_dx.update_bays_reg()
            print (np.trace(post_var[0]))
    
            posterior_sigma.append(np.trace(post_var[0]))
            print("POSTERIOR SIGMA ", posterior_sigma)
            ksd_val = my_dx.thin_data_new('ksd')
            print("KSD val ", ksd_val)
        """ 

    pbar.close()
    envs.close()
    writer.close()
    csv_logger.flush()


if __name__ == "__main__":
    main()
