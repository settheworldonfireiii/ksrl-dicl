import gymnasium as gym
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from gymnasium import Wrapper

import argparse
import ast

import torch

import numpy as np
import pandas as pd

from dicl.rl.sac_continuous_action_dicl import Actor, SoftQNetwork

from stable_baselines3.common.monitor import Monitor

from scipy.stats import kstest, ks_2samp



from stable_baselines3.common.evaluation import evaluate_policy



class ActorWrapper:
    def __init__(self, actor: torch.nn.Module, device: torch.device = None):
        """
        Wraps your Actor so it has a .predict() like SB3.

        Args:
            actor: instance of your Actor(nn.Module)
            device: torch.device (e.g. 'cuda' or 'cpu'); if None, inferred from actor parameters
        """
        self.actor = actor
        # infer device
        self.device = next(actor.parameters()).device if device is None else device
        self.actor.to(self.device)
        self.actor.eval()

    def predict(
        self,
        observations: np.ndarray,
        state: tuple = None,
        episode_start: np.ndarray = None,
        deterministic: bool = False,
    ):
        """
        Args:
            observations: np.ndarray of shape (n_envs, obs_dim)
            state: ignored (for compatibility with recurrent policies)
            episode_start: ignored
            deterministic: if True, use the mean action; otherwise, sample
        Returns:
            actions: np.ndarray of shape (n_envs, act_dim)
            state: same as input (None)
        """
        # 1) to tensor
        obs_tensor = torch.as_tensor(observations, device=self.device).float()
        # 2) get_action returns (action, logprob, mean) all torch tensors of shape [n_envs, act_dim]
        with torch.no_grad():
            actions_sample, _, means = self.actor.get_action(obs_tensor)

        # 3) choose deterministic or not
        out_actions = means if deterministic else actions_sample

        # 4) to numpy
        return out_actions.cpu().numpy(), state




def make_env_not_war(args):
    def _thunk():
        
        if args.render is not None:
            env = gym.make(args.env, render_mode=args.render)
        else:
            env = gym.make(args.env)
        
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(args.seed)

        return env
    return _thunk

def make_fixed_env_not_war(fixed_state, args):
    def thunk():
        raw = gym.make("HalfCheetah-v4")
        wrapped = FixedStartWrapper(raw, fixed_state)
        # wrap with Monitor, RecordVideo, etc., then seed
        env = Monitor(wrapped)
        env = gym.wrappers.RecordEpisodeStatistics(env)
 
        env.action_space.seed(args.seed)
        env.observation_space.seed(args.seed)
        return wrapped
    return thunk
   




class FixedStartWrapper(Wrapper):
    def __init__(self, env, fixed_state):
        super().__init__(env)
        self.fixed_state = np.array(fixed_state)


    def reset(self, *, seed=None, options=None):
        # 1) Let the base MujocoEnv (and any upstream wrappers) do their reset:
        obs, info = super().reset(seed=seed, options=options)

        # 2) Bypass all wrappers to get at the MuJoCoEnv instance
        base = self.unwrapped

        # 3) Split your 18‑dim fixed_state into qpos (nq) and qvel (nv)
        nq = base.model.nq
        qpos = self.fixed_state[:nq]
        qvel = self.fixed_state[nq : nq + base.model.nv]

        # 4) Use the built‑in setter (does mj_forward for you)
        base.set_state(qpos, qvel)

        # 5) Re‑extract the true MuJoCo observation
        new_obs = base._get_obs()
        #print("OBS ", obs)
        #print("TRUE OBS ", new_obs)
        return new_obs, info





       



def evaluate_policies_sb(actor,env,n_eps=10):
    mean_reward, std_reward = evaluate_policy(actor, env, n_eval_episodes=n_eps)
    #add plots
    return mean_reward, std_reward


def get_fixed_state(df) -> np.ndarray:
    """
    Args:
        df:  pandas df.
    
    Returns:
        fixed_state: A NumPy array of shape (18,) with the fixed state.
    """
    # Load the CSV file.
    
    
    # Get the first row (assuming it is the desired initial row).
    row = df.iloc[0]
    
    # Build qpos. Here we insert 0.0 as the x position (not in the CSV),
    # then the following values from the CSV.
    fixed_qpos = np.array([
        0.0,                        # x position (manually set)
        row["rootz"],
        row["rooty"],
        row["bthigh"],
        row["bshin"],
        row["bfoot"],
        row["fthigh"],
        row["fshin"],
        row["ffoot"]
    ], dtype=np.float64)
    
    # Build qvel from the CSV.
    fixed_qvel = np.array([
        row["rootx_dot"],
        row["rootz_dot"],
        row["rooty_dot"],
        row["bthigh_dot"],
        row["bshin_dot"],
        row["bfoot_dot"],
        row["fthigh_dot"],
        row["fshin_dot"],
        row["ffoot_dot"]
    ], dtype=np.float64)
    
    # Concatenate qpos and qvel to form an 18-dimensional fixed state.
    fixed_state = np.concatenate([fixed_qpos, fixed_qvel])
  
    return fixed_state




# DO NOT FORGET NUMBER OF SAMPLES WHEN ACUTALLY DOING MC DRAWS MORE THAN 1
def compute_ks(X_gt, X_pred, n_features, n_samples, n_traces):
    # assume shape n, 17 for X_pred
    # n_samples = steps?
    # n_traces = number of MC samples from the PDF
    # not applicable here as we evaluate the agent acting in the env, not PDF predicted by the LLM
    # n_features is 17
    kss = np.zeros((n_features,))
    ks_quantiles = np.zeros((n_features, n_samples-1))
    for dim in range(n_features):
        per_dim_groundtruth = X_gt[1:500,dim].flatten()

        # Compute quantiles
        quantiles = np.sort(
            np.array(
                [g > m for g, m in zip(per_dim_groundtruth, X_pred[1:500,:, dim])]
            ).sum(axis=1, dtype = float)
        )
        
        # for Marlin to look: originally, the axis of summation for the above was dimensiobn 1. hmmm
        # but since I reduced the dimensionality when passing it as an argument x[:,0,:], I guess 0 makes sense?
        #quantiles = quantiles / n_traces
        flat_preds = X_pred[1:500,:,dim].flatten()

        # Compute KS metric
        
        kss[dim] = np.max(
            np.abs(quantiles - (np.arange(len(quantiles)) / len(quantiles)))
        )
        D, p = ks_2samp(flat_preds, per_dim_groundtruth)
        ks_quantiles[dim, :] = quantiles

    return kss, ks_quantiles, D, p


def normalized_score(score, random_score=-280.2, expert_score=12135.0):
    return (score - random_score)/(expert_score - random_score) * 100




def compute_metrics(X, mean, args, n_features=17, n_traces=1, n_samples=500):
        """
        Compute the prediction metrics such as MSE and KS test.

        Args:
            burnin (int, optional): Number of initial steps to ignore when computing
                metrics. Defaults to 0.

        Returns:
            dict: Dictionary containing various prediction metrics.
        """
        metrics = {}

        # ------- MSE --------

        perdim_squared_errors = (X[1:500,:17] - mean[1:,0,:17]) ** 2
        agg_squared_error = np.linalg.norm(X[1:500, :17] - mean[1:,0,:17], axis=1)
        if args.burnin is not None:
            burnin = args.burnin
        else:
            burnin = 0
        metrics["average_agg_squared_error"] = agg_squared_error[burnin:].mean(axis=0)
        metrics["agg_squared_error"] = agg_squared_error
        metrics["average_perdim_squared_error"] = perdim_squared_errors[burnin:].mean(
            axis=0
        )
        metrics["perdim_squared_error"] = agg_squared_error

        # ------ KS -------
        kss, quantiles, D, p = compute_ks(
            X_gt=X[:,:n_features],
            X_pred=mean,
            #probably could substitute mean instead of it
            n_features=n_features,
            n_traces=n_traces,
            n_samples=n_samples
        )
        
        metrics["perdim_ks"] = kss
        metrics["agg_ks"] = kss.mean(axis=0)
        metrics["two_sample_kss"] = D
        metrics["p-value"] = p
        metrics["quantiles"] = quantiles
        return metrics







if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', type=str, required=True, help='path of list of paths to models that we are going to evaluate')
    parser.add_argument('--baseline', type=str, required=True, help='path to csv with ground truth trajectory')
    parser.add_argument('--env', nargs='?', type=str, help='environment in which we are going to evaluate our agent', const='HalfCheetah', default='HalfCheetah')
    parser.add_argument('--render', nargs='?', type=str, help='render mode', const=None, default=None)
    parser.add_argument('--seed',nargs='?', type=int, help='seed', const=42, default=42)
    parser.add_argument('--neps',nargs='?', type=int, help='number of episodes', const=20, default=20)
    parser.add_argument('--burnin',nargs='?', type=int, help='number of burnin steps', const=None, default=None)
    args = parser.parse_args()

    envs = gym.vector.SyncVectorEnv(
        [make_env_not_war(args)]
    )
    for model in args.models:
        pre_actor = Actor(envs)
        actor = ActorWrapper(pre_actor, 'cuda')
        actor.actor.load_state_dict(torch.load(model, weights_only=True))
        print(evaluate_policies_sb(actor, envs, args.neps))
        del actor
    if args.env != 'HalfCheetah':
        #for now, until we get to process hdf5s for Hopper, Kitchen and Pendulum
        exit()
    del envs


    X = pd.read_csv(args.baseline, index_col=0)
    
    fixed_state = get_fixed_state(X)
    env = gym.vector.SyncVectorEnv([make_fixed_env_not_war(fixed_state, args)])
    next_obs, _ = env.reset()
    pre_actor = Actor(env)
    actor = ActorWrapper(pre_actor, 'cuda')
    actor.actor.load_state_dict(torch.load(model, weights_only=True))

    #now, run the agent for some duration T, sample from this duration
    #pick the same indices' samples from csv, and compute_metric

    # Question for Marlin and/or Mario: is 500 a sufficient number of steps
    # to evaluate
    # also, do we evaluate n steps within 1 episode, or we randomly restart
    # episode some m times and measure x first steps
    obss, rews, infoss = [],[],[]
    next_obs = np.zeros((1, 17))
    obss.append(next_obs)
    print("next_obs ",next_obs)
    for i in range(499):
        actions, _, mean_actions = actor.actor.get_action(
                        torch.Tensor(next_obs).to('cuda')
                    )
        actions = actions.detach().cpu().numpy()
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        rews.append(rewards)
        next_obs = np.zeros((1, 17))
        print("next_obs ",next_obs)

        infoss.append(infos)
        obss.append(next_obs)
    
    obss = np.asarray(obss)
    #print(obss.shape)
    #print(X.shape)
    Xs = X.to_numpy(dtype=np.float64)
    Xs[:500, :17] = np.zeros((500, 17))
    #print(Xs[:500,:17])
    #print(obss[:,:,:17])
    mse = np.linalg.norm(obss[:,:,:17] - Xs[:500,:17])
    metrics = compute_metrics(Xs, obss[:,:], args)
    print("MSE ", mse)
    print("Other metrics (including KS) ", metrics)
    norm_score= np.sum(rews)/500
    print("normalized score ", normalized_score(norm_score))
