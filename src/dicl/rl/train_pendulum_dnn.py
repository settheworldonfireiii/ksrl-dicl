import os
import numpy as np
import tensorflow as tf
import gymnasium as gym
from tqdm import tqdm
import argparse

from dicl.rl.tf_models.constructor import construct_model, format_samples_for_training
from dicl.rl.tf_models.bnn import BNN

def collect_pendulum_data(num_episodes=100, max_steps=200):
    env = gym.make('Pendulum-v1')
    env.reset(seed=0)
    
    data = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': []
    }
    
    print("Collecting data from Pendulum environment...")
    for episode in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        for step in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            data['observations'].append(obs)
            data['actions'].append(action)
            data['rewards'].append([reward])  # Ensure reward is a 1D array with one element
            data['next_observations'].append(next_obs)
            
            obs = next_obs
            
            if terminated or truncated:
                break
    
    for key in data:
        data[key] = np.array(data[key])
    
    print(f"Collected {len(data['observations'])} transitions")
    return data

def train_bnn_model(data, obs_dim, act_dim, num_networks=5, num_elites=3, hidden_dim=256, save_dir="./models"):
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    tf_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    tf_session = tf.compat.v1.Session(config=tf_config)
    
    model = construct_model(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=hidden_dim,
        num_networks=num_networks,
        num_elites=num_elites,
        session=tf_session
    )
    
    inputs, outputs = format_samples_for_training(data)    
    print("Training BNN model...")
    model.train(
        inputs=inputs,
        targets=outputs,
        batch_size=32,
        epochs=50,
        hide_progress=False,
        holdout_ratio=0.2
    )
    
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, "pendulum_bnn")
    model.save(model_path, timestep=0)
    print(f"Model saved to {model_path}")
    
    return model, tf_session

def freeze_bnn_weights(model):
    """Freeze the weights of the BNN model"""
    print("Freezing BNN weights...")
    model._original_optvars = model.optvars.copy()
    
    model.weights_frozen = True
    
    original_train = model.train
    
    def train_with_frozen_weights(*args, **kwargs):
        print("Training with frozen BNN weights")
        return original_train(*args, **kwargs)
    
    model.train = train_with_frozen_weights
    
    print("BNN weights frozen")
    return model

def train_sac_with_frozen_bnn(args):
    """Main function to train SAC with a frozen BNN"""
    np.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)
    
    env = gym.make('Pendulum-v1')
    obs_dim = env.observation_space.shape[0] 
    act_dim = env.action_space.shape[0]  
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("Collecting data from the environment")
    data = collect_pendulum_data(num_episodes=args.num_episodes)
    
    print("Training the BNN model")
    bnn_model, tf_session = train_bnn_model(
        data=data,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_networks=args.num_networks,
        num_elites=args.num_elites,
        hidden_dim=args.hidden_dim,
        save_dir=args.save_dir
    )
    
    print("Freezing BNN weights")
    bnn_model = freeze_bnn_weights(bnn_model)
    
    model_path = os.path.join(args.save_dir, "pendulum_bnn_frozen")
    bnn_model.save(model_path, timestep=0)
    print(f"Model with frozen weights saved to {model_path}")
    
    print("BNN training and freezing completed. The model is ready to be used with SAC.")
    
    return bnn_model, tf_session

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train BNN model on Pendulum environment and freeze weights')
    
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of episodes to collect data for BNN training')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size for BNN')
    parser.add_argument('--num_networks', type=int, default=5, help='Number of networks in the ensemble')
    parser.add_argument('--num_elites', type=int, default=3, help='Number of elite networks to use')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    bnn_model, tf_session = train_sac_with_frozen_bnn(args)
    
    tf_session.close()

