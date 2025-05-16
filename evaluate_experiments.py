import gymnasium as gym
from gymnasium import Wrapper # type: ignore
import argparse
import json
import os
import re # For parsing checkpoint numbers
import torch
import numpy as np
import pandas as pd
from dicl.rl.sac_continuous_action_dicl import Actor 
from stable_baselines3.common.evaluation import evaluate_policy

class ActorWrapper:
    def __init__(self, actor: torch.nn.Module, device: torch.device = None):
        self.actor = actor
        if device is None or str(device) == 'auto':
            self.device = next(actor.parameters()).device
        else:
            self.device = torch.device(device)
        self.actor.to(self.device)
        self.actor.eval()

    def predict(
        self,
        observations: np.ndarray,
        state: tuple = None,
        episode_start: np.ndarray = None,
        deterministic: bool = False,
    ):
        is_single_obs = observations.ndim == 1
        if is_single_obs:
            observations = observations[np.newaxis, :]

        obs_tensor = torch.as_tensor(observations, device=self.device).float()
        with torch.no_grad():
            actions_sample, _, means = self.actor.get_action(obs_tensor)

        out_actions = means if deterministic else actions_sample
        
        if is_single_obs:
            return out_actions.cpu().numpy()[0], state
        return out_actions.cpu().numpy(), state

def make_env_thunk(env_id: str, seed: int, render_mode: str = None, idx: int = 0):
    def _thunk():
        env_kwargs = {}
        if render_mode and render_mode == "human": # Only pass render_mode if it's 'human' for gym.make
            env_kwargs['render_mode'] = render_mode
        
        env = gym.make(env_id, **env_kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx) 
        env.action_space.seed(seed + idx)
        try:
            env.observation_space.seed(seed + idx)
        except AttributeError:
            pass 
        return env
    return _thunk

def find_latest_checkpoint(model_dir: str, prefix: str, suffix: str):
    """
    Finds the checkpoint file with the highest iteration number in its name.
    Assumes filename format: <prefix><number><suffix>
    Returns:
        A tuple (filepath, step_number) or (None, None) if not found.
    """
    latest_step = -1
    latest_checkpoint_file_name = None
    
    pattern_str = f"^{re.escape(prefix)}(\\d+){re.escape(suffix)}$"
    try:
        pattern = re.compile(pattern_str)
    except re.error as e:
        print(f"  Error compiling regex with prefix='{prefix}', suffix='{suffix}': {e}")
        return None, None

    try:
        for filename in os.listdir(model_dir):
            match = pattern.match(filename)
            if match:
                try:
                    step_number = int(match.group(1))
                    if step_number <= 40000:
                        continue
                    if step_number > latest_step:
                        latest_step = step_number
                        latest_checkpoint_file_name = filename
                except ValueError:
                    print(f"  Warning: Could not parse step number from {filename} in {model_dir}")
                    continue
    except FileNotFoundError:
        print(f"  Warning: Directory not found during checkpoint search: {model_dir}")
        return None, None
    except Exception as e:
        print(f"  Warning: Error listing files in {model_dir} for checkpoints: {e}")
        return None, None

    if latest_checkpoint_file_name:
        return os.path.join(model_dir, latest_checkpoint_file_name), latest_step
    else:
        # This message is now printed in the main loop if None,None is returned.
        # print(f"  No checkpoints found matching pattern '{prefix}<NUMBER>{suffix}' in {model_dir}")
        return None, None

def normalized_score(score, random_score=-280.2, expert_score=12135.0):
    if expert_score == random_score:
        return 0.0 if score == random_score else float('inf')
    return (score - random_score) / (expert_score - random_score) * 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL agent checkpoints.")
    parser.add_argument('--runs_base_dir', type=str, required=True, 
                        help='Base directory containing all experiment run folders.')
    parser.add_argument('--output_csv', type=str, default='evaluation_results.csv',
                        help='Path to save the CSV file with evaluation results.')
    parser.add_argument('--env_id_key_in_json', type=str, default='env_id',
                        help='The key in args.json that specifies the environment ID (e.g., "env_id", "env").')
    parser.add_argument('--checkpoint_prefix', type=str, default='actor_checkpoint_',
                        help="Prefix for checkpoint files (e.g., 'actor_checkpoint_').")
    parser.add_argument('--checkpoint_suffix', type=str, default='.pth',
                        help="Suffix for checkpoint files (e.g., '.pth').")
    parser.add_argument('--render', type=str, default=None, const='human', nargs='?',
                        help='Render mode for the environment. E.g., "human". Invoke with no arg for human.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Global fallback seed. Also used as base for evaluation seeding.')
    parser.add_argument('--neps', type=int, default=20,
                        help='Number of episodes for evaluation.')
    parser.add_argument('--deterministic_eval', action=argparse.BooleanOptionalAction, default=True,
                        help='Use deterministic actions. Disable with --no-deterministic_eval.')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                        help="Device for torch ('cpu', 'cuda', 'auto').")

    cli_args = parser.parse_args()

    if cli_args.device == 'auto':
        selected_device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        selected_device_str = cli_args.device
    print(f"Using device: {selected_device_str}")
    torch_device = torch.device(selected_device_str)

    results_data = []
    
    if not os.path.isdir(cli_args.runs_base_dir):
        print(f"Error: Base directory '{cli_args.runs_base_dir}' not found.")
        exit(1)

    for item_name in sorted(os.listdir(cli_args.runs_base_dir)):
        model_dir = os.path.join(cli_args.runs_base_dir, item_name)
        if not os.path.isdir(model_dir):
            continue

        print(f"\nProcessing experiment: {item_name}")

        args_json_path = os.path.join(model_dir, 'args.json')
        
        if not os.path.exists(args_json_path):
            print(f"  Skipping: args.json not found in {model_dir}")
            continue

        actor_checkpoint_path, evaluated_step_number = find_latest_checkpoint(
            model_dir, cli_args.checkpoint_prefix, cli_args.checkpoint_suffix
        )
        
        if not actor_checkpoint_path:
            print(f"  Skipping: No suitable checkpoint found in {model_dir} with prefix='{cli_args.checkpoint_prefix}' and suffix='{cli_args.checkpoint_suffix}'.")
            continue
        
        print(f"  Found latest checkpoint: {os.path.basename(actor_checkpoint_path)} (Step: {evaluated_step_number})")

        try:
            with open(args_json_path, 'r') as f:
                exp_args_dict = json.load(f)
        except json.JSONDecodeError:
            print(f"  Skipping: Could not decode args.json in {model_dir}")
            continue

        env_id = exp_args_dict.get(cli_args.env_id_key_in_json)
        if not env_id:
            env_id = exp_args_dict.get('env') # Common alternative key
            if not env_id:
                print(f"  Skipping: Env ID key ('{cli_args.env_id_key_in_json}' or 'env') not found in {args_json_path}")
                continue
        
        exp_eval_seed = exp_args_dict.get('seed', cli_args.seed) 

        single_eval_env = None
        temp_vec_env_for_actor_init = None
        try:
            eval_env_thunk = make_env_thunk(env_id=env_id, seed=exp_eval_seed, render_mode=cli_args.render)
            single_eval_env = eval_env_thunk()
            
            temp_vec_env_for_actor_init = gym.vector.SyncVectorEnv(
                [make_env_thunk(env_id=env_id, seed=exp_eval_seed)]
            )
        except Exception as e:
            print(f"  Skipping: Failed to create env '{env_id}' for {item_name}. Error: {e}")
            if single_eval_env: single_eval_env.close()
            if temp_vec_env_for_actor_init: temp_vec_env_for_actor_init.close()
            continue
        
        actor_model_instance = None
        try:
            # IMPORTANT: If your `Actor` class requires specific architectural parameters
            # (e.g., hidden layer sizes) from `exp_args_dict`, you MUST pass them here.
            # Example: pre_actor = Actor(temp_vec_env_for_actor_init, hidden_dim=exp_args_dict.get('actor_hidden_dim', 256))
            pre_actor = Actor(temp_vec_env_for_actor_init) 
            actor_model_instance = ActorWrapper(pre_actor, device=torch_device)
            
            actor_model_instance.actor.load_state_dict(
                torch.load(actor_checkpoint_path, map_location=torch_device, weights_only=True)
            )
            actor_model_instance.actor.eval()

            print(f"  Evaluating model from {os.path.basename(actor_checkpoint_path)} on {env_id} for {cli_args.neps} episodes...")
            
            mean_reward, std_reward = evaluate_policy(
                actor_model_instance, 
                single_eval_env, 
                n_eval_episodes=cli_args.neps, 
                deterministic=cli_args.deterministic_eval,
                render=(cli_args.render == "human") 
            )

            print(f"  Evaluation for {item_name}: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            current_result = {'experiment_folder_name': item_name}
            current_result.update(exp_args_dict) 
            current_result['mean_reward'] = mean_reward
            current_result['std_reward'] = std_reward
            current_result['evaluated_checkpoint_filename'] = os.path.basename(actor_checkpoint_path)
            current_result['evaluated_step_number'] = evaluated_step_number # Store the step number

            if "HalfCheetah" in env_id:
                 current_result['normalized_score_hc'] = normalized_score(mean_reward)

            results_data.append(current_result)

        except Exception as e:
            print(f"  ERROR during model loading or evaluation for {item_name}: {e}")
        finally:
            if single_eval_env: single_eval_env.close()
            if temp_vec_env_for_actor_init: temp_vec_env_for_actor_init.close()
            if actor_model_instance and hasattr(actor_model_instance, 'actor'): 
                del actor_model_instance.actor 
            if actor_model_instance: 
                del actor_model_instance

    if results_data:
        df = pd.DataFrame(results_data)
        core_cols = [
            'experiment_folder_name', 'mean_reward', 'std_reward', 
            'evaluated_checkpoint_filename', 'evaluated_step_number' # Added step number here
        ]
        if 'normalized_score_hc' in df.columns:
            core_cols.append('normalized_score_hc')
        
        args_cols = sorted([col for col in df.columns if col not in core_cols])
        df = df[core_cols + args_cols]
        
        try:
            df.to_csv(cli_args.output_csv, index=False)
            print(f"\nEvaluation complete. Results saved to {cli_args.output_csv}")
        except Exception as e:
            print(f"\nError saving CSV to {cli_args.output_csv}: {e}")
            print("Dumping results to console instead:")
            print(df.to_string())
    else:
        print("\nNo experiments were successfully processed or no data was collected.")

    print("Script finished.")