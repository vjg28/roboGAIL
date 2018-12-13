import click
import numpy as np
import pickle
from tqdm import tqdm

from baselines import logger
from baselines.common import set_global_seeds
import baselines.her.experiment.config as config
from varshil.roboIRL.expert_traj import RolloutWorker           

def clean_data(episode):
    for key in episode.keys():
        for i in range(int((np.array(episode[key]).shape)[0])):
            temp = episode[key][i][0]
            episode[key][i] = temp
            
    return episode["o"], episode["g"], episode["ag"], episode["u"], episode["rew"], episode["info"]

@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=0)
@click.option('--n_test_rollouts', type=int, default=5000)		##Change here
@click.option('--render', type=int, default=0)
@click.option('--fname', type=str, default = "varshil/fetchEnvData/deterministic.trpo.FetchPickAndPlace.0.01.npz")	##Change here

def main(policy_file, seed, n_test_rollouts, render,  fname):
    set_global_seeds(seed)
    percent = 0.001
    obs_npy, achieved_goal_npy, goal_npy, action_npy, reward_npy, info_npy = [], [], [], [], [], []
    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['T', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]
    
    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)

    # Run evaluation.
    evaluator.clear_history()
    print("Progress Rate ::")
    #print("*") 
    for i in tqdm(range(n_test_rollouts)):
        episode = evaluator.generate_rollouts()
        
        obs, goals, achieved_goals, acts, reward_arr, successes = clean_data(episode)
        
        obs_npy.append(np.array(obs))
        goal_npy.append(np.array(goals))
        achieved_goal_npy.append(np.array(achieved_goals))
        action_npy.append(np.array(acts))
        reward_npy.append(np.array(reward_arr))
        info_npy.append(np.array(successes))
        #print(int(n_test_rollouts * percent))
        #print(i)
        if int(i) == int(n_test_rollouts * percent):
            #print("*")
            percent += 0.001
    
    # Saving in a file
    np.savez(fname, obs= obs_npy, g = goal_npy, ag=achieved_goal_npy, acts = action_npy, rew =reward_npy, info = info_npy)
    
    
    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
