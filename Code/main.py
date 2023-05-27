import torch
import os
import numpy as np
from runner import Runner
from smac.env import StarCraft2Env
from common.arguments import get_common_args, get_coma_args, get_mixer_args, get_centralv_args, get_reinforce_args, get_commnet_args, get_g2anet_args


if __name__ == '__main__':
    for i in range(4):
        args = get_common_args()

        env = StarCraft2Env(map_name=args.map,
                            step_mul=args.step_mul,
                            difficulty=args.difficulty,
                            game_version=args.game_version,
                            replay_dir=args.replay_dir)
        env_info = env.get_env_info()

        args.n_actions = env_info["n_actions"]
        print("n_actions is ", args.n_actions)
        args.n_agents = env_info["n_agents"]
        print("n_agents is ", args.n_agents)
        args.state_shape = env_info["state_shape"]
        print("state_shape is ", args.state_shape)
        args.obs_shape = env_info["obs_shape"]
        print("obs_shape is ", args.obs_shape)
        args.episode_limit = env_info["episode_limit"]
        print("episode_limit is ", args.episode_limit)


        runner = Runner(env, args)
        if not args.evaluate:
            runner.run(i)
        else:
            win_rate, _ = runner.evaluate()
            print('The win rate of {} is  {}'.format(args.alg, win_rate))
            break
        env.close()
