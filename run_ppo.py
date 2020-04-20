# Run the experiments 
import argparse
import gym

from gym.envs.registration import EnvSpec, register, registry
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from learning_wardrop import BraessNetwork

# Make sure all algorithms are registered here
algos = {
    'PPO': PPO2
}

N_STEPS = 2
T = 1000
TEST_T = 100
NMINIBATCH = 1
N_EPOCH = 16
GAMMA = 0 #Don't take future into account


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='Braess', help='Choose the routing network. Options: (1) Braess')
    parser.add_argument('--policy', type=str, default='MlpPolicy', help="What policy function should the algorithms & the \
                                                                         agent use to choose its actions")
    parser.add_argument('--algo', type=str, default='PPO', help="What algorithm to use in the experiment")
    parser.add_argument('--opt', type=str, default='social', help="Do we want the nash or social optimum?")
    args = parser.parse_args()

    if args.network == "Braess": 
        # Set parameters for environment.
        my_env_name = "BraessEnv-v0"
        env_location = "learning_wardrop:RoutingEnv"
        braess = BraessNetwork()
        env_params = {
            "network": braess,
            "routes": braess.routes
        }
    if args.opt == "nash":
        env_params["optimum"] = "nash"
    elif args.opt == "social":
        env_params["optimum"] = "social"
    else:
        raise Exception("Need to specify which optimum we want to find.")
    
    register(
        id = my_env_name, 
        entry_point = env_location, 
        kwargs={
            "params": env_params
        }) # The keys here should correspond to the names of the arguments in your env.
    my_env = gym.make(my_env_name)
    env = DummyVecEnv([lambda: my_env])  # The algorithms require a vectorized environment to run

    # Set parameters for algorithm and policy. Need to edit this to take multiple algorithms
    policy = args.policy
    model = algos[args.algo](policy, env, verbose=1, n_steps=N_STEPS, nminibatches=NMINIBATCH, noptepochs=N_EPOCH, gamma=GAMMA)

    model.learn(total_timesteps=T)

    obs = env.reset()
    my_env.set_data_collection()
    for i in range(TEST_T):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    
    my_env.graph()