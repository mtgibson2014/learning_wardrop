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

ROLLOUT_SIZE = 2
T = 1000
TEST_T = 100
NMINIBATCH = 1
N_EPOCH = 16


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, help='Choose the routing network. Options: (1) Braess')
    parser.add_argument('--policy', type=str, default='MlpPolicy', help="What policy function should the algorithms & the \
                                                                         agent use to choose its actions")
    parser.add_argument('--algo', type=str, default='PPO', help="What algorithm to use in the experiment")
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
    
    
    register(
        id = my_env_name, 
        entry_point = env_location, 
        kwargs={
            "params": env_params
        }) # The keys here should correspond to the names of the arguments in your env.
    env = gym.make(my_env_name)
    env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    # Set parameters for algorithm and policy. Need to edit this to take multiple algorithms
    policy = args.policy
    model = algos[args.algo](policy, env, verbose=1, n_steps=ROLLOUT_SIZE, nminibatches=NMINIBATCH, noptepochs=N_EPOCH)

    model.learn(total_timesteps=T)

    obs = env.reset()
    for i in range(TEST_T):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()