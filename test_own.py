import gym
import torch
import pickle
from core.agent import Agent

env = gym.make(
    "LunarLander-v2",
    render_mode='human',
)
observation, info = env.reset(seed=42)

state_dim = env.observation_space.shape[0]
print(state_dim)
is_disc_action = len(env.action_space.shape) == 0
print(env.action_space.shape)
#running_state = ZFilter((state_dim,), clip=5)

#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/MountainCar_1/2023-01-13-exp-1-MountainCar-v0/intermediate_model/model_iter_50.p'
#model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_1/2023-01-15-exp-1-LunarLander-v2/intermediate_model/model_iter_10.p'
model_path = '/home/pauel/PycharmProjects/RL_sapana/PyTorch-CPO/assets/learned_models/CPO/LunarLander_2/2023-01-15-exp-1-LunarLander-v2/intermediate_model/model_iter_500.p'
policy_net, _, _ = pickle.load(open(model_path, "rb"))
device = 'cpu'
policy_net.to(device)

"""create agent"""
#agent = Agent(env, policy_net, device, running_state=running_state, render=True)


for _ in range(1000):
   state_var = torch.tensor(observation, dtype = torch.float64).unsqueeze(0)
   print(observation[0])
   action = policy_net.select_action(state_var)[0]
   #print(action, int(action))
   observation, reward, terminated, truncated, info = env.step(int(action.detach().numpy()))
   #env.render()

   if terminated or truncated:
      observation, info = env.reset()
env.close()
