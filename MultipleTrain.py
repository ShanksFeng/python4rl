import torch
from airfoilAction import perform_action
from agent import DDPGAgent, ReplayBuffer, Actor, Critic, calculate_drag, load_and_process_data
from config import device
import torch.multiprocessing as mp
import sys
import os

state_dim = 2  # x, y coordinates
action_dim = 5  # number of actions
max_action = 1  # maximum value of action
action_param_dim = 3

num_episodes = 400
max_timesteps = 1
batch_size = 1

SAVE_PATH_ACTOR = "./actor.pth"
SAVE_PATH_CRITIC = "./critic.pth"

# Shared actor and critic
actor = Actor(state_dim, action_dim, max_action).to(device)
critic = Critic(state_dim, action_dim, action_param_dim).to(device)
# Number of agents
num_agents = 2

agents = []
for _ in range(num_agents):
    replay_buffer = ReplayBuffer(max_size=10)
    agent = DDPGAgent(actor, critic, calculate_drag, max_action, replay_buffer)
    agents.append(agent)

def worker(agentIndex, agent, episode_rewards, replay_buffer_queue):

    #print(f"For the agent index = : {agentIndex}")
    state = load_and_process_data(agent.filepaths[agentIndex])  
    agent.previous_drag = agent.calculate_drag(agentIndex)
    episode_reward = 0

    for t in range(max_timesteps):
        action4execute, action_params4execute, action, action_params = agent.select_action(state)
        perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)
        next_state = load_and_process_data(agent.filepaths[agentIndex])  
        reward = agent.get_reward(agentIndex)
        episode_reward += reward
        #print("State is on GPU:", state.is_cuda)
        #print("Action is on GPU:", action.is_cuda)
        #all_tensors_on_gpu = all(all(tensor.is_cuda for tensor in sublist) for sublist in action_params)
        #print("All tensors in action_params are on GPU:", all_tensors_on_gpu)
        


        #agent.replay_buffer.add((state, action, action_params, next_state, reward))
        #print("Type of state:", type(state))
        #print("Type of action:", type(action))
        #print("Type of action_params:", type(action_params))
        #print("Type of next_state:", type(next_state))
        #print("Type of reward:", type(reward))
        state_np = state.cpu().numpy()
        action_np = action.cpu().numpy()
        next_state_np = next_state.cpu().numpy()
        action_params_cpu = [[tensor.cpu() for tensor in sublist] for sublist in action_params]


        replay_buffer_queue.put((state_np, action_np, action_params_cpu, next_state_np, reward))
        #agent.replay_buffer.add(experience)
        state = next_state

    episode_rewards.put(episode_reward)
    # Synchronize CUDA operations and clear cache
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    
    #for experience in agent.replay_buffer.storage:
        #replay_buffer_queue.put(experience)
        
if __name__ == '__main__':
    if os.path.exists(SAVE_PATH_ACTOR) and os.path.exists(SAVE_PATH_CRITIC):
        agent.actor.load_state_dict(torch.load(SAVE_PATH_ACTOR))
        agent.critic.load_state_dict(torch.load(SAVE_PATH_CRITIC))
    else:
        print("Model weights not found. Using initialized weights.")
    torch.multiprocessing.set_sharing_strategy('file_system')
    mp.set_start_method('spawn')  # IMPORTANT for CUDA
    replay_buffer_queues = [mp.Queue() for _ in range(num_agents)]

    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")

        processes = []
        episode_rewards = mp.Queue()

        #replay_buffer_queue = mp.Queue()
        
        for agentIndex, agent in enumerate(agents):
            p = mp.Process(target=worker, args=(agentIndex, agent, episode_rewards, replay_buffer_queues[agentIndex]))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
        total_rewards = [episode_rewards.get() for _ in agents]
        
        for agentIndex, agent in enumerate(agents):
            while not replay_buffer_queues[agentIndex].empty():
                experience_np = replay_buffer_queues[agentIndex].get()
                # Convert numpy arrays back to tensors
                state_tensor = torch.tensor(experience_np[0], device=device)
                action_tensor = torch.tensor(experience_np[1], device=device)
                next_state_tensor = torch.tensor(experience_np[3], device=device)
                experience_tensor = (state_tensor, action_tensor, experience_np[2], next_state_tensor, experience_np[4])
                agent.replay_buffer.add(experience_tensor)
        
        #while not replay_buffer_queue.empty():
           # experience = replay_buffer_queue.get()
            #for agent in agents:  # Add the experience to all agents' ReplayBuffer
             #   agent.replay_buffer.add(experience)

        print(f"Total episode rewards for all agents: {total_rewards}")
        
        for agentIndex, agent in enumerate(agents):
            print(f"Size of replay buffer for agent {agentIndex}: {len(agent.replay_buffer)}")
            if len(agent.replay_buffer) >= batch_size:
                agent.train(iterations=3, agentIndex=agentIndex, batch_size=batch_size)
         # Save the weights after training
        torch.save(agent.actor.state_dict(), SAVE_PATH_ACTOR)
        torch.save(agent.critic.state_dict(), SAVE_PATH_CRITIC)