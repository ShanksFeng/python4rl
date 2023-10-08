from airfoilAction import perform_action
from agent import DDPGAgent, ReplayBuffer, Actor, Critic,calculate_drag, load_and_process_data
from config import device
from torch.utils.tensorboard import SummaryWriter
from agent import save_best_reward_state, reset_to_best_reward_state, reset_to_origin_state
import os
import shutil
import torch

def save_model(agent, agentIndex):
    """Save the model weights."""
    actor_path = f"./models/actor_agent{agentIndex}.pth"
    critic_path = f"./models/critic_agent{agentIndex}.pth"
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    
def load_model(agent, agentIndex):
    """Load the model weights if they exist."""
    actor_path = f"./models/actor_agent{agentIndex}.pth"
    critic_path = f"./models/critic_agent{agentIndex}.pth"
    if os.path.exists(actor_path):
        agent.actor.load_state_dict(torch.load(actor_path))
    if os.path.exists(critic_path):
        agent.critic.load_state_dict(torch.load(critic_path))


#这里永远都不要改!!!
state_dim = 2  # x, y coordinates
action_dim = 5  # number of actions
max_action = 1  # maximum value of action
action_param_dim = 3

# Shared actor and critic
actor = Actor(state_dim, action_dim, max_action).to(device)
critic = Critic(state_dim, action_dim, action_param_dim).to(device)
criticTwin = Critic(state_dim, action_dim, action_param_dim).to(device)


writer_agent_0 = SummaryWriter('runs/experiment/agent0')
writer_agent_1 = SummaryWriter('runs/experiment/agent1')
writers = [writer_agent_0, writer_agent_1]


# Number of agents
num_agents = 1
MINIMUM_REWARD_THRESHOLD = -6
agents = []
for _ in range(num_agents):
    replay_buffer = ReplayBuffer(max_size=10000)
    agent = DDPGAgent(actor, critic, criticTwin, calculate_drag, max_action, replay_buffer)
    agents.append(agent)



# Phase 1: Populate the replay buffer with short episodes
initial_timesteps = 8
initial_episodes = 32


for agentIndex, agent in enumerate(agents):
    for episode in range(initial_episodes):
        print(f"phase1 : Episode {episode + 1}/{initial_episodes}")
        episode_reward = 0
        reset_to_origin_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", agentIndex=agentIndex)
        # Reload the state to start training from the best state
        state = load_and_process_data(agent.filepaths[agentIndex]).to(device)
        agent.accumlateReward[agentIndex] = 0
        agent.previous_drag = agent.calculate_drag(agentIndex)
        print(f"successfully reset the environment")
        for t in range(initial_timesteps):
            print(f"timestep {t + 1}/{initial_timesteps}")
            # Select action
            action4execute, action_params4execute, action, action_params = agent.select_action(state)

            # Perform action
            success, _, warning_occurred = perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)

            # Get the new state
            next_state = load_and_process_data(agent.filepaths[agentIndex])

            # Get the reward
            reward = agent.get_reward(agentIndex)
            if warning_occurred:
                reward -= 3
            elif reward > 0:
                reward *= 1.5 
                
            episode_reward += reward
            agent.accumlateReward[agentIndex] += reward
            # Store transition in the replay buffer
            agent.replay_buffer.add((state, action, action_params, next_state, reward))
            print(f"Size of replay buffer: {len(agent.replay_buffer.storage)}")          
            # Additional metrics recorded at each time step
            writers[agentIndex].add_scalar('Metrics/Reward', reward, episode * initial_timesteps + t)
            lr_actor = agent.actor_optimizer.param_groups[0]['lr']
            lr_critic = agent.critic_optimizer.param_groups[0]['lr']
            writers[agentIndex].add_scalar('Learning_Rate/Actor', lr_actor, episode * initial_timesteps + t)
            writers[agentIndex].add_scalar('Learning_Rate/Critic', lr_critic, episode * initial_timesteps + t)
            writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], episode * initial_timesteps + t)
            
            # Update the state
            state = next_state
        print(f"For the agent index = : {agentIndex}")
        print(f"Episode reward: {episode_reward}")

        writers[agentIndex].add_scalar('Metrics/Reward', reward, episode * initial_timesteps + initial_timesteps - 1)
        writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], episode * initial_timesteps + initial_timesteps - 1)
 
        
        
#phase 2: train the agent
Previous_step = initial_timesteps*initial_episodes
second_timesteps = 16
second_episodes = 16
batch_size = 256

for agentIndex, agent in enumerate(agents):
    for episode in range(second_episodes):
        print(f"phase2 : Episode {episode + 1}/{second_timesteps}")
        episode_reward = 0
        reset_to_origin_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", agentIndex=agentIndex)
        # Reload the state to start training from the best state
        state = load_and_process_data(agent.filepaths[agentIndex]).to(device)
        agent.accumlateReward[agentIndex] = 0
        agent.previous_drag = agent.calculate_drag(agentIndex)
        print(f"successfully reset the environment")
        for t in range(second_timesteps):
            print(f"timestep {t + 1}/{second_timesteps}")
            # Select action
            action4execute, action_params4execute, action, action_params = agent.select_action(state)

            # Perform action
            success, _, warning_occurred = perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)

            # Get the new state
            next_state = load_and_process_data(agent.filepaths[agentIndex])

            # Get the reward
            reward = agent.get_reward(agentIndex)
            if warning_occurred:
                reward -= 3
            elif reward > 0:
                reward *= 1.5 
                
            episode_reward += reward
            agent.accumlateReward[agentIndex] += reward
            # Store transition in the replay buffer
            agent.replay_buffer.add((state, action, action_params, next_state, reward))
            print(f"Size of replay buffer: {len(agent.replay_buffer.storage)}")       
            
            # Train the agent
            print(f"Size of replay buffer: {len(agent.replay_buffer.storage)}")
            if len(agent.replay_buffer) >= batch_size:
                actor_loss, critic_loss = agent.train(iterations=3, agentIndex = agentIndex, batch_size = batch_size)
                writers[agentIndex].add_scalar('Losses/Actor_Loss', actor_loss,  episode * second_timesteps + t)
                writers[agentIndex].add_scalar('Losses/Critic_Loss', critic_loss, episode * second_timesteps + t)
               
            # Additional metrics recorded at each time step
            writers[agentIndex].add_scalar('Metrics/Reward', reward, Previous_step + episode * second_timesteps + t)
            lr_actor = agent.actor_optimizer.param_groups[0]['lr']
            lr_critic = agent.critic_optimizer.param_groups[0]['lr']
            writers[agentIndex].add_scalar('Learning_Rate/Actor', lr_actor, Previous_step + episode * second_timesteps + t)
            writers[agentIndex].add_scalar('Learning_Rate/Critic', lr_critic, Previous_step + episode * second_timesteps + t)
            writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], episode * second_timesteps + t)
            
            # Update the state
            state = next_state
        print(f"For the agent index = : {agentIndex}")
        print(f"Episode reward: {episode_reward}")

        writers[agentIndex].add_scalar('Metrics/Reward', reward, Previous_step + episode * second_timesteps + second_timesteps - 1)
        writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], Previous_step + episode * second_timesteps + second_timesteps - 1)
        if len(agent.replay_buffer) >= batch_size:
            writers[agentIndex].add_scalar('Losses/Actor_Loss', actor_loss, episode * second_timesteps + second_timesteps - 1)
            writers[agentIndex].add_scalar('Losses/Critic_Loss', critic_loss, episode * second_timesteps + second_timesteps - 1)
  


num_episodes = 200
max_timesteps = 8
totalSecondStep = second_timesteps*second_episodes
reset_to_origin_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", agentIndex=agentIndex)


for agentIndex, agent in enumerate(agents):
    load_model(agent, agentIndex)

for episode in range(num_episodes):
    print(f"Episode {episode + 1}/{num_episodes}")
    for agentIndex, agent in enumerate(agents):
        
        print(f"For the agent index = : {agentIndex}")
        state = load_and_process_data(agent.filepaths[agentIndex])
        agent.previous_drag = agent.calculate_drag(agentIndex)
        episode_reward = 0

        for t in range(max_timesteps):
            print(f"timestep {t + 1}/{max_timesteps}")
            # Check if this is the best reward so far and save the state if it is
            if agent.accumlateReward[agentIndex] > agent.best_reward_so_far[agentIndex]:
                agent.best_reward_so_far[agentIndex] = agent.accumlateReward[agentIndex]
                save_best_reward_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", dx_filepath = f"./agent{agentIndex}/CurveCapture.dx", agentIndex=agentIndex)
            # Check if the reward is below the minimum threshold and reset to the best state if it is
            if agent.accumlateReward[agentIndex] < MINIMUM_REWARD_THRESHOLD:
                reset_to_origin_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", agentIndex=agentIndex)
                # Reload the state to start training from the best state
                state = load_and_process_data(agent.filepaths[agentIndex]).to(device)
                agent.accumlateReward[agentIndex] = 0
                agent.previous_drag = agent.calculate_drag(agentIndex)
                print(f"successfully reset the environment")
            
            # Select action
            action4execute, action_params4execute, action, action_params = agent.select_action(state)

            # Perform action
            success, _, warning_occurred = perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)

            # Get the new state
            next_state = load_and_process_data(agent.filepaths[agentIndex])

            # Get the reward
            reward = agent.get_reward(agentIndex)
            if warning_occurred:
                reward -= 2
            elif reward > 0:
                reward *= 1.5 
                
            episode_reward += reward
            agent.accumlateReward[agentIndex] += reward
            # Store transition in the replay buffer
            agent.replay_buffer.add((state, action, action_params, next_state, reward))

            # Train the agent
            print(f"Size of replay buffer: {len(agent.replay_buffer.storage)}")
            if len(agent.replay_buffer) >= batch_size:
                actor_loss, critic_loss = agent.train(iterations=3, agentIndex = agentIndex, batch_size = batch_size)
                writers[agentIndex].add_scalar('Losses/Actor_Loss', actor_loss, totalSecondStep + episode * max_timesteps + t)
                writers[agentIndex].add_scalar('Losses/Critic_Loss', critic_loss, totalSecondStep + episode * max_timesteps + t)

            # Additional metrics recorded at each time step
            writers[agentIndex].add_scalar('Metrics/Reward', reward, Previous_step + totalSecondStep + episode * max_timesteps + t)
            lr_actor = agent.actor_optimizer.param_groups[0]['lr']
            lr_critic = agent.critic_optimizer.param_groups[0]['lr']
            writers[agentIndex].add_scalar('Learning_Rate/Actor', lr_actor, Previous_step + totalSecondStep + episode * max_timesteps + t)
            writers[agentIndex].add_scalar('Learning_Rate/Critic', lr_critic, Previous_step + totalSecondStep+ episode * max_timesteps + t)
            writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex],  Previous_step + totalSecondStep + episode * max_timesteps + t)
            
            # Update the state
            state = next_state
        print(f"For the agent index = : {agentIndex}")
        print(f"Episode reward: {episode_reward}")

        writers[agentIndex].add_scalar('Metrics/Reward', reward, Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
        writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
        if len(agent.replay_buffer) >= batch_size:
            writers[agentIndex].add_scalar('Losses/Actor_Loss', actor_loss, totalSecondStep + episode * max_timesteps + max_timesteps - 1)
            writers[agentIndex].add_scalar('Losses/Critic_Loss', critic_loss, totalSecondStep + episode * max_timesteps + max_timesteps - 1)

    # writers[agentIndex].add_scalar('Metrics/Episode_Reward', episode_reward, episode)

    for agentIndex, agent in enumerate(agents):
        save_model(agent, agentIndex)

for writer in writers:
    writer.close()



