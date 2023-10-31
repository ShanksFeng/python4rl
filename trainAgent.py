from airfoilAction import perform_action
from agent import DDPGAgent, ReplayBuffer, Actor, Critic,calculate_drag, load_and_process_data
from config import device
from torch.utils.tensorboard import SummaryWriter
from agent import save_best_reward_state, reset_to_best_reward_state, reset_to_origin_state, save_opposite_state, reload_opposite_state
import os
import shutil
import torch
import numpy as np
import torch.nn as nn
from torch.nn.utils import clip_grad_value_

class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.0005):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state
    
def register_activation_hooks(model, writer, agentIndex, initial_timesteps, episode):
    activations = {}
    
    def hook_fn(module, input, output):
        activations[module] = output

    for name, layer in model.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d, nn.ReLU)):  # You can specify the layer types you're interested in
            layer.register_forward_hook(hook_fn)

    return activations

def calculate_entropy(probabilities):
    epsilon = 1.0e-8  # 小的正值，用来防止 log(0)
    clipped_probs = torch.clamp(probabilities, min=epsilon, max=1-epsilon)  # 裁剪概率值
    log_probabilities = torch.log(clipped_probs)  # 计算对数概率
    entropy = - torch.sum(probabilities * log_probabilities, dim=-1)  # 计算熵
    return entropy

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
action_dim = 3  # number of actions
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
    replay_buffer = ReplayBuffer(max_size=1000)
    agent = DDPGAgent(actor, critic, criticTwin, calculate_drag, max_action, replay_buffer)
    agents.append(agent)



# Phase 1: Populate the replay buffer with short episodes
initial_timesteps = 16
initial_episodes = 4


for agentIndex, agent in enumerate(agents):
    for episode in range(initial_episodes):
        print(f"phase1 : Episode {episode + 1}/{initial_episodes}")
        episode_reward = 0
        
        reset_to_origin_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", current_mapping_filepath= f"./agent{agentIndex}/CurrentPoint.dat" ,agentIndex=agentIndex)
        # Reload the state to start training from the best state
        state = load_and_process_data(agent.filepaths[agentIndex]).to(device)
        agent.accumlateReward[agentIndex] = 0
        agent.previous_drag = agent.calculate_drag(agentIndex)
        print(f"successfully reset the environment")
        
        last_action4execute = None
        last_action_params4execute = None
        last_action = None
        last_action_params = None
        
        perturb_mode = False
        
        for t in range(initial_timesteps):
            print(f"timestep {t + 1}/{initial_timesteps}")
            
                
            # Select action
            if not perturb_mode:
                action4execute, action_params4execute, action, action_params = agent.select_action(state)
            else:
                action = last_action
                action_params = last_action_params
                action4execute = last_action4execute #This must be zero
                action_params4execute = last_action_params4execute
                
                noise_generator1= OUNoise(action_dim=1)
                noise_generator2= OUNoise(action_dim=1)
                noise_generator3= OUNoise(action_dim=1)
                noise_1 = noise_generator1.sample().item()
                noise_2 = noise_generator2.sample().item()
                noise_3 = noise_generator3.sample().item()
                
                print("[PERMUTE_MODE] Parameters before noise:", action_params4execute)
                
                param_limits = [(-0.1, 0.1), (-0.0005, 0.0005), (-0.0005, 0.0005)]
                action_params4execute[0] += 40 * noise_1
                action_params4execute[1] += noise_2
                action_params4execute[2] += noise_3
                # 添加噪声并裁剪
                for i, (min_val, max_val) in enumerate(param_limits):
                    action_params4execute[i] = np.clip(action_params4execute[i], min_val + last_action_params4execute[i] * abs(1 - action_params4execute[0]) * abs(action_params4execute[0]), max_val + last_action_params4execute[i] * abs(1 - action_params4execute[0]) * abs(action_params4execute[0]))
                    action_params[0][i] = action_params4execute[i]
                action_params4execute[0] = np.clip(action_params4execute[0], 0.01, 0.99)
                if action_params4execute[1] > 0:
                    action_params4execute[1] = np.clip(action_params4execute[1], 0.0002, 0.002)
                else:
                    action_params4execute[1] = np.clip(action_params4execute[1], -0.002, -0.0002)
                if action_params4execute[2] > 0:
                    action_params4execute[2] = np.clip(action_params4execute[2], 0.0002, 0.002)
                else:
                    action_params4execute[2] = np.clip(action_params4execute[2], -0.002, -0.0002)
                # 打印添加噪声后的参数
                print("[PERMUTE_MODE] Parameters after noise:", action_params4execute)
                    
                
            # Perform action
            success, _, warning_occurred = perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)

            # Get the new state
            next_state = load_and_process_data(agent.filepaths[agentIndex])
            
            entropy = calculate_entropy(action)
            print(f"Entropy = : {entropy}") 

            # Get the reward
            reward = agent.get_reward(agentIndex)
            
                
            if warning_occurred:
                reward -= 2
            elif reward > 0:
                if action4execute == 0:
                    reward *= 1.3
                else: 
                    reward *= 1.0
                    
                    
            if action4execute == 0 and reward < 0:
                print(f"[OPPSITE] Now we try the opposite direction")
                save_opposite_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat",  mapping_filepath=f"./agent{agentIndex}/CurrentPoint.dat", agentIndex=agentIndex,)
                
                action_params4execute[1] = - 2 * action_params4execute[1]
                action_params4execute[2] = - 2 * action_params4execute[2]
                action_params[0][1] = action_params4execute[1]
                action_params[0][2] = action_params4execute[2]     
                success, _, warning_occurred = perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)  
                opposite_state = load_and_process_data(agent.filepaths[agentIndex])
                opposite_reward = agent.get_reward(agentIndex) + reward
                print("[OPPOSITE] Parameters in the opposite:", action_params4execute)
                if opposite_reward > reward:
                    next_state = opposite_state
                    reward = opposite_reward
                    if reward > 0:
                        reward *= 2.0
                    print(f"[OPPOSITE] Oppsite direction is better, reward = {reward}")
                else:
                    action_params4execute[1] = - 0.5 * action_params4execute[1]
                    action_params4execute[2] = - 0.5 * action_params4execute[2]
                    action_params[0][1] = action_params4execute[1]
                    action_params[0][2] = action_params4execute[2] 
                    reload_opposite_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat",  mapping_filepath=f"./agent{agentIndex}/CurrentPoint.dat", agentIndex=agentIndex,)
                    print(f"[OPPOSITE] Opposite direction is worse, reward = {reward}")
                    
                              
            episode_reward += reward
            agent.accumlateReward[agentIndex] += reward
            print(f"Accumlate Reward = : {agent.accumlateReward[agentIndex]}") 
            
            '''
            # Check if this is the best reward so far and save the state if it is
            if agent.accumlateReward[agentIndex] > agent.best_reward_so_far[agentIndex]:
                print(f"This is the Best Reward So Far: {agent.best_reward_so_far[agentIndex]}") 
                agent.best_reward_so_far[agentIndex] = agent.accumlateReward[agentIndex]
                if reward > 0:
                   reward *= 5.0
            '''                           
            
            
            # Store transition in the replay buffer
            agent.replay_buffer.add((state, action, action_params, next_state, reward))
            print(f"Size of replay buffer: {len(agent.replay_buffer.storage)}")          
            # Additional metrics recorded at each time step
            writers[agentIndex].add_scalar('Metrics/Reward', reward, episode * initial_timesteps + t)
            writers[agentIndex].add_scalar('Metrics/Entropy', entropy, episode * initial_timesteps + t)
            lr_actor = agent.actor_optimizer.param_groups[0]['lr']
            lr_critic = agent.critic_optimizer.param_groups[0]['lr']
            writers[agentIndex].add_scalar('Learning_Rate/Actor', lr_actor, episode * initial_timesteps + t)
            writers[agentIndex].add_scalar('Learning_Rate/Critic', lr_critic, episode * initial_timesteps + t)
            writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], episode * initial_timesteps + t)
            # 在循环结束时记录权重和梯度
            for name, param in agent.actor.named_parameters():
                writers[agentIndex].add_histogram(f'Weights/{name}', param.clone().cpu().data.numpy(), episode * initial_timesteps + t)
                if param.grad is not None:
                    writers[agentIndex].add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), episode * initial_timesteps + t)
            
            # Update the state
            state = next_state
            if reward > 0.01 and action4execute == 0:
                perturb_mode = True
            else:
                perturb_mode = False

            # 存储这一步的动作和参数，以便下一步使用
            last_action4execute = action4execute
            last_action_params4execute = action_params4execute
            last_action = action
            last_action_params = action_params
            
        print(f"For the agent index = : {agentIndex}")
        print(f"Episode reward: {episode_reward}")
        #agent.update_epsilon()
        

        writers[agentIndex].add_scalar('Metrics/Reward', reward, episode * initial_timesteps + initial_timesteps - 1)
        writers[agentIndex].add_scalar('Metrics/Entropy', entropy, episode * initial_timesteps + initial_timesteps - 1)
        writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], episode * initial_timesteps + initial_timesteps - 1)
        # 在循环结束时记录权重和梯度
        for name, param in agent.actor.named_parameters():
            writers[agentIndex].add_histogram(f'Weights/{name}', param.clone().cpu().data.numpy(), episode * initial_timesteps + initial_timesteps - 1)
            if param.grad is not None:
                writers[agentIndex].add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), episode * initial_timesteps + initial_timesteps - 1)
        for name, param in agent.critic.named_parameters():
            writers[agentIndex].add_histogram(f'Weights/{name}', param.clone().cpu().data.numpy(), episode * initial_timesteps + initial_timesteps - 1)
            if param.grad is not None:
                writers[agentIndex].add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), episode * initial_timesteps + initial_timesteps - 1)
 
        
        
#phase 2: train the agent
Previous_step = initial_timesteps*initial_episodes
second_timesteps = 64
second_episodes = 8
batch_size = 64
bonus_reward = 0.5

for agentIndex, agent in enumerate(agents):
    for episode in range(second_episodes):
        print(f"phase2 : Episode {episode + 1}/{second_timesteps}")
        
         # Check if this is the best reward so far and save the state if it is
        if agent.accumlateReward[agentIndex] > agent.best_reward_so_far[agentIndex]:
            agent.best_reward_so_far[agentIndex] = agent.accumlateReward[agentIndex]
            save_best_reward_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", dx_filepath = f"./agent{agentIndex}/CurveCapture.dx", agentIndex=agentIndex)
        # Check if the reward is below the minimum threshold and reset to the best state if it is
        if agent.accumlateReward[agentIndex] < MINIMUM_REWARD_THRESHOLD:
            reset_to_origin_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", current_mapping_filepath= f"./agent{agentIndex}/CurrentPoint.dat" ,agentIndex=agentIndex)
            # Reload the state to start training from the best state
            state = load_and_process_data(agent.filepaths[agentIndex]).to(device)
            agent.accumlateReward[agentIndex] = 0
            agent.previous_drag = agent.calculate_drag(agentIndex)
            print(f"successfully reset the environment")
        
        episode_reward = 0
        agent.accumlateReward[agentIndex] = 0
        reset_to_origin_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", current_mapping_filepath= f"./agent{agentIndex}/CurrentPoint.dat" ,agentIndex=agentIndex)
        # Reload the state to start training from the best state
        state = load_and_process_data(agent.filepaths[agentIndex]).to(device)
        
        last_action4execute = None
        last_action_params4execute = None
        last_action = None
        last_action_params = None
        
        perturb_mode = False
        
        agent.previous_drag = agent.calculate_drag(agentIndex)
        print(f"successfully reset the environment")
        for t in range(second_timesteps):
            print(f"timestep {t + 1}/{second_timesteps}")
            
            # Select action
            if not perturb_mode:
                action4execute, action_params4execute, action, action_params = agent.select_action(state)
            else:
                action = last_action
                action_params = last_action_params
                action4execute = last_action4execute #This must be zero
                action_params4execute = last_action_params4execute
                
                noise_generator1= OUNoise(action_dim=1)
                noise_generator2= OUNoise(action_dim=1)
                noise_generator3= OUNoise(action_dim=1)
                noise_1 = noise_generator1.sample().item()
                noise_2 = noise_generator2.sample().item()
                noise_3 = noise_generator3.sample().item()
                
                print("[PERMUTE_MODE] Parameters before noise:", action_params4execute)
                
                param_limits = [(-0.1, 0.1), (-0.0005, 0.0005), (-0.0005, 0.0005)]
                action_params4execute[0] += 40 * noise_1
                action_params4execute[1] += noise_2
                action_params4execute[2] += noise_3
                # 添加噪声并裁剪
                for i, (min_val, max_val) in enumerate(param_limits):
                    action_params4execute[i] = np.clip(action_params4execute[i], min_val + last_action_params4execute[i], max_val + last_action_params4execute[i])
                    action_params[0][i] = action_params4execute[i]
                    
                action_params4execute[0] = np.clip(action_params4execute[0], 0.01, 0.99)
                if action_params4execute[1] > 0:
                    action_params4execute[1] = np.clip(action_params4execute[1], 0.0002, 0.002)
                else:
                    action_params4execute[1] = np.clip(action_params4execute[1], -0.002, -0.0002)
                if action_params4execute[2] > 0:
                    action_params4execute[2] = np.clip(action_params4execute[2], 0.0002, 0.002)
                else:
                    action_params4execute[2] = np.clip(action_params4execute[2], -0.002, -0.0002)
                
                # 打印添加噪声后的参数
                print("[PERMUTE_MODE] Parameters after noise:", action_params4execute)
            

            # Perform action
            success, _, warning_occurred = perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)

            # Get the new state
            next_state = load_and_process_data(agent.filepaths[agentIndex])

            # Get the reward
            reward = agent.get_reward(agentIndex)
            
            if warning_occurred:
                reward -= 2
            elif reward > 0:
                if action4execute == 0:
                    reward *= 1.3
                else: 
                    reward *= 1.0
            
            if action4execute == 0 and reward < 0:
                print(f"[OPPSITE] Now we try the opposite direction")
                save_opposite_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat",  mapping_filepath=f"./agent{agentIndex}/CurrentPoint.dat", agentIndex=agentIndex,)
                
                action_params4execute[1] = - 2 * action_params4execute[1]
                action_params4execute[2] = - 2 * action_params4execute[2]
                action_params[0][1] = action_params4execute[1]
                action_params[0][2] = action_params4execute[2]     
                success, _, warning_occurred = perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)  
                opposite_state = load_and_process_data(agent.filepaths[agentIndex])
                opposite_reward = agent.get_reward(agentIndex) + reward
                print("[OPPOSITE] Parameters in the opposite:", action_params4execute)
                if opposite_reward > reward:
                    next_state = opposite_state
                    reward = opposite_reward
                    if reward > 0:
                        reward *= 2.0
                    print(f"[OPPOSITE] Oppsite direction is better, reward = {reward}")
                else:
                    action_params4execute[1] = - 0.5 * action_params4execute[1]
                    action_params4execute[2] = - 0.5 * action_params4execute[2]
                    action_params[0][1] = action_params4execute[1]
                    action_params[0][2] = action_params4execute[2] 
                    reload_opposite_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat",  mapping_filepath=f"./agent{agentIndex}/CurrentPoint.dat", agentIndex=agentIndex,)
                    print(f"[OPPOSITE] Opposite direction is worse, reward = {reward}")
            
            episode_reward += reward
            agent.accumlateReward[agentIndex] += reward
            print(f"Accumlate Reward = : {agent.accumlateReward[agentIndex]}")
            
            # Check if this is the best reward so far and save the state if it is
            #if agent.accumlateReward[agentIndex] > agent.best_reward_so_far[agentIndex]:
             #   print(f"This is the Best Reward So Far: {agent.best_reward_so_far[agentIndex]}")
              #  agent.best_reward_so_far[agentIndex] = agent.accumlateReward[agentIndex]
               # if reward > 0:
                #   reward *= 5.0
            
            entropy = calculate_entropy(action)
            print(f"Entropy = : {entropy}") 
            # Store transition in the replay buffer
            agent.replay_buffer.add((state, action, action_params, next_state, reward))
            print(f"Size of replay buffer: {len(agent.replay_buffer.storage)}")   
            if agent.accumlateReward[agentIndex] <= MINIMUM_REWARD_THRESHOLD:
                print(f"Early stopping of episode due to reaching minimum threshold of accumulate reward: {agent.accumlateReward[agentIndex]}")
                break
    
            activations = register_activation_hooks(agent.actor, writers[agentIndex], agentIndex, t, episode)
            # Train the agent
            print(f"Size of replay buffer: {len(agent.replay_buffer.storage)}")
            if len(agent.replay_buffer.storage) >= batch_size:
                actor_loss, critic_loss, vae_loss = agent.train(iterations=3, agentIndex = agentIndex, batch_size = batch_size)
                writers[agentIndex].add_scalar('Losses/Actor_Loss', actor_loss,  episode * second_timesteps + t)
                writers[agentIndex].add_scalar('Losses/Critic_Loss', critic_loss, episode * second_timesteps + t)
                writers[agentIndex].add_scalar('Losses/vae_Loss', vae_loss, episode * second_timesteps + t)
            # Additional metrics recorded at each time step
            writers[agentIndex].add_scalar('Metrics/Reward', reward, Previous_step + episode * second_timesteps + t)
            writers[agentIndex].add_scalar('Metrics/Entropy', entropy, Previous_step + episode * second_timesteps + t)
            lr_actor = agent.actor_optimizer.param_groups[0]['lr']
            lr_critic = agent.critic_optimizer.param_groups[0]['lr']
            writers[agentIndex].add_scalar('Learning_Rate/Actor', lr_actor, Previous_step + episode * second_timesteps + t)
            writers[agentIndex].add_scalar('Learning_Rate/Critic', lr_critic, Previous_step + episode * second_timesteps + t)
            writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], episode * second_timesteps + t)
            # 在循环结束时记录权重和梯度
            for name, param in agent.actor.named_parameters():
                writers[agentIndex].add_histogram(f'Weights/{name}', param.clone().cpu().data.numpy(), Previous_step + episode * second_timesteps + t)
                if param.grad is not None:
                    writers[agentIndex].add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), Previous_step + episode * second_timesteps + t)
                    
            for name, param in agent.critic.named_parameters():
                writers[agentIndex].add_histogram(f'Weights/{name}', param.clone().cpu().data.numpy(), Previous_step + episode * second_timesteps + t)
                if param.grad is not None:
                    writers[agentIndex].add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), Previous_step + episode * second_timesteps + t)
                    
            for name, param in agent.actor.named_parameters():
                if 'weight' in name:
                    weight_norm = torch.norm(param).item()
                    writers[agentIndex].add_scalar(f"Weight_Norms/{name}", weight_norm, Previous_step + episode * second_timesteps + t)
            for name, activation in activations.items():
                activation_norm = torch.norm(activation).item()
                writers[agentIndex].add_scalar(f"Activation_Norms/{name}", activation_norm, episode * second_timesteps + t)


            
            
            # Update the state
            state = next_state
            if reward > 0.01 and action4execute == 0:
                perturb_mode = True
            else:
                perturb_mode = False

            # 存储这一步的动作和参数，以便下一步使用
            last_action4execute = action4execute
            last_action_params4execute = action_params4execute
            last_action = action
            last_action_params = action_params
            
            
             # 在每个 episode 结束后检查是否提前结束
            if t == second_timesteps-1:  # 说明 episode 没有提前结束
                print(f"Giving extra bonus reward: {bonus_reward} for not early stopping.")
                agent.accumlateReward[agentIndex] += bonus_reward
                
        print(f"For the agent index = : {agentIndex}")
        print(f"Episode reward: {episode_reward}")
        
        agent.critic_scheduler.step()
        agent.criticTwin_scheduler.step()
        agent.actor_scheduler.step()
        agent.vae_scheduler.step()

        writers[agentIndex].add_scalar('Metrics/Reward', reward, Previous_step + episode * second_timesteps + second_timesteps - 1)
        writers[agentIndex].add_scalar('Metrics/Entropy', entropy, Previous_step + episode * second_timesteps + second_timesteps - 1)
        writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], Previous_step + episode * second_timesteps + second_timesteps - 1)
        if len(agent.replay_buffer) >= batch_size:
            writers[agentIndex].add_scalar('Losses/Actor_Loss', actor_loss, episode * second_timesteps + second_timesteps - 1)
            writers[agentIndex].add_scalar('Losses/Critic_Loss', critic_loss, episode * second_timesteps + second_timesteps - 1)
            writers[agentIndex].add_scalar('Losses/vae_Loss', vae_loss, episode * second_timesteps + second_timesteps - 1)
        agent.update_epsilon()
        # 在循环结束时记录权重和梯度
        for name, param in agent.actor.named_parameters():
            writers[agentIndex].add_histogram(f'Weights/{name}', param.clone().cpu().data.numpy(), Previous_step + episode * second_timesteps + second_timesteps - 1)
            if param.grad is not None:
                writers[agentIndex].add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), Previous_step + episode * second_timesteps + second_timesteps - 1)
        for name, param in agent.actor.named_parameters():
            if 'weight' in name:
                weight_norm = torch.norm(param).item()
                writers[agentIndex].add_scalar(f"Weight_Norms/{name}", weight_norm, Previous_step + episode * second_timesteps + second_timesteps - 1)
        for name, activation in activations.items():
                activation_norm = torch.norm(activation).item()
                writers[agentIndex].add_scalar(f"Activation_Norms/{name}", activation_norm, episode * second_timesteps + second_timesteps - 1)

    


num_episodes = 200
max_timesteps = 8
totalSecondStep = second_timesteps*second_episodes
reset_to_origin_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", current_mapping_filepath= f"./agent{agentIndex}/CurrentPoint.dat" ,agentIndex=agentIndex)


for agentIndex, agent in enumerate(agents):
    load_model(agent, agentIndex)

for episode in range(num_episodes):
    print(f"Episode {episode + 1}/{num_episodes}")
    for agentIndex, agent in enumerate(agents):
        
        print(f"For the agent index = : {agentIndex}")
        state = load_and_process_data(agent.filepaths[agentIndex])
        
        
        last_action4execute = None
        last_action_params4execute = None
        last_action = None
        last_action_params = None
        
        perturb_mode = False
        
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
                reset_to_origin_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat", current_mapping_filepath= f"./agent{agentIndex}/CurrentPoint.dat" ,agentIndex=agentIndex)
                # Reload the state to start training from the best state
                state = load_and_process_data(agent.filepaths[agentIndex]).to(device)
                agent.accumlateReward[agentIndex] = 0
                agent.previous_drag = agent.calculate_drag(agentIndex)
                print(f"successfully reset the environment")
            
            # Select action
            if not perturb_mode:
                action4execute, action_params4execute, action, action_params = agent.select_action(state)
            else:
                action = last_action
                action_params = last_action_params
                action4execute = last_action4execute #This must be zero
                action_params4execute = last_action_params4execute
                
                noise_generator1= OUNoise(action_dim=1)
                noise_generator2= OUNoise(action_dim=1)
                noise_generator3= OUNoise(action_dim=1)
                noise_1 = noise_generator1.sample().item()
                noise_2 = noise_generator2.sample().item()
                noise_3 = noise_generator3.sample().item()
                
                print("[PERMUTE_MODE] Parameters before noise:", action_params4execute)
                
                param_limits = [(-0.1, 0.1), (-0.0005, 0.0005), (-0.0005, 0.0005)]
                action_params4execute[0] += 40 * noise_1
                action_params4execute[1] += noise_2
                action_params4execute[2] += noise_3
                # 添加噪声并裁剪
                for i, (min_val, max_val) in enumerate(param_limits):
                    action_params4execute[i] = np.clip(action_params4execute[i], min_val + last_action_params4execute[i], max_val + last_action_params4execute[i])
                    action_params[0][i] = action_params4execute[i]
                    
                action_params4execute[0] = np.clip(action_params4execute[0], 0.01, 0.99)
                if action_params4execute[1] > 0:
                    action_params4execute[1] = np.clip(action_params4execute[1], 0.0002, 0.002)
                else:
                    action_params4execute[1] = np.clip(action_params4execute[1], -0.002, -0.0002)
                if action_params4execute[2] > 0:
                    action_params4execute[2] = np.clip(action_params4execute[2], 0.0002, 0.002)
                else:
                    action_params4execute[2] = np.clip(action_params4execute[2], -0.002, -0.0002)
                # 打印添加噪声后的参数
                print("[PERMUTE_MODE] Parameters after noise:", action_params4execute)

            # Perform action
            success, _, warning_occurred = perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)

            # Get the new state
            next_state = load_and_process_data(agent.filepaths[agentIndex])

            # Get the reward
            reward = agent.get_reward(agentIndex)
            
            if warning_occurred:
                reward -= 2
            elif reward > 0:
                if action4execute == 0:
                    reward *= 1.3
                else: 
                    reward *= 1.0

            if action4execute == 0 and reward < 0:
                print(f"[OPPSITE] Now we try the opposite direction")
                save_opposite_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat",  mapping_filepath=f"./agent{agentIndex}/CurrentPoint.dat", agentIndex=agentIndex,)
                
                action_params4execute[1] = - 2 * action_params4execute[1]
                action_params4execute[2] = - 2 * action_params4execute[2]
                action_params[0][1] = action_params4execute[1]
                action_params[0][2] = action_params4execute[2]     
                success, _, warning_occurred = perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)  
                opposite_state = load_and_process_data(agent.filepaths[agentIndex])
                opposite_reward = agent.get_reward(agentIndex) + reward
                print("[OPPOSITE] Parameters in the opposite:", action_params4execute)
                if opposite_reward > reward:
                    next_state = opposite_state
                    reward = opposite_reward
                    if reward > 0:
                        reward *= 2.0
                    print(f"[OPPOSITE] Oppsite direction is better, reward = {reward}")
                else:
                    action_params4execute[1] = - 0.5 * action_params4execute[1]
                    action_params4execute[2] = - 0.5 * action_params4execute[2]
                    action_params[0][1] = action_params4execute[1]
                    action_params[0][2] = action_params4execute[2] 
                    reload_opposite_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat",  mapping_filepath=f"./agent{agentIndex}/CurrentPoint.dat", agentIndex=agentIndex,)
                    print(f"[OPPOSITE] Opposite direction is worse, reward = {reward}")
            
                
            episode_reward += reward
            agent.accumlateReward[agentIndex] += reward
            
            # Check if this is the best reward so far and save the state if it is
            #if agent.accumlateReward[agentIndex] > agent.best_reward_so_far[agentIndex]:
             #   agent.best_reward_so_far[agentIndex] = agent.accumlateReward[agentIndex]
              #  if reward > 0:
               #    reward *= 5.0
            # Store transition in the replay buffer
            agent.replay_buffer.add((state, action, action_params, next_state, reward))
            entropy = calculate_entropy(action)

            # Train the agent
            print(f"Size of replay buffer: {len(agent.replay_buffer.storage)}")
            if len(agent.replay_buffer.storage) >= batch_size:
                actor_loss, critic_loss, vae_loss = agent.train(iterations=3, agentIndex = agentIndex, batch_size = batch_size)
                writers[agentIndex].add_scalar('Losses/Actor_Loss', actor_loss, totalSecondStep + episode * max_timesteps + t)
                writers[agentIndex].add_scalar('Losses/Critic_Loss', critic_loss, totalSecondStep + episode * max_timesteps + t)
                writers[agentIndex].add_scalar('Losses/vae_Loss', vae_loss, totalSecondStep + episode * max_timesteps + t)

            # Additional metrics recorded at each time step
            writers[agentIndex].add_scalar('Metrics/Reward', reward, Previous_step + totalSecondStep + episode * max_timesteps + t)
            writers[agentIndex].add_scalar('Metrics/Entropy', entropy,Previous_step + totalSecondStep + episode * max_timesteps + t)
            lr_actor = agent.actor_optimizer.param_groups[0]['lr']
            lr_critic = agent.critic_optimizer.param_groups[0]['lr']
            writers[agentIndex].add_scalar('Learning_Rate/Actor', lr_actor, Previous_step + totalSecondStep + episode * max_timesteps + t)
            writers[agentIndex].add_scalar('Learning_Rate/Critic', lr_critic, Previous_step + totalSecondStep+ episode * max_timesteps + t)
            writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex],  Previous_step + totalSecondStep + episode * max_timesteps + t)
            # 在循环结束时记录权重和梯度
            for name, param in agent.actor.named_parameters():
                writers[agentIndex].add_histogram(f'Weights/{name}', param.clone().cpu().data.numpy(), Previous_step + totalSecondStep + episode * max_timesteps + t)
                if param.grad is not None:
                    writers[agentIndex].add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), Previous_step + totalSecondStep + episode * max_timesteps + t)
                    
            for name, param in agent.actor.named_parameters():
                if 'weight' in name:
                    weight_norm = torch.norm(param).item()
                    writers[agentIndex].add_scalar(f"Weight_Norms/{name}", weight_norm, Previous_step + totalSecondStep + episode * max_timesteps + t)
            for name, param in agent.critic.named_parameters():
                if 'weight' in name:
                    weight_norm = torch.norm(param).item()
                    writers[agentIndex].add_scalar(f"Weight_Norms/{name}", weight_norm, Previous_step + totalSecondStep + episode * max_timesteps + t)
            
            # Update the state
            state = next_state
            if reward > 0 and action4execute == 0:
                perturb_mode = True
            else:
                perturb_mode = False

            # 存储这一步的动作和参数，以便下一步使用
            last_action4execute = action4execute
            last_action_params4execute = action_params4execute
            last_action = action
            last_action_params = action_params
            
            
        print(f"For the agent index = : {agentIndex}")
        print(f"Episode reward: {episode_reward}")
        
        agent.critic_scheduler.step()
        agent.criticTwin_scheduler.step()
        agent.actor_scheduler.step()
        agent.vae_scheduler.step()

        writers[agentIndex].add_scalar('Metrics/Reward', reward, Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
        writers[agentIndex].add_scalar('Metrics/Entropy', entropy,Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
        writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
        if len(agent.replay_buffer) >= batch_size:
            writers[agentIndex].add_scalar('Losses/Actor_Loss', actor_loss, totalSecondStep + episode * max_timesteps + max_timesteps - 1)
            writers[agentIndex].add_scalar('Losses/Critic_Loss', critic_loss, totalSecondStep + episode * max_timesteps + max_timesteps - 1)
            writers[agentIndex].add_scalar('Losses/vae_Loss', vae_loss, totalSecondStep + episode * max_timesteps + max_timesteps - 1)
        # 在循环结束时记录权重和梯度
        for name, param in agent.actor.named_parameters():
            writers[agentIndex].add_histogram(f'Weights/{name}', param.clone().cpu().data.numpy(), Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
            if param.grad is not None:
                writers[agentIndex].add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
                
        for name, param in agent.critic.named_parameters():
            writers[agentIndex].add_histogram(f'Weights/{name}', param.clone().cpu().data.numpy(), Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
            if param.grad is not None:
                writers[agentIndex].add_histogram(f'Gradients/{name}', param.grad.clone().cpu().data.numpy(), Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
                
        for name, param in agent.actor.named_parameters():
                if 'weight' in name:
                    weight_norm = torch.norm(param).item()
                    writers[agentIndex].add_scalar(f"Weight_Norms/{name}", weight_norm, Previous_step + totalSecondStep + episode * max_timesteps + max_timesteps - 1)
        

    # writers[agentIndex].add_scalar('Metrics/Episode_Reward', episode_reward, episode)
    agent.update_epsilon()
    for agentIndex, agent in enumerate(agents):
        save_model(agent, agentIndex)

for writer in writers:
    writer.close()



