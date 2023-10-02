from airfoilAction import perform_action
from agent import DDPGAgent, ReplayBuffer, Actor, Critic,calculate_drag, load_and_process_data
from config import device
from torch.utils.tensorboard import SummaryWriter


#这里永远都不要改!!!
state_dim = 2  # x, y coordinates
action_dim = 5  # number of actions
max_action = 1  # maximum value of action
action_param_dim = 3

# Shared actor and critic
actor = Actor(state_dim, action_dim, max_action).to(device)
critic = Critic(state_dim, action_dim, action_param_dim).to(device)

writer_agent_0 = SummaryWriter('runs/experiment/agent0')
writer_agent_1 = SummaryWriter('runs/experiment/agent1')
writers = [writer_agent_0, writer_agent_1]


# Number of agents
num_agents = 2

agents = []
for _ in range(num_agents):
    replay_buffer = ReplayBuffer(max_size=200)
    agent = DDPGAgent(actor, critic, calculate_drag, max_action, replay_buffer)
    agents.append(agent)


# Then start your training loop
num_episodes = 400
max_timesteps = 5
batch_size = 5


for episode in range(num_episodes):
    print(f"Episode {episode + 1}/{num_episodes}")
    for agentIndex, agent in enumerate(agents):
        
        print(f"For the agent index = : {agentIndex}")
        state = load_and_process_data(agent.filepaths[agentIndex])
        agent.previous_drag = agent.calculate_drag(agentIndex)
        episode_reward = 0

        for t in range(max_timesteps):
            # Select action
            action4execute, action_params4execute, action, action_params = agent.select_action(state)

            # Perform action
            perform_action(agent.filepaths[agentIndex], action4execute, action_params4execute)

            # Get the new state
            next_state = load_and_process_data(agent.filepaths[agentIndex])

            # Get the reward
            reward = agent.get_reward(agentIndex)
            if reward > 0: reward = 2*reward
            episode_reward += reward

            # Store transition in the replay buffer
            agent.replay_buffer.add((state, action, action_params, next_state, reward))

            # Train the agent
            print(f"Size of replay buffer: {len(agent.replay_buffer.storage)}")
            if len(agent.replay_buffer) >= batch_size:
                actor_loss, critic_loss = agent.train(iterations=3, agentIndex = agentIndex, batch_size = batch_size)
                writers[agentIndex].add_scalar('Losses/Actor_Loss', actor_loss, episode)
                writers[agentIndex].add_scalar('Losses/Critic_Loss', critic_loss, episode)

            # Update the state
            state = next_state
        print(f"For the agent index = : {agentIndex}")
        print(f"Episode reward: {episode_reward}")
        writers[agentIndex].add_scalar('Metrics/Reward', reward, episode)
        writers[agentIndex].add_scalar('Metrics/Episode_Reward', episode_reward, episode)
        lr_actor = agent.actor_optimizer.param_groups[0]['lr']
        lr_critic = agent.critic_optimizer.param_groups[0]['lr']
        writers[agentIndex].add_scalar('Learning_Rate/Actor', lr_actor, episode)
        writers[agentIndex].add_scalar('Learning_Rate/Critic', lr_critic, episode)
        writers[agentIndex].add_scalar('Metrics/Cumulative_Reward', agent.accumlateReward[agentIndex], episode)
    
for writer in writers:
    writer.close()



