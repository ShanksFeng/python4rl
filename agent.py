import torch
import torch.nn as nn
import torch.nn.functional as F
import subprocess
import pandas as pd
import time
import os
import torch.optim as optim
import numpy as np
from airfoilAction import calculate_total_distance
from torch.optim.lr_scheduler import StepLR
import shutil
from airfoilAction import perform_action
from torch.nn.utils.rnn import pad_sequence
import copy
from config import device
from torch.optim import lr_scheduler


def calculate_entropy(probabilities):
    epsilon = 1.0e-8  # 小的正值，用来防止 log(0)
    clipped_probs = torch.clamp(probabilities, min=epsilon, max=1-epsilon)  # 裁剪概率值
    log_probabilities = torch.log(clipped_probs)  # 计算对数概率
    entropy = - torch.sum(probabilities * log_probabilities, dim=-1)  # 计算熵
    return entropy

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


def print_structure_and_type(obj, indent=0):
    """Recursively print the structure and type of a nested object."""
    spacing = ' ' * indent
    if isinstance(obj, list):
        print(f"{spacing}List of length {len(obj)}:")
        for i, item in enumerate(obj):
            print(f"{spacing}  Index {i}:")
            print_structure_and_type(item, indent + 4)
    elif isinstance(obj, tuple):
        print(f"{spacing}Tuple of length {len(obj)}:")
        for i, item in enumerate(obj):
            print(f"{spacing}  Index {i}:")
            print_structure_and_type(item, indent + 4)
    elif isinstance(obj, dict):
        print(f"{spacing}Dict with keys: {list(obj.keys())}")
        for key, value in obj.items():
            print(f"{spacing}  Key '{key}':")
            print_structure_and_type(value, indent + 4)
    else:
        print(f"{spacing}Type: {type(obj)}")
        print(f"{spacing}Value: {obj}")
        
def save_opposite_state(mesh_filepath, airfoil_data_filepath, mapping_filepath, agentIndex):
    """
    Save the current state files, prepare for the oppsite action.

    :param mesh_filepath: Path to the current mesh file.
    :param airfoil_data_filepath: Path to the current airfoil data file.
    """
     # Paths to the best reward state files
    print("Save current and step to opposite State")
    opposite_mesh_filepath = f"./agent{agentIndex}/opposite.mesh"
    opposite_airfoil_data_filepath = f"../data/multipleAgent/agent{agentIndex}/OppositeAirfoil.dat"
    opposite_mapping_filepath = f"./agent{agentIndex}/mappingOppositePoint.dat"
    

    # Ensure the directories exist before copying files
    os.makedirs(os.path.dirname(opposite_mesh_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(opposite_airfoil_data_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(opposite_mapping_filepath), exist_ok=True)

    # Copy the current files to the opposite state files
    shutil.copy(mesh_filepath, opposite_mesh_filepath)
    shutil.copy(airfoil_data_filepath, opposite_airfoil_data_filepath)  
    shutil.copy(mapping_filepath, opposite_mapping_filepath)
    
def reload_opposite_state(mesh_filepath, airfoil_data_filepath, mapping_filepath, agentIndex):
    """
    Reset the environment to the opposite state.

    :param mesh_filepath: Path to the current mesh file.
    :param airfoil_data_filepath: Path to the current airfoil data file.
    """
    print(" we recover to the opposite state")
    # Paths to the best reward state files
    opposite_mesh_filepath = f"./agent{agentIndex}/opposite.mesh"
    opposite_airfoil_data_filepath = f"../data/multipleAgent/agent{agentIndex}/OppositeAirfoil.dat"
    opposite_mapping_filepath = f"./agent{agentIndex}/mappingOppositePoint.dat"
    
    print(f"Copying from {opposite_mesh_filepath} to {mesh_filepath}")
    print(f"Copying from {opposite_airfoil_data_filepath} to {airfoil_data_filepath}")
    print(f"Copying from {opposite_mapping_filepath} to {mapping_filepath}")

    
    try:
        # Copy the best reward state files to the current files
        shutil.copy(opposite_mesh_filepath, mesh_filepath)
        print(f"Successfully copied mesh file to {mesh_filepath}")
        
        shutil.copy(opposite_airfoil_data_filepath, airfoil_data_filepath)
        print(f"Successfully copied airfoil data file to {airfoil_data_filepath}")
        
        shutil.copy(opposite_mapping_filepath, mapping_filepath)
        print(f"Successfully copied mapping file to {mapping_filepath}")
        
    except Exception as e:
        print(f"Error occurred: {e}")   
        

def save_best_reward_state(mesh_filepath, airfoil_data_filepath, dx_filepath, agentIndex):
    """
    Save the current best reward state files.

    :param mesh_filepath: Path to the current mesh file.
    :param airfoil_data_filepath: Path to the current airfoil data file.
    """
     # Paths to the best reward state files
    print("This is the best state, we save it !!!")
    best_mesh_filepath = f"./agent{agentIndex}/bestReward.mesh"
    best_airfoil_data_filepath = f"../data/multipleAgent/agent{agentIndex}/Bestairfoil.dat"
    best_dx_filepath = f"./agent{agentIndex}/best.dx"

    # Ensure the directories exist before copying files
    os.makedirs(os.path.dirname(best_mesh_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(best_airfoil_data_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(best_dx_filepath), exist_ok=True)

    # Copy the current files to the best reward state files
    shutil.copy(mesh_filepath, best_mesh_filepath)
    shutil.copy(airfoil_data_filepath, best_airfoil_data_filepath)
    shutil.copy(dx_filepath, best_dx_filepath)

def reset_to_best_reward_state(mesh_filepath, airfoil_data_filepath, agentIndex):
    """
    Reset the environment to the best reward state.

    :param mesh_filepath: Path to the current mesh file.
    :param airfoil_data_filepath: Path to the current airfoil data file.
    """
    print("The model crap!!! we recover to the best state")
    # Paths to the best reward state files
    best_mesh_filepath = f"./agent{agentIndex}/bestReward.mesh"
    best_airfoil_data_filepath = f"../data/multipleAgent/agent{agentIndex}/Bestairfoil.dat"
    #best_mapping_filepath = f"./agent{agentIndex}/OriginMap.dat"
    
    print(f"Copying from {best_mesh_filepath} to {mesh_filepath}")
    print(f"Copying from {best_airfoil_data_filepath} to {airfoil_data_filepath}")
    #print(f"Copying from {current_mapping_filepath} to {best_mapping_filepath}")
    
    try:
        # Copy the best reward state files to the current files
        shutil.copy(best_mesh_filepath, mesh_filepath)
        print(f"Successfully copied mesh file to {mesh_filepath}")
        
        shutil.copy(best_airfoil_data_filepath, airfoil_data_filepath)
        print(f"Successfully copied airfoil data file to {airfoil_data_filepath}")
        
        #shutil.copy(best_mapping_filepath, current_mapping_filepath)
        #print(f"Successfully copied mapping file to {best_mapping_filepath}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
def reset_to_origin_state(mesh_filepath, airfoil_data_filepath, current_mapping_filepath, agentIndex):
    """
    Reset the environment to the best reward state.

    :param mesh_filepath: Path to the current mesh file.
    :param airfoil_data_filepath: Path to the current airfoil data file.
    """
    print("The model crap!!! we recover to the Initial state")
    # Paths to the best reward state files
    best_mesh_filepath = f"./agent{agentIndex}/Origin.mesh"
    best_airfoil_data_filepath = f"../data/airfoils/naca0012Revised.dat"
    best_mapping_filepath = f"./agent{agentIndex}/OriginMap.dat"
    
    print(f"Copying from {best_mesh_filepath} to {mesh_filepath}")
    print(f"Copying from {best_airfoil_data_filepath} to {airfoil_data_filepath}")
    print(f"Copying from {best_mapping_filepath} to {current_mapping_filepath}")
    
    try:
        # Copy the best reward state files to the current files
        shutil.copy(best_mesh_filepath, mesh_filepath)
        print(f"Successfully copied mesh file to {mesh_filepath}")
        
        shutil.copy(best_airfoil_data_filepath, airfoil_data_filepath)
        print(f"Successfully copied airfoil data file to {airfoil_data_filepath}")
        
        shutil.copy(best_mapping_filepath, current_mapping_filepath)
        print(f"Successfully copied mapping file to {current_mapping_filepath}")
        
    except Exception as e:
        print(f"Error occurred: {e}")


def calculate_drag(agentIndex=None):
    # Set the working directory based on the agentIndex
    working_directory = os.getcwd()  # Get the current working directory
    if agentIndex is not None:
        working_directory = os.path.join(working_directory, f"agent{agentIndex}")

    # Check if the done.txt file exists from a previous run and remove it
    done_file_path = os.path.join(working_directory, 'done.txt')
    if os.path.exists(done_file_path):
        os.remove(done_file_path)

    # Start the shell script
    process = subprocess.Popen(["./Autorun.sh"], cwd=working_directory)

    # Wait for the done.txt file to appear
    while not os.path.exists(done_file_path):
        time.sleep(1)

    # Now the script has finished, we can read the result file
    result_file_path = os.path.join(working_directory, 'residualElement.dat')
    df = pd.read_csv(result_file_path, sep=" ", header=None, on_bad_lines='skip')

    # Get the value from the last line, second column
    result = df.iloc[-1, 1]
    print(result)
    #print(type(result))
    # Optionally, remove the done.txt file after reading the result
    os.remove(done_file_path)

    return float(result)


def load_and_process_data(filepath):
    # 读取数据
    with open(filepath, 'r') as file:
        data = file.readlines()

    # 提取坐标
    coordinates = []

    for line in data[3:]:  # 从第4行开始
        parts = line.split()
        if len(parts) != 2:  # 如果不是两个值，跳过这一行
            continue

        try:
            x, y = map(float, parts)
            coordinates.append([x, y])
        except ValueError:
            print(f"警告：跳过了行 '{line}', 因为它不含有能转换为浮点数的元素")

    # 转换为 Tensor
    coordinates = torch.tensor(coordinates, dtype=torch.float32)

    # 调整形状以适应网络的输入
    coordinates = coordinates.unsqueeze(0).to(device)

    return coordinates

class ScaledDotProductAttention(nn.Module):
    def __init__(self, feature_dim):
        super(ScaledDotProductAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = torch.sqrt(torch.FloatTensor([feature_dim])).to(device)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        attention_weights = F.softmax(Q.matmul(K.transpose(-2, -1)) / self.scale, dim=-1)
        output = attention_weights.matmul(V)
        
        return output
class PenalizedSigmoid:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def __call__(self, x):
        sigm = torch.sigmoid(x)
        return sigm + self.alpha * sigm * (1 - sigm)

    
class PenalizedTanh:
    # This is a activation function that penalizes the output of tanh, it can be used as a replacement of tanh
    def __init__(self, alpha=0.05):
        self.alpha = alpha
    
    def __call__(self, x):
        return F.tanh(x) - self.alpha * F.tanh(x)**2

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.layer_norm1 = nn.LayerNorm(hidden_features)

        self.fc2 = nn.Linear(hidden_features, out_features) 
        self.layer_norm2 = nn.LayerNorm(out_features)

        self.leaky_relu = nn.LeakyReLU()
        
        # 添加一个shortcut层来调整residual的维度（如果需要的话）
        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.fc1(x)
        out = self.layer_norm1(out)
        out = self.leaky_relu(out)
        
        out = self.fc2(out)
        out = self.layer_norm2(out)
        
        out = self.leaky_relu(out + residual)
        return out
    
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        # Split the input data into two parts: values and gates
        values, gates = torch.chunk(x, 2, dim=1)
        return values * torch.sigmoid(gates)



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.fc = nn.Linear( 66 * 2 * 2, 256) 
        #66 for the upper, 2 for the both side, upper and lower, 2 for the x and y
        #self.lstm = nn.LSTM(state_dim, 256, 1, batch_first=True)
        self.attention = ScaledDotProductAttention(256)


        
        self.res_block1 = ResidualBlock(256, 512, 256)
        self.res_block2 = ResidualBlock(256, 512, 256)
        #self.res_block3 = ResidualBlock(512, 256, 256)
        
        self.glu = GLU()

        self.layer_action = nn.Linear(128, action_dim)
        

        # For action 0
        self.layer_param1_0 = nn.Linear(128, 16)
        # Then we have glu layer
        #self.gluAction0 = GLU()
        self.layer_param2_0 = nn.Linear(16, 1)
        self.layer_param3_0 = nn.Linear(16, 1)
        self.layer_param4_0 = nn.Linear(16, 1)
        # For action 1
        self.layer_param1_1 = nn.Linear(128, 16)
        #self.gluAction1 = GLU()
        self.layer_param2_1 = nn.Linear(16, 1)

        # For action 2
        self.layer_param1_2 = nn.Linear(128, 16)
        #self.gluAction2 = GLU()
        self.layer_param2_2 = nn.Linear(16, 1)

        # For action 3
        #self.layer_param1_3 = nn.Linear(256, 128)
        #self.layer_param2_3 = nn.Linear(128, 1)

        # For action 4
        #self.layer_param1_4 = nn.Linear(256, 128)
        #self.layer_param2_4 = nn.Linear(128, 1)

        self.max_action = max_action

        # Initialize weights
        self._initialize_weights()

    def forward(self, state):
        batch_size = state.size(0)
        x0 = F.relu(self.fc(state.reshape(batch_size, -1)))
        print("actor_state shape: ", x0.shape)
        #h_t, _ = self.lstm(state)
        #h_t = h_t[:, -1]  # 使用 LSTM 的最后一个输出
        x1 = self.attention(x0)
        #h_t = self.feature_extractor(h_t)
        
        
        batch_size = state.size(0)

        x2 = self.res_block1(x1 + x0)
        x3 = self.res_block2(x2 + x1)
        #h_t = self.res_block3(h_t)

        x4 = self.glu(x3)
        #The dimension will be half of the input after the GLU layer
        
        action_logits = F.leaky_relu(self.layer_action(x4))
        action = F.softmax(action_logits, dim=-1)

        Constriant = 0.002
        Lbound = 0
        Rbound = 1
        # 对于每个动作，使用相应的线性层生成参数
        penalized_tanh = PenalizedTanh(alpha=0.1)
        penalized_sigmoid = PenalizedSigmoid(alpha=0.1)
        param1_0 = F.relu(self.layer_param1_0(x4))
        #param1_0 = self.gluAction0(x5)
        
        # This detach is to remove the gradient calculation of sigmoid and tanh, since this is the physical constraint, we do not want it to influence the gradient calculation, which may lead to the gradient disappear
        param2_0 = Lbound + penalized_sigmoid(self.layer_param2_0(param1_0.detach())) * (Rbound -Lbound)
        param3_0 = Constriant * penalized_tanh(self.layer_param3_0(param1_0.detach()))
        param4_0 = Constriant * penalized_tanh(self.layer_param4_0(param1_0.detach()))

        #param1_1 = Lbound + torch.sigmoid(F.relu(self.layer_param1_1(h_t))) * (Rbound - Lbound)
        #param2_1 = Lbound + torch.sigmoid(self.layer_param2_1(param1_1)) * (Rbound -Lbound)

        #param1_2 = Lbound + torch.sigmoid(F.relu(self.layer_param1_2(h_t))) * (Rbound -Lbound)
        #param2_2 = Lbound + torch.sigmoid(self.layer_param2_2(param1_2)) * (Rbound -Lbound)

        param1_1 = F.relu(self.layer_param1_1(x4))
        
        param2_1 = Constriant * penalized_tanh(self.layer_param2_1(param1_1.detach()))

        param1_2 = F.relu(self.layer_param1_2(x4))
        param2_2 = Constriant * penalized_tanh(self.layer_param2_2(param1_2.detach()))

        action_params = [[param2_0, param3_0, param4_0], 
                         [param2_1], 
                         [param2_2]] 
                         #[param2_3], 
                         #[param2_4]]

        return action, action_params

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 特定层的初始化
                if m in [self.layer_param2_0, self.layer_param3_0, self.layer_param4_0, self.layer_param2_1, self.layer_param2_2]:
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                # 输出action logits的层
                elif m == self.layer_action:
                    nn.init.uniform_(m.weight, -0.07, 0.07)
                    nn.init.constant_(m.bias, 0)
                # 其他层的初始化
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)



class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        weights = torch.nn.functional.softmax(self.linear(x), dim=-1)
        weighted_features = weights * x
        return weighted_features



class Critic(nn.Module):
    def __init__(self, state_dim, action_dim = 3, action_param_dim = 3):
        super(Critic, self).__init__()

        # State processing
        self.state_fc1 = nn.Linear(66 * 2 * 2, 256)
        # 66 for the upper, 2 for the both side, upper and lower, 2 for the x and y
        
        self.bn_state = nn.BatchNorm1d(256)
        self.attention = ScaledDotProductAttention(256) # 32 * 8
        
        # Action processing
        self.action_fc = nn.Linear(action_dim, 32)
        self.bn_action = nn.BatchNorm1d(32) # 32 * 1

        # Action parameter processing
        self.action_param_fc1 = nn.Linear(action_dim * action_param_dim, 128)
        self.bn_action_param = nn.BatchNorm1d(128)
        self.attentionAction = ScaledDotProductAttention(128) # 32 * 4
        
        # Interaction processing
        self.interaction_fc1 = nn.Linear( 256 * 32, 256)
        #self.gluInteraction1 = GLU()
        self.interaction_fc2 = nn.Linear( 32 * 128, 256)
        #self.gluInteraction2 = GLU()
        self.interaction_fc3 = nn.Linear( 256 * 128, 256 * 4)
        #self.gluInteraction3 = GLU()
        self.interaction_fc = nn.Linear(256 * 6, 128) # 32 * 4

        # Fusion and Q-value estimation
        self.fusion_fc1 = nn.Linear( 32 * 17, 256)
        self.bn_fusion1 = nn.BatchNorm1d(256)
        self.fusion_fc2 = nn.Linear(256, 32)
        self.bn_fusion2 = nn.BatchNorm1d(32)
        self.q_value_fc = nn.Linear(32, 1)

        # Dropout layer
        self.dropout = nn.Dropout(p=0.02)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, state, action, action_params):
        
        batch_size = state.size(0)
        state = state.reshape(batch_size, -1)
        # Process state through LSTM and a fully connected layer
        state = F.relu(self.state_fc1(state))
        state = self.attention(state)   # Applying attention mechanism
        #state = F.elu(self.state_fc(state))
        #state = self.bn_state(state)
        #print("critic_state shape: ", state.shape)
        
        # Process action through a fully connected layer
        action = action.float()
        action = action.reshape(batch_size, -1) 
        #print("critic_action shape: ", action.shape)
        action = F.relu(self.action_fc(action))
        #action = self.attentionAction(action)   # Applying attention mechanism
        #action = self.bn_action(action)
        #print("critic_action shape after fc: ", action.shape)
        
        # Process action parameters through a fully connected layer
        action_params = action_params.view(-1, action_params.size(-2) * action_params.size(-1))  # Reshape to 2D
        action_params = F.elu(self.action_param_fc1(action_params))
        #action_params = self.bn_action_param(action_params)
        #print("critic_action_params shape: ", action_params.shape)
        
        # Create interaction terms through outer product operation
        interaction_term1 = torch.bmm(state[:, :, None], action[:, None, :]).view(-1, 256 * 32)
        interaction_term2 = torch.bmm(action[:, :, None], action_params[:, None, :]).view(-1, 32 * 128)
        interaction_term3 = torch.bmm(state[:, :, None], action_params[:, None, :]).view(-1, 256 * 128)
        
        # Reduce the dimensionality of the interaction terms
        interaction_term1 = F.relu(self.interaction_fc1(interaction_term1))
        #interaction_term1 = self.gluInteraction1(interaction_term1)
        interaction_term2 = F.relu(self.interaction_fc2(interaction_term2))
        #interaction_term2 = self.gluInteraction2(interaction_term2)
        interaction_term3 = F.relu(self.interaction_fc3(interaction_term3))
        #interaction_term3 = self.gluInteraction3(interaction_term3)
        interaction_term = torch.cat([interaction_term1, interaction_term2, interaction_term3], dim=1)
        interaction_term = F.relu(self.interaction_fc(interaction_term))
        
        # Concatenate all the features
        concat_features = torch.cat([state, action, action_params, interaction_term], dim=1)
        # Process the concatenated features through the fusion network
        fusion = F.relu(self.fusion_fc1(concat_features))
        #fusion = self.bn_fusion1(fusion)
        fusion = self.dropout(fusion)  # Apply dropout
        fusion = F.relu(self.fusion_fc2(fusion))
        #fusion = self.bn_fusion2(fusion)
        
        # Estimate the Q-value
        q_value = self.q_value_fc(fusion)
        
        return q_value

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ReplayBuffer:
    def __init__(self, max_size):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[self.ptr] = transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
            print("Added to buffer. Current size:", len(self.storage))

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        batch_states, batch_actions, batch_action_params, batch_next_states, batch_rewards = [], [], [], [], []

        for i in ind:
            state, action, action_params, next_state, reward = self.storage[i]
            batch_states.append(state)
            batch_actions.append(action)
            batch_action_params.append(action_params)
            batch_next_states.append(next_state)
            batch_rewards.append(reward)
        return batch_states, batch_actions,  batch_action_params, batch_next_states, batch_rewards
    
    def __len__(self):
        return len(self.storage)

class DDPGAgent:
    def __init__(self, actor, critic, criticTwin , calculate_drag, max_action, replay_buffer):
        self.actor = actor.to(device) 
        self.critic = critic.to(device)
        self.criticTwin =critic.to(device)
        self.calculate_drag = calculate_drag
        self.replay_buffer = replay_buffer
        self.filepaths = [
        "../data/multipleAgent/agent0/naca0012Revised.dat",
        "../data/multipleAgent/agent1/naca0012Revised.dat",
        "../data/multipleAgent/agent2/naca0012Revised.dat",
        "../data/multipleAgent/agent3/naca0012Revised.dat",
        ]
        self.max_action = max_action
        self.previous_drag = None
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=5e-4)
        #Here we use the Twin from the Twin delayed DDPG algorithm
        self.criticTwin_optimizer = optim.Adam(self.criticTwin.parameters(), lr=2e-4)
        self.noiseAction = OUNoise(action_dim=3, sigma= 0.001) # for action
        self.noise_0 = OUNoise(action_dim=3) # for param2_0, param3_0
        self.noise_3 = OUNoise(action_dim=1) # for param2_3
        self.noise_4 = OUNoise(action_dim=1) # for param2_4
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.1  # minimum exploration rate
        self.epsilon_decay = 0.999
        self.bias_for_action_0 = 0.5 #give a bias(more) for action 0

        #self.noise_scale = 0.005
        #self.noise_reduction_factor = 0.85
        self.actor_scheduler = lr_scheduler.StepLR(self.actor_optimizer, step_size=10, gamma=0.95)
        self.critic_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=10, gamma=0.95)
        self.criticTwin_scheduler = lr_scheduler.StepLR(self.criticTwin_optimizer, step_size=10, gamma=0.95)
        self.updateActor_frequency = 2
        self.update_frequency = 5
        self.update_counter = 0
        self.best_reward_so_far = [0.1, 0.1, 0.1, 0.1]
        # Create target networks
        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic_target = copy.deepcopy(critic).to(device)
        self.criticTwin_target = copy.deepcopy(critic).to(device)
        self.tauTrain = 0.01
        self.update_frequencyTrain = 2
        self.accumlateReward = [0, 0, 0, 0]
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.bias_for_action_0 = max(self.bias_for_action_0 * self.epsilon_decay, 0.05)

    def select_action(self, state):
        state = state.to(device)
        
        # Decide whether to explore or exploit
        explore = np.random.rand() < self.epsilon
        guide = np.random.rand() < self.epsilon * 0.5
        
        with torch.no_grad():
            # 如果state不是批次输入（即它只有两个维度），我们添加一个批次维度
            is_single_input = len(state.shape) == 2
            if is_single_input:
                state = state.unsqueeze(0)

            #state = torch.FloatTensor(state)
            action_probs, action_params = self.actor(state)
            action_noise = torch.tensor(self.noiseAction.sample()).to(device)
            print("Action probabilities before noise:", action_probs)
            if explore:
                action_probs = action_probs + action_noise * 100
                print("noise :", action_noise)  
                 # 给动作0增加一个偏置
                action_probs[0][0] += self.bias_for_action_0
                print("Action probabilities after bias:", action_probs)
        
                # 确保概率和为1
                action_probs = action_probs / action_probs.sum()
            else:
                action_probs = action_probs + action_noise * 10
                action_probs = action_probs / action_probs.sum()   
                epsilon = 5.0e-2  # 小的正值，用来防止 log(0)
                action_probs = torch.clamp(action_probs, min=epsilon, max=1-epsilon)  # 裁剪概率值  
                action_probs = action_probs / action_probs.sum()          
        
            # 打印动作概率
            print("Action probabilities after noise:", action_probs)
        
       
            action4execute = torch.argmax(action_probs).item()
            action_params4execute = [p.item() for p in action_params[action4execute]]

            if is_single_input:
            # 如果输入是单个状态，我们移除批次维度
                action_params = [param[0].item() for param_list in action_params for param in param_list]

            # 打印添加噪声前的参数
            print("Parameters before noise:", action_params4execute)            
            # Add Gaussian noise to the specific action parameters
            noise_indices = {
                0: [0, 1, 2],  # indices for param2_0, param3_0
                1: [0],    # index for param2_3
                2: [0]     # index for param2_4
            }
            if action4execute in noise_indices:
                for idx in noise_indices[action4execute]:
                    if action4execute == 0:
                        noise = self.noise_0.sample()
                        for i, param_idx in enumerate(noise_indices[action4execute]):
                            if explore:
                                if i == 0:
                                    action_params4execute[i] += noise[i] * 100
                                    action_params[0][0] += noise[i] * 100
                                else:
                                    action_params4execute[i] += noise[i] * 2
                                    action_params[0][i] += noise[i] * 2
                                    if guide:
                                        action_params4execute[1] = -abs(action_params4execute[1])
                                        action_params4execute[2] = abs(action_params4execute[2])
                                        action_params[0][1] = action_params4execute[1]
                                        action_params[0][2] = action_params4execute[2]
                            else:
                                action_params4execute[param_idx] += noise[i]
                                action_params[0][param_idx] += noise[i]
                    elif action4execute == 1:
                        noise = self.noise_3.sample()
                        if explore:
                            action_params4execute[0] += noise[0] * 2
                            action_params[1][0] += noise[0] * 2
                        else:
                            action_params4execute[0] += noise[0]
                            action_params[1][0] += noise[0]
                    elif action4execute == 2:
                        noise = self.noise_4.sample()
                        if explore:
                            action_params4execute[0] += noise[0] * 2
                            action_params[2][0] += noise[0] * 2
                            if guide:
                                action_params4execute[0] = abs(action_params4execute[0])
                                action_params[2][0] = action_params4execute[0]
                                
                        else:
                            action_params4execute[0] += noise[0]
                            action_params[2][0] += noise[0]

            # 打印添加噪声后的参数
            print("Parameters after noise:", action_params4execute)

        #self.noise_scale *= self.noise_reduction_factor
        
            # Clip the parameters after noise
            # Assume that parameter limits are stored in a dictionary like:
            # param_limits = {0: [(-limit_1, limit_1), (-limit_2, limit_2), (-limit_3, limit_3)], 
            #                 3: [(-limit_4, limit_4)], 
            #                 4: [(-limit_5, limit_5)]}
            # And more generally, we consider the positve and negative separately
            # For point near to the end point, we multiplicate abs(1 - action_params4execute[0]) * abs(action_params4execute[0]) to avoid parameters out of range
            param_limits = {
                0: [(0.02, 0.99), (-0.005, 0.005), (-0.005, 0.005)],
                1: [(-0.002, 0.002)],
                2: [(-0.002, 0.002)]
            }
            threshold = 0.15
            if action4execute in param_limits:
                for i, (min_val, max_val) in enumerate(param_limits[action4execute]):
                    if action4execute == 0 and (action_params4execute[0] >= 1 - threshold or action_params4execute[0] <= threshold  ) and i != 0:
                        if action_params4execute[i] > 0:
                            action_params4execute[i] = np.clip(action_params4execute[i], 0.08 * max_val  * abs(1 - action_params4execute[0]) * abs(action_params4execute[0]), max_val  * abs(1 - action_params4execute[0]) * abs(action_params4execute[0]))
                            action_params[action4execute][i] = action_params4execute[i]
                        else:
                            action_params4execute[i] = np.clip(action_params4execute[i], min_val  * abs(1 - action_params4execute[0]) * abs(action_params4execute[0]), 0.08 * max_val  * abs(1 - action_params4execute[0]) * abs(action_params4execute[0]))
                            action_params[action4execute][i] = action_params4execute[i]
                    
                    else:       
                        if action_params4execute[i] > 0:
                            action_params4execute[i] = np.clip(action_params4execute[i], 0.3 * max_val, max_val)
                            action_params[action4execute][i] = action_params4execute[i]
                        else:
                            action_params4execute[i] = np.clip(action_params4execute[i], min_val, 0.3 * min_val)
                            action_params[action4execute][i] = action_params4execute[i]
            print("Parameters after noise and clipping:", action_params4execute)
            

        return action4execute, action_params4execute ,action_probs, action_params


    def get_reward(self, agentIndex):
        current_drag = self.calculate_drag(agentIndex)
        total_points = load_and_process_data(self.filepaths[agentIndex]).cpu()
        total_points_list = total_points.squeeze(0).numpy().tolist()
        total_distance = calculate_total_distance(total_points_list)
    
        # 如果 total_distance 超出范围，返回一个惩罚值
        if not (1.9 <= total_distance <= 3.6):
            print(f"total_distance = {total_distance}")
            return -6.0

        # 否则，根据您的原始计算返回奖励
        if self.previous_drag is None:
            reward = 0.0
        else:
            reward = 100* (self.previous_drag - current_drag)
        self.previous_drag = current_drag
        
        return reward

        
        #This is the function to pad the action parameters to a consistent length
    def pad_action_params(self, action_params_batch, max_length=3):
        #print("action_params_batch type: ", type(action_params_batch))
        #print("action_params_batch: ", action_params_batch)
        padded_batch = []
        for action_params in action_params_batch:
            padded_action_params = []
            for params in action_params:  # 遍历每个动作的参数列表
                if isinstance(params, torch.Tensor):
                    params = params.cpu().numpy()
                    params_list = params.tolist()
                else:
                    params_list = params
                padded_params = params_list + [0.0] * (max_length - len(params_list))
                padded_action_params.append(padded_params)
            padded_batch.append(padded_action_params)
        return padded_batch
    
    def pad_actor_action_params(self, batch_size4actor, next_action_params):
    # The number of action dimensions
        action_dim = 3
    
    # The number of action parameter dimensions
        action_param_dim = 3
    
    # Creating a tensor filled with zeros, having the desired output shape
        batch_action_params_padded = torch.zeros((batch_size4actor, action_dim, action_param_dim), device = device)
    
    # Loop through each action dimension
        for i, action_param_list in enumerate(next_action_params):
        # Loop through each batch
            for j in range(batch_size4actor):
            # Loop through each action parameter tensor in the list to extract the necessary values
                for k, param_tensor in enumerate(action_param_list):
                # Note: we're assuming param_tensor has a shape of [batch_size4actor, 1]
                    batch_action_params_padded[j, i, k] = param_tensor[j]
                
        return batch_action_params_padded


    def train(self, iterations, agentIndex, batch_size, action_param_dim=3,discount=0.96, tau=0.005):
        print("Entered the train method.")
        
        for it in range(iterations):
            print(" For the Iteration: ", it)
            # Sample action from actor
            state = load_and_process_data(self.filepaths[agentIndex]).to(device)
            
            
            action4execute, action_params4execute, action, action_params = self.select_action(state)
            entropy = calculate_entropy(action)
            
            # Apply the action and get the new state
            success, _, warning_occurred = perform_action(self.filepaths[agentIndex], action4execute, action_params4execute)
            next_state = load_and_process_data(self.filepaths[agentIndex]).to(device)

            # Get the reward     
            reward = self.get_reward(agentIndex)
            print("During the training iteration {}, the reward is {}".format(it, reward))
            
            
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
                success, _, warning_occurred = perform_action(self.filepaths[agentIndex], action4execute, action_params4execute)  
                opposite_state = load_and_process_data(self.filepaths[agentIndex])
                opposite_reward = self.get_reward(agentIndex) + reward
                print("[OPPOSITE] Parameters in the opposite:", action_params4execute)
                if opposite_reward > reward:
                    next_state = opposite_state
                    reward = opposite_reward
                    if reward > 0:
                        reward *= 1.3
                    print(f"[OPPOSITE] Oppsite direction is better, reward = {reward}")
                else:
                    action_params4execute[1] = - 0.5 * action_params4execute[1]
                    action_params4execute[2] = - 0.5 * action_params4execute[2]
                    action_params[0][1] = action_params4execute[1]
                    action_params[0][2] = action_params4execute[2] 
                    reload_opposite_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat",  mapping_filepath=f"./agent{agentIndex}/CurrentPoint.dat", agentIndex=agentIndex,)
                    print(f"[OPPOSITE] Opposite direction is worse, reward = {reward}")  
                    
            self.accumlateReward[agentIndex] += reward
            print(f"Iteration {it}: Reward = {reward}")
            print(f"Iteration {it}: AccumlateReward = {self.accumlateReward[agentIndex]}") 
            print(self.best_reward_so_far, type(self.best_reward_so_far))

            # Store the transition in the replay buffer
            self.replay_buffer.add((state, action, action_params, next_state, reward))

            # Start training after the replay buffer is filled with enough transitions
            if len(self.replay_buffer.storage) >= batch_size:
            # Sample a batch of transitions from the replay buffer
                batch_states, batch_actions, batch_action_params, batch_next_states, batch_rewards = self.replay_buffer.sample(batch_size)
                # Stack states and next states into tensors
                batch_states = torch.cat(batch_states, dim=0).to(device)
                batch_next_states = torch.cat(batch_next_states, dim=0).to(device)
                print("batch_states shape: ", batch_states.shape)
                print("batch_next_states shape: ", batch_next_states.shape)
                # Convert actions and rewards lists to tensors
                # print("batch_actions shape before reshaping: ", batch_actions.shape)
                batch_actions = torch.stack(batch_actions).squeeze().to(device)
                #print("batch_actions shape after reshaping: ", batch_actions.shape)
                #print("batch_actions content after reshaping: ", batch_actions)
                batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device)
                #print("batch_rewards shape: ", batch_rewards.shape)
                #print("batch_rewards content: ", batch_rewards)
                # Pad action parameters and convert them to a tensor
                batch_action_params_padded = self.pad_action_params(batch_action_params)
                batch_action_params_padded = torch.tensor(batch_action_params_padded, dtype=torch.float32).to(device)
                #print("batch_action_params_padded shape after stacking: ", batch_action_params_padded.shape)
                #print("batch_action_params_padded content after stacking: ", batch_action_params_padded)
                # Get current Q estimate
                current_Q = self.critic(batch_states, batch_actions, batch_action_params_padded)
                print("current_Q shape : ", current_Q.shape)
                print("current_Q content : ", current_Q)
                # Compute the target Q value
                next_actions, next_action_params = self.actor_target(batch_next_states) 
                
                
                #print("next_actions shape: ", next_actions.shape)
                #print("next_actions content: ", next_actions)
                #next_actions = torch.argmax(next_actions, dim=1)  # Find the most probable action for each sample
                # Pad the next action parameters and convert them to a tensor
                #print("next_action_params type: ", type(next_action_params))
                #print("next_action_params: ", next_action_params)

                batch_size4actor = next_actions.shape[0]
                #print("batch_size4actor: ", batch_size4actor)
                #print_structure_and_type(next_action_params)
                next_action_params_padded = self.pad_actor_action_params(batch_size4actor, next_action_params)
                #print("next_action_params_padded shape: ", next_action_params_padded.shape)
                #print("next_action_params_padded content: ", next_action_params_padded)
                # Handle action parameters for next_action_params
                target_Q1 = self.critic_target(batch_next_states, next_actions, next_action_params_padded)
                target_Q2 = self.criticTwin_target(batch_next_states, next_actions, next_action_params_padded)
                target_Q = torch.min(target_Q1, target_Q2)
                batch_rewards = batch_rewards.unsqueeze(1).to(device)
                target_Q = batch_rewards + (discount * target_Q).detach()
                #print("target_Q shape : ", target_Q.shape)
                print("target_Q1 content : ", target_Q1)
                print("target_Q2 content : ", target_Q2)
                print("target_Q content : ", target_Q)
                # Compute critic loss
                critic_loss = F.mse_loss(current_Q, target_Q)

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                self.criticTwin_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.criticTwin_optimizer.step()
                self.critic_scheduler.step()
                self.criticTwin_scheduler.step()

                if it % self.updateActor_frequency == 0:
                    new_actions, new_action_params = self.actor(batch_states)
                    new_action_params_padded = self.pad_actor_action_params(batch_size4actor, new_action_params)
                    actor_loss = -self.critic(batch_states, new_actions, new_action_params_padded).mean() - 2.0 * entropy 
                    
                    # Optimize the actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    self.actor_scheduler.step()
                
                # Update the target networks
                # The target networks update with formular tau * theta + (1 - tau) * theta_target
                if it % self.update_frequencyTrain == 0:
                    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                        target_param.data.copy_(self.tauTrain * param.data + (1 - self.tauTrain) * target_param.data)

                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tauTrain * param.data + (1 - self.tauTrain) * target_param.data)
                    for param, target_param in zip(self.criticTwin.parameters(), self.criticTwin_target.parameters()):
                        target_param.data.copy_(self.tauTrain * param.data + (1 - self.tauTrain) * target_param.data)
                self.noise_0.reset()
                self.noise_3.reset()
                self.noise_4.reset()
        return actor_loss.item(), critic_loss.item()


