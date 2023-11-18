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
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import StandardScaler
from bezierFit import fit_bezier_curves

def custom_min(q1, q2):
    abs_q1 = torch.abs(q1)
    abs_q2 = torch.abs(q2)
    mask = abs_q1 < abs_q2
    return torch.where(mask, q1, q2)


def calculate_entropy(probabilities):
    epsilon = 1.0e-8  # 小的正值，用来防止 log(0)
    clipped_probs = torch.clamp(probabilities, min=epsilon, max=1-epsilon)  # 裁剪概率值
    log_probabilities = torch.log(clipped_probs)  # 计算对数概率
    entropy = - torch.sum(probabilities * log_probabilities, dim=-1)  # 计算熵
    return entropy

    

class OUNoise:
    def __init__(self, mu=0, theta=0.15, sigma=0.2):
        self.mu = mu
        self.theta = theta
        self.sigma = np.array(sigma) if isinstance(sigma, list) else np.array([sigma])
        self.action_dim = len(self.sigma)
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
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
    
def save_historyInfo(mesh_filepath, airfoil_data_filepath, bodyDx, rhoDx, rhoUDx, rhoVDx, eDx,epochIndex, agentIndex):
    """
    Save the current file for current epoch to the history during iteration
    """
     # Paths to the best reward state files
    print("Save current and step to next epoch")
    history_mesh_filepath = f"./historyOpt/epoch{epochIndex}/body.mesh"
    history_airfoil_data_filepath = f"./historyOpt/epoch{epochIndex}/airfoil.dat"
    history_bodyDx = f"./historyOpt/epoch{epochIndex}/body.dx"
    history_rhoDx = f"./historyOpt/epoch{epochIndex}/rho_h.dx"
    history_rhoUDx = f"./historyOpt/epoch{epochIndex}/rho_u_h.dx"
    history_rhoVDx = f"./historyOpt/epoch{epochIndex}/rho_v_h.dx"
    history_eDx = f"./historyOpt/epoch{epochIndex}/e_h.dx"

    # Ensure the directories exist before copying files
    os.makedirs(os.path.dirname(history_mesh_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(history_airfoil_data_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(history_bodyDx), exist_ok=True)
    os.makedirs(os.path.dirname(history_rhoDx), exist_ok=True)
    os.makedirs(os.path.dirname(history_rhoUDx), exist_ok=True)
    os.makedirs(os.path.dirname(history_rhoVDx), exist_ok=True)
    os.makedirs(os.path.dirname(history_eDx), exist_ok=True)

    # Copy the current files to the opposite state files
    shutil.copy(mesh_filepath, history_mesh_filepath)
    shutil.copy(airfoil_data_filepath, history_airfoil_data_filepath)
    shutil.copy(bodyDx, history_bodyDx)
    shutil.copy(rhoDx, history_rhoDx)
    shutil.copy(rhoUDx, history_rhoUDx)
    shutil.copy(rhoVDx, history_rhoVDx)
    shutil.copy(eDx, history_eDx)    
    
def save_historyInfoPhase3(mesh_filepath, airfoil_data_filepath, bodyDx, rhoDx, rhoUDx, rhoVDx, eDx,epochIndex, agentIndex, folder_count):
    """
    Save the current file for current epoch to the history during iteration
    """
     # Paths to the best reward state files
    print("Save current and step to next epoch")
    history_mesh_filepath = f"./historyOpt/Phase3epoch_{folder_count}/body.mesh"
    history_airfoil_data_filepath = f"./historyOpt/Phase3epoch_{folder_count}/airfoil.dat"
    history_bodyDx = f"./historyOpt/Phase3epoch_{folder_count}/body.dx"
    history_rhoDx = f"./historyOpt/Phase3epoch_{folder_count}/rho_h.dx"
    history_rhoUDx = f"./historyOpt/Phase3epoch_{folder_count}/rho_u_h.dx"
    history_rhoVDx = f"./historyOpt/Phase3epoch_{folder_count}/rho_v_h.dx"
    history_eDx = f"./historyOpt/Phase3epoch_{folder_count}/e_h.dx"

    # Ensure the directories exist before copying files
    os.makedirs(os.path.dirname(history_mesh_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(history_airfoil_data_filepath), exist_ok=True)
    os.makedirs(os.path.dirname(history_bodyDx), exist_ok=True)
    os.makedirs(os.path.dirname(history_rhoDx), exist_ok=True)
    os.makedirs(os.path.dirname(history_rhoUDx), exist_ok=True)
    os.makedirs(os.path.dirname(history_rhoVDx), exist_ok=True)
    os.makedirs(os.path.dirname(history_eDx), exist_ok=True)

    # Copy the current files to the opposite state files
    shutil.copy(mesh_filepath, history_mesh_filepath)
    shutil.copy(airfoil_data_filepath, history_airfoil_data_filepath)
    shutil.copy(bodyDx, history_bodyDx)
    shutil.copy(rhoDx, history_rhoDx)
    shutil.copy(rhoUDx, history_rhoUDx)
    shutil.copy(rhoVDx, history_rhoVDx)
    shutil.copy(eDx, history_eDx) 
    
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
        self.fc1x = nn.Linear(hidden_features, 2 * hidden_features)
        self.layer_norm1x = nn.LayerNorm(2 * hidden_features)

        self.fc2x = nn.Linear(2 * hidden_features, hidden_features)
        self.layer_norm2x = nn.LayerNorm( hidden_features)
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
        out = self.fc1x(out)
        out = self.layer_norm1x(out)
        out = self.leaky_relu(out)
        
        out = self.fc2x(out)
        out = self.layer_norm2x(out)
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
    
class VAE(nn.Module):
    def __init__(self, state_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(66 * 2 * 4 , 2048) # 66 * 2 * 2 for state 66 * 2 for thickness and location
        self.fc21 = nn.Linear(2048, latent_dim)  # mean
        self.fc22 = nn.Linear(2048, latent_dim)  # log variance
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 1024)
        self.fc4 = nn.Linear( 1024, state_dim)
        
    def encode(self, x):
        batch_size = x.size(0)
        thickness = torch.zeros(batch_size, 66, 2).to(device)
        for i in range(66):
            assert (x[:, i, 0] == x[:, i + 66, 0]).all(), f"Index {i} failed the assertion"
            thickness[:, i, 0] = x[:, i, 0]
            thickness[:, i, 1] = abs(x[:, i, 1] - x[:, i + 66, 1])
        thickness = thickness.view(batch_size, -1)
        stateVec = x.view(batch_size, -1).to(device) 
        
        vectors = x[:, 1:] - x[:, :-1]
        # 为循环曲线添加最后一个点与第一个点的差值
        last_vector = x[:, 0] - x[:, -1]
        vectors = torch.cat((vectors, last_vector.unsqueeze(1)), dim=1)
        
        # 计算相邻两个向量的夹角
        angles = torch.zeros(batch_size, 66 * 2).to(device)
        for i in range( 66 * 2 ):
            v1 = vectors[:, i]
            v2 = vectors[:, (i + 1) % 66 * 2]  # 使用模运算实现循环
            cos_theta = (v1 * v2).sum(dim=-1) / (torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1))
            cos_theta = torch.clamp(cos_theta, -1, 1)  # 避免数值误差导致超出[-1, 1]范围
            angles[:, i] = torch.acos(cos_theta)
        
        h0 = torch.cat((stateVec, thickness, angles), dim=1).to(device) 
        h1 = F.selu(self.fc1(h0))
        return self.fc21(h1), self.fc22(h1)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.selu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3)).view(-1, 66*2, 2)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss for VAE: reconstruction loss + KL divergence
def loss_function(recon_x, x, mu, logvar):
    #print("recon_x shape: ", recon_x.shape)
    #print("x shape: ", x.shape)
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    print("MSE: ", MSE)
    print("KLD: ", KLD)
    return MSE + KLD

class ControlPointsAttention(nn.Module):
    def __init__(self, feature_dim = 2, attention_dim = 2):
        super(ControlPointsAttention, self).__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.query = nn.Linear(feature_dim, attention_dim)
        self.key = nn.Linear(feature_dim, attention_dim)
        self.value = nn.Linear(feature_dim, attention_dim)
        
    def forward(self, x):
        # x 的形状为 [num_control_points, features]
        Q = self.query(x)  # [num_control_points, attention_dim]
        K = self.key(x)    # [num_control_points, attention_dim]
        V = self.value(x)  # [num_control_points, attention_dim]
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.attention_dim**0.5
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重
        weighted_V = torch.matmul(attention_weights, V)
        
        return weighted_V, attention_weights


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        #self.fc = nn.Linear( 66 * 2 * 2, 256) 
        #66 for the upper, 2 for the both side, upper and lower, 2 for the x and y

        self.attention = ControlPointsAttention(feature_dim=2, attention_dim = 2)

        
        self.res_block1 = ResidualBlock(60, 1024, 60)
        self.res_block2 = ResidualBlock(60, 512, 60)
        
        self.norm1 = nn.LayerNorm(60)
        self.fc = nn.Linear(60, 256)
        self.norm2 = nn.LayerNorm(256)

        self.layer_action = nn.Linear(256, action_dim)
        self.layer_action_param = nn.Linear(256, 128)
        

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

        self.max_action = max_action

        # Initialize weights
        self._initialize_weights()

    def forward(self, state):
        batch_size = state.size(0)
        
        ActorState = state
        #print("actor_state shape: ", ActorState.shape)
        attention_output, attention_weights = self.attention(ActorState)                 
        enhanced_features = ActorState + attention_output
        flat_features = enhanced_features.view(batch_size, -1)
        
        z0 = F.selu(self.res_block1(flat_features))
        x0 = F.selu(self.res_block2(z0))
        
        x1 =self.norm2(F.selu(self.fc(self.norm1(x0))))     
    
        
        action_logits = F.selu(self.layer_action(x1))
        action_logits[:, 0] += 10.0
        action = F.softmax(action_logits, dim=-1)
        
        x2 = F.selu(self.layer_action_param(x1))

        Constriant = 0.003
        Lbound = 0
        Rbound = 1
        # 对于每个动作，使用相应的线性层生成参数
        penalized_tanh = PenalizedTanh(alpha=0.1)
        penalized_sigmoid = PenalizedSigmoid(alpha=0.5)
        param1_0 = F.selu(self.layer_param1_0(x2))
        #param1_0 = self.gluAction0(x5)
        
        # This detach is to remove the gradient calculation of sigmoid and tanh, since this is the physical constraint, we do not want it to influence the gradient calculation, which may lead to the gradient disappear
        param2_0 = 0.5 + F.softsign(self.layer_param2_0(param1_0.detach())) * 0.45
        
        param3_0 = Constriant * F.softsign(self.layer_param3_0(param1_0.detach()))
        param4_0 = Constriant * F.softsign(self.layer_param4_0(param1_0.detach()))

        param1_1 = F.selu(self.layer_param1_1(x2))
        
        param2_1 = Constriant * F.softsign(self.layer_param2_1(param1_1.detach()))

        param1_2 = F.selu(self.layer_param1_2(x2))
        param2_2 = Constriant * F.softsign(self.layer_param2_2(param1_2.detach()))

        action_params = [[param2_0, param3_0, param4_0], 
                         [param2_1], 
                         [param2_2]] 


        return action, action_params, x2, param2_0

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
    def _initialize_vae_weights(self):
        for m in self.vae.modules():
            if isinstance(m, nn.Linear):
                # 初始化VAE的权重
                nn.init.kaiming_normal_(m.weight)
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
    
class AttentionLayer(nn.Module):
    def __init__(self, dim, nhead):
        super(AttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim, nhead)
        self.cross_attn = nn.MultiheadAttention(dim, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.ReLU(),
            nn.Linear(2 * dim, dim)
        )
        self.linear1 = nn.Linear(51 , 64)
        self.linear2 = nn.Linear(10, 64)
        self.linear3 = nn.Linear(4, 64)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, state, action, action_param, max_action):
        # 拼接数据
        #print(state.dtype)
        #print(action.dtype)
        #print(action_param.dtype)
        #print(max_action.dtype)

        x = torch.cat([state.float(), action.float(), action_param.float(), max_action.float()], dim=1)
        #print(x.dtype)
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)
        # 交叉注意力
        combined_query = state.float()
        
        combined_query = F.selu(self.linear1(combined_query.float()))
        
        action_param = torch.cat([action_param.float(), max_action.float()], dim=1)
        action_param = F.selu(self.linear2(action_param.float()))
        
        action = torch.cat([action.float(), max_action.float()], dim=1)
        action = F.selu(self.linear3(action.float()))
        
        attn_output, _ = self.cross_attn(combined_query.float(), action.float(), action_param.float())
        x = x + attn_output
        x = self.norm2(x)
        # 前馈网络
        x = x + self.ffn(x)
        x = self.norm3(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim = 3, action_param_dim = 3):
        super(Critic, self).__init__()
        
        self.CriticAttention = ControlPointsAttention(feature_dim=2, attention_dim = 2)

        self.stateTransformer = nn.Linear(15 * 2 * 2, 51)
        
        self.ActionParam_norm = nn.LayerNorm(9)
        
        #self.bn_state = nn.BatchNorm1d(256)
        self.attention_layer = AttentionLayer(dim = 64, nhead = 8)
        
        self.res_blockCritic1 = ResidualBlock(64, 512, 64)
        self.res_blockCritic2 = ResidualBlock(64, 256, 64)
        
        #self.fullyConnect = nn.Linear(64, 16)
        
        # Fusion and Q-value estimation
        
        self.fusion_fc2 = nn.Linear(64, 16)
        self.fusion_fc3 = nn.Linear(16, 4)
        
        self.q_value_fc = nn.Linear(4, 1)

        # Dropout layer
        self.dropout1 = nn.Dropout(p=0.02)
        self.dropout2 = nn.Dropout(p=0.02)

        # Initialize weights
        self.apply(self._init_weights)

    def forward(self, state, action, action_params):
        
        batch_size = state.size(0)
        print("state shape: ", state.shape)
        CriticState = state
        
        
        #print("actor_state shape: ", ActorState.shape)
        attention_output, attention_weights = self.CriticAttention(CriticState)                 
        enhanced_features = CriticState + attention_output
        flat_features = enhanced_features.view(batch_size, -1)
        flat_features = self.stateTransformer(flat_features)
        

        # 选择概率最大的动作及其对应的参数
        CriticAction = action
        print("CriticAction content: ", CriticAction)
        _, max_index = torch.max(CriticAction, dim=1, keepdim=True)
        max_index = max_index.detach()
        
        CriticAction_param = action_params
        
        selected_param = torch.gather(CriticAction_param, dim=1, index=max_index.unsqueeze(-1).expand(-1, -1, CriticAction_param.size(-1)))
        selected_param = selected_param.view(batch_size, -1)
        
        CriticAction_param = CriticAction_param.view(batch_size, -1)
        CriticAction_param = self.ActionParam_norm(CriticAction_param)
        
        max_action = max_index.view(batch_size, -1)
        print("max_action content: ", max_action)
        print("selected_param content: ", selected_param)
        
        # 融合特征
        x = self.attention_layer(flat_features, CriticAction, CriticAction_param, max_action)
        #  51 for state, 3 for action, 9 for selected_param, 1 for max_action
        fusionSAP = torch.cat([flat_features.float(), CriticAction.float(), CriticAction_param.float(), max_action.float()], dim=1)
        
        #x = self.res_blockCritic1(x)
        x= F.selu(self.res_blockCritic1(x + fusionSAP))
        x= self.dropout1(x)
        x= F.selu(self.res_blockCritic2(x))
        x= self.dropout2(x)
        x = F.selu(self.fusion_fc2(x)) 
        x = F.selu(self.fusion_fc3(x)) 
        
        # Estimate the Q-value
        q_value = self.q_value_fc(x)
        
        return q_value

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
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
            
    def sample(self, batch_size, use_weighted_sampling= True):
        # 确保 batch_size 可以被 2 整除
        assert batch_size % 2 == 0, "batch_size 需要是偶数"

        half_batch = batch_size // 2

        if use_weighted_sampling:
            # 提取所有奖励
            rewards = np.array([transition[4] for transition in self.storage])

            # 计算权重（这里以指数函数为例，你可以根据需要调整）
            weights = np.exp(rewards - np.max(rewards))  # 减去最大值以避免数值问题
            weights /= np.sum(weights)  # 归一化权重

            # 根据权重进行随机采样半批次
            ind_weighted = np.random.choice(len(self.storage), size=half_batch, p=weights)
        else:
            ind = np.random.randint(0, len(self.storage), size=half_batch)

        # 进行常规随机采样另一半批次
        ind_random = np.random.randint(0, len(self.storage), size=half_batch)

        # 合并两种采样索引
        ind = np.concatenate([ind_weighted, ind_random])

        # 以下为采样逻辑，与原始代码相同
        batch_states, batch_actions, batch_action_params, batch_next_states, batch_rewards = [], [], [], [], []
        for i in ind:
            state, action, action_params, next_state, reward = self.storage[i]
            batch_states.append(state)
            batch_actions.append(action)
            batch_action_params.append(action_params)
            batch_next_states.append(next_state)
            batch_rewards.append(reward)

        # 标准化batch_action_params
        '''
        flat_list = [item for sublist in batch_action_params for item in sublist]
        data = np.array(flat_list)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)

        index = 0
        scaled_batch_action_params = []
        for sublist in batch_action_params:
            sublist_len = len(sublist)
            scaled_sublist = data_scaled[index:index+sublist_len].tolist()
            scaled_batch_action_params.append(scaled_sublist)
            index += sublist_len
        '''

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
        self.baselinepath = "../data/multipleAgent/agent0/baseline.dat"
        self.max_action = max_action
        self.previous_drag = None
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=2e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=6e-3)
        #Here we use the Twin from the Twin delayed DDPG algorithm
        self.criticTwin_optimizer = optim.Adam(self.criticTwin.parameters(), lr=3e-3)
        self.noiseAction = OUNoise(sigma= [0.001, 0.001, 0.001]) # for action
        self.noise_0 = OUNoise(sigma = [0.015, 0.000005, 0.000005]) # for param2_0, param3_0
        self.noise_3 = OUNoise(sigma = 0.000005) # for param2_3
        self.noise_4 = OUNoise(sigma = 0.000005) # for param2_4
        self.epsilon = 1.0  # initial exploration rate
        self.epsilon_min = 0.1  # minimum exploration rate
        self.epsilon_decay = 0.98
        self.bias_for_action_0 = 0.5 #give a bias(more) for action 0

        #self.noise_scale = 0.005
        #self.noise_reduction_factor = 0.85
        self.actor_scheduler = lr_scheduler.StepLR(self.actor_optimizer, step_size=2, gamma=0.98)
        self.critic_scheduler = lr_scheduler.StepLR(self.critic_optimizer, step_size=1, gamma=0.95)
        self.criticTwin_scheduler = lr_scheduler.StepLR(self.criticTwin_optimizer, step_size=1, gamma=0.95)
        #self.vae_scheduler = lr_scheduler.StepLR(self.vae_optimizer, step_size=1, gamma=0.95)
        self.updateActor_frequency = 3
        self.update_frequency = 5
        self.update_counter = 0
        self.best_reward_so_far = [0.1, 0.1, 0.1, 0.1]
        # Create target networks
        self.actor_target = copy.deepcopy(actor).to(device)
        self.critic_target = copy.deepcopy(critic).to(device)
        self.criticTwin_target = copy.deepcopy(critic).to(device)
        self.tauTrain = 0.005
        self.update_frequencyTrain = 3
        self.accumlateReward = [0, 0, 0, 0]
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.bias_for_action_0 = max(self.bias_for_action_0 * self.epsilon_decay, 0.05)

    def select_action(self, state):
        
        control_states = fit_bezier_curves(state.cpu())
        #print("control_states shape: ", control_states.shape)
        #print("control_states content: ", control_states)
        state = control_states.to(device)
        
        
        # Decide whether to explore or exploit
        explore = np.random.rand() < self.epsilon
        guide = np.random.rand() < self.epsilon 
        
        with torch.no_grad():
            # 如果state不是批次输入（即它只有两个维度），我们添加一个批次维度
            is_single_input = len(state.shape) == 2
            if is_single_input:
                state = state.unsqueeze(0)

            #state = torch.FloatTensor(state)
            action_probs, action_params, layerNODETACH, param2_0 = self.actor(state)
            action_noise = torch.tensor(self.noiseAction.sample()).to(device)
            print("Action probabilities before noise:", action_probs)
            if explore:
                action_probs = action_probs + action_noise * 100
                print("noise :", action_noise * 100)  
                 # 给动作0增加一个偏置
                action_probs[0][0] += self.bias_for_action_0
                print("Action probabilities after bias:", action_probs)
        
                # 确保概率和为1
                action_probs = action_probs / action_probs.sum()
            else:
                action_probs = action_probs + action_noise * 5
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
                                    if action_params4execute[i] < 0.3 :
                                        action_params4execute[i] += abs(noise[i] * 15)
                                        action_params[0][0] += abs(noise[i] * 15)
                                    elif action_params4execute[i] > 0.7:
                                        action_params4execute[i] -= abs(noise[i] * 15)
                                        action_params[0][0] -= abs(noise[i] * 15)
                                    else:
                                        action_params4execute[i] += noise[i] * 10
                                        action_params[0][0] += noise[i] * 10
                                else:
                                    action_params4execute[i] += noise[i] 
                                    action_params[0][i] += noise[i] 
                                    
                                    if guide:
                                        if action_params4execute[1] > 0:
                                            action_params4execute[2] = -abs(action_params4execute[2])
                                            action_params[0][2] = action_params4execute[2]
                                            
                                        else:
                                            action_params4execute[2] = abs(action_params4execute[2])
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
                0: [(0.05, 0.95), (-0.005, 0.005), (-0.005, 0.005)],
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
                            
                    elif action4execute == 0 and (action_params4execute[0] >= 1 - threshold or action_params4execute[0] <= threshold  ) and i == 0:
                        action_params4execute[i] = np.clip(action_params4execute[i], min_val, max_val)
                    
                    elif i!= 0:       
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
        if not (2.0 <= total_distance <= 3.2):
            print(f"total_distance = {total_distance}")
            return -6.0

        # 否则，根据您的原始计算返回奖励
        if self.previous_drag is None:
            reward = 0.0
        else:
            reward = 100 * (self.previous_drag - current_drag)
            if reward > 0:
                reward += ( 0.046 - current_drag) * 20
            elif 0.046 - current_drag < 0:
                reward += ( 0.046 - current_drag) * 20
            
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
            success, _, warning_occurred = perform_action(self.filepaths[agentIndex], action4execute, action_params4execute, self.baselinepath)
            next_state = load_and_process_data(self.filepaths[agentIndex]).to(device)

            # Get the reward     
            reward = self.get_reward(agentIndex)
            print("During the training iteration {}, the reward is {}".format(it, reward))
            
            
            if warning_occurred:
                reward -= 0.5
            elif reward > 0:
                if action4execute == 0:
                    reward *= 1.3
                else: 
                    reward *= 1.0
                    
            if action4execute == 0 and reward < 0:
                print(f"[OPPSITE] Now we try the opposite direction")
                save_opposite_state(mesh_filepath=f"./agent{agentIndex}/body.mesh", airfoil_data_filepath=f"../data/multipleAgent/agent{agentIndex}/naca0012Revised.dat",  mapping_filepath=f"./agent{agentIndex}/CurrentPoint.dat", agentIndex=agentIndex,)
                action_paramsSaved = action_params
                
                action_params4execute[1] = - 2 * action_params4execute[1]
                action_params4execute[2] = - 2 * action_params4execute[2]
                action_params[0][1] = action_params4execute[1]
                action_params[0][2] = action_params4execute[2]     
                success, _, warning_occurred = perform_action(self.filepaths[agentIndex], action4execute, action_params4execute, self.baselinepath)  
                opposite_state = load_and_process_data(self.filepaths[agentIndex])
                opposite_reward = self.get_reward(agentIndex) + reward
                print("[OPPOSITE] Parameters in the opposite:", action_params4execute)
                if opposite_reward > reward:
                    ###################################################################################
                    # The oppsite state is continue to be used, now we add the origin to replay buffer
                    # The next state has not been changed, this is the origin next state
                    self.replay_buffer.add((state, action, action_paramsSaved, next_state, reward))
                    ###################################################################################
                    next_state = opposite_state
                    reward = opposite_reward
                    if reward > 0:
                        reward *= 1.3
                    print(f"[OPPOSITE] Oppsite direction is better, reward = {reward}")
                else:
                    ###################################################################################
                    # The original state is continue to be used, now we add the opposite to replay buffer
                    self.replay_buffer.add((state, action, action_params, opposite_state, opposite_reward))
                    ###################################################################################
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
                #print("batch_actions shape before reshaping: ", batch_actions.shape)
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
                
                control_states = fit_bezier_curves(batch_states.cpu())
                control_states = control_states.to(device)
                #print("control_states shape: ", control_states.shape)
                #print("control_states content: ", control_states)
                
                
                
                # Get current Q estimate
                current_Q1= self.critic(control_states, batch_actions, batch_action_params_padded)
                print("current_Q1 shape : ", current_Q1.shape)
                print("current_Q1 content : ", current_Q1)
                current_Q2= self.criticTwin(control_states, batch_actions, batch_action_params_padded)
                print("current_Q2 shape : ", current_Q2.shape)
                print("current_Q2 content : ", current_Q2)
                current_Q = custom_min(current_Q1, current_Q2)
                
                # Compute the target Q value
                #print("batch_next_states content: ", batch_next_states)
                next_actions, next_action_params, BATCHlayerNODETACH, BATCHparam2_0 = self.actor_target(control_states) 
                
                
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
                
                explore = np.random.rand() < self.epsilon
                action_noise = torch.tensor(self.noiseAction.sample()).to(device)
                if explore:
                    # 为整个batch添加噪音
                    #next_actions = next_actions + action_noise * 100
                    # 打印噪音
                    #print("noise :", action_noise)  
                    # 为动作0添加偏差
                    next_actions[:, 0] += self.bias_for_action_0
                    # 打印添加偏差后的动作概率
                    print("Action probabilities after bias:", next_actions)
                    # 确保概率和为1
                    next_actions = F.softmax(next_actions, dim=1)
                #else:
                    # 为整个batch添加噪音
                    #action_probs = action_probs + action_noise * 10
                    # 裁剪概率值
                    #epsilon = 5.0e-2  # 小的正值，用来防止 log(0)
                    #action_probs = torch.clamp(action_probs, min=epsilon, max=1-epsilon)
                    # 确保概率和为1
                    #action_probs = F.softmax(action_probs, dim=1)

                #print("next_action_params_padded shape: ", next_action_params_padded.shape)
                #print("next_action_params_padded content: ", next_action_params_padded)
                # Handle action parameters for next_action_params
                
                next_control_states = fit_bezier_curves(batch_next_states.cpu())
                next_control_states = next_control_states.to(device)
                
                target_Q1 = self.critic_target(next_control_states, next_actions, next_action_params_padded)
                target_Q2 = self.criticTwin_target(next_control_states, next_actions, next_action_params_padded)
                target_Q = custom_min(target_Q1, target_Q2)
                batch_rewards = batch_rewards.unsqueeze(1).to(device)
                print("batch_rewards content : ", batch_rewards)
                target_Q= batch_rewards + (discount * target_Q).detach()
                #print("target_Q shape : ", target_Q.shape)
                print("target_Q1 content : ", target_Q1)
                print("target_Q2 content : ", target_Q2)
                print("target_Q content : ", target_Q)
                # Compute critic loss
                critic_loss = F.smooth_l1_loss(current_Q, target_Q, beta=0.5) 

                # Optimize the critic
                self.critic_optimizer.zero_grad()
                self.criticTwin_optimizer.zero_grad()
                critic_loss.backward()
                clip_grad_norm_(self.actor.parameters(), max_norm=2.0)
                self.critic_optimizer.step()
                self.criticTwin_optimizer.step()
                
                # update the critic target network, every time the critic is updated, the critic target network is updated as well
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tauTrain * param.data + (1 - self.tauTrain) * target_param.data)

                for param, target_param in zip(self.criticTwin.parameters(), self.criticTwin_target.parameters()):
                    target_param.data.copy_(self.tauTrain * param.data + (1 - self.tauTrain) * target_param.data)
                

                if (it + 1) % self.updateActor_frequency == 0 or warning_occurred:
                    new_actions, new_action_params, layerNODETACH, param2_0 = self.actor(control_states)
                    #batch_states4loss = batch_states.view(batch_size, -1)
                    #vae_loss = loss_function(batch_recon_state, batch_states, batch_mu, batch_logvar)
                    #InitialVAE = vae_loss
                    new_action_params_padded = self.pad_actor_action_params(batch_size4actor, new_action_params)
                    #print( "vae_loss: ", vae_loss)
                    
                    regularization = torch.norm(layerNODETACH, p=2)
                    print("regularization: ", regularization)
                    
                    #now we add penalty for param2_0
                    distance = torch.min(param2_0 - 0.04, 0.96 - param2_0)
                    epsilon4penalty = 0.04
                    
                    penalty = torch.where(torch.abs(distance) < epsilon4penalty, 1 / (torch.abs(distance) + 1e-6), torch.zeros_like(distance))

                    penalty_loss = torch.norm(penalty, p=2)
                    print("penalty_loss: ", penalty_loss)
                    
                        
                    #actor_loss = -self.critic(batch_states, new_actions, new_action_params_padded).mean() - 2.0 * entropy + 0.0003 * regularization + penalty_loss
                    mean_value= self.critic(control_states, new_actions, new_action_params_padded)
                    print("mean_value: ", mean_value) 
                    #actor_loss = - mean_value.mean() - 2.0 * entropy + 0.0003 * regularization
                    actor_loss = - mean_value.mean()+ 0.0003 * regularization - 2.0 * entropy
                     
                    # Optimize the VAE
                    '''
                    for vae_turn in range(100):
                        VAE_actions, VAE_action_params, VAE_recon_state, VAE_mu, VAE_logvar, _, _= self.actor(batch_states)
                        vae_loss = loss_function(VAE_recon_state, batch_states, VAE_mu, VAE_logvar)
                        #print("VAE turn: ", vae_turn, "VAE loss: ", vae_loss)
                        if vae_loss <max( 0.01 * InitialVAE, 0.1 * abs(actor_loss)) or vae_turn == 99:
                            self.vae_optimizer.zero_grad()
                            vae_loss.backward(retain_graph = True)
                            break
                        self.vae_optimizer.zero_grad()
                        vae_loss.backward()
                        clip_grad_norm_(self.actor.vae.parameters(), max_norm=5.0)
                        self.vae_optimizer.step()
                        '''
                        
                        
                        
                    torch.autograd.set_detect_anomaly(True)     
                    
                    # Optimize the actor
                    self.actor_optimizer.zero_grad()
                    #self.vae_optimizer.zero_grad()   
                    #actor_loss.backward()
                    torch.autograd.set_detect_anomaly(True)      
                    actor_loss.backward()
                    clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
                    self.actor_optimizer.step()
                    #self.vae_optimizer.step()

        

                    
                
                # Update the target networks
                # The target networks update with formular tau * theta + (1 - tau) * theta_target
                if (it + 1) % self.update_frequencyTrain == 0:
                    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                        target_param.data.copy_(self.tauTrain * param.data + (1 - self.tauTrain) * target_param.data)

                self.noise_0.reset()
                self.noise_3.reset()
                self.noise_4.reset()
                
            if warning_occurred:
                break
                
        return actor_loss.item(), critic_loss.item()


