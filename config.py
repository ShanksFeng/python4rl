# config.py
import torch
#device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")