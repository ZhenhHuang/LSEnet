import torch
import numpy as np
import os
import random
import argparse
from exp import Exp
from logger import create_logger


seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description='Hyperbolic Structural Entropy')

# Experiment settings
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--is_graph', type=bool, default=True)
parser.add_argument('--root_path', type=str, default='./dataset')
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--eval_freq', type=int, default=10)
parser.add_argument('--exp_iters', type=int, default=1)
parser.add_argument('--version', type=str, default="run")
parser.add_argument('--log_path', type=str, default="./results/v2302152230/cls_Cora.log")

parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--height', type=int, default=3)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--w_decay', type=float, default=5e-6)
parser.add_argument('--embed_dim', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--nonlin', type=str, default='relu')
parser.add_argument('--kappa', type=float, default=-1.0, help='curvature of simple manifolds')
parser.add_argument('--temperature', type=float, default=0.2)
parser.add_argument('--n_cluster_trials', type=int, default=5)

parser.add_argument('--patience', type=int, default=50, help='early stopping patience')
parser.add_argument('--save_path', type=str, default='model.pt')
# GPU
parser.add_argument('--use_gpu', action='store_false', help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--devices', type=str, default='0,1',
                    help='device ids of multiple gpus')

configs = parser.parse_args()
log_path = f"./results/{configs.version}/{configs.dataset}.log"
configs.log_path = log_path
if not os.path.exists(f"./results"):
    os.mkdir("./results")
if not os.path.exists(f"./results/{configs.version}"):
    os.mkdir(f"./results/{configs.version}")
print(f"Log path: {configs.log_path}")
logger = create_logger(configs.log_path)
logger.info(configs)

exp = Exp(configs)
exp.train()
torch.cuda.empty_cache()
