import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from config import config

from Models.Attention_resnet_lossinforward import Resnet34_Triplet


print("Using {} model architecture.".format(config['model']))
start_epoch = 0

if config['model'] == "resnet34_cheahom":
    model = Resnet34_Triplet(
        embedding_dimension=config['embedding_dim'])


flag_train_gpu = torch.cuda.is_available()
flag_train_multi_gpu = False
if flag_train_gpu and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.cuda()
    flag_train_multi_gpu = True
    print('Using multi-gpu training.')
elif flag_train_gpu and torch.cuda.device_count() == 1:
    model.cuda()
    print('Using single-gpu training.')

# optimizer
print("Using {} optimizer.".format(config['optimizer']))
if config['optimizer'] == "sgd":
    optimizer_model = torch.optim.SGD(model.parameters(), lr=config['Learning_rate'])
    
elif config['optimizer'] == "adagrad":
    optimizer_model = torch.optim.Adagrad(model.parameters(), lr=config['Learning_rate'])
    
elif config['optimizer'] == "rmsprop":
    optimizer_model = torch.optim.RMSprop(model.parameters(), lr=config['Learning_rate'])
    
elif config['optimizer'] == "adam":
    optimizer_model = torch.optim.Adam(model.parameters(), lr=config['Learning_rate'])

# old model
if os.path.isfile(config['resume_path']):
    print("\nLoading checkpoint {} ...".format(config['resume_path']))
    checkpoint = torch.load(config['resume_path'])
    start_epoch = checkpoint['epoch']
    if flag_train_multi_gpu:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
    print("\nCheckpoint loaded: start epoch from checkpoint = {}\nRunning for {} epochs.\n".format(start_epoch,config['epochs']+start_epoch))

