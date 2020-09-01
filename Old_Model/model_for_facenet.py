import sys
import os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from config import config
from models.resnet18 import Resnet18Triplet
from models.resnet34 import Resnet34Triplet
from models.resnet50 import Resnet50Triplet
from models.resnet101 import Resnet101Triplet
from models.inceptionresnetv2 import InceptionResnetV2Triplet
from models.resnet34_cbam import Resnet34_CBAM_Triplet


print("Using {} model architecture.".format(config['model']))
start_epoch = 0
if config['model'] == "resnet18":
    model = Resnet18Triplet(
        embedding_dimension=config['embedding_dim'],
        pretrained=config['pretrained']
    )
elif config['model'] == "resnet34":
    model = Resnet34Triplet(
        embedding_dimension=config['embedding_dim'],
        pretrained=config['pretrained']
    )
elif config['model'] == "resnet50":
    model = Resnet50Triplet(
        embedding_dimension=config['embedding_dim'],
        pretrained=config['pretrained']
    )
elif config['model'] == "resnet101":
    model = Resnet101Triplet(
        embedding_dimension=config['embedding_dim'],
        pretrained=config['pretrained']
    )
elif config['model'] == "inceptionresnetv2":
    model = InceptionResnetV2Triplet(
        embedding_dimension=config['embedding_dim'],
        pretrained=config['pretrained']
    )

elif config['model'] == "resnet34_cbam":
    model = Resnet34_CBAM_Triplet(
        embedding_dimension=config['embedding_dim'],
        pretrained=config['pretrained']
    )


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

# model
# optimizer_model
# start_epoch
# flag_train_multi_gpu
