# 路径置顶
import sys 
import os 
sys.path.append(os.getcwd()) 
# 导入包 
import torchvision.transforms as transforms 
import torch 
# 导入文件
from Data_loader.Data_loader_test_mask import TestDataset
from Data_loader.Data_loader_train_mask import TrainDataset
from Data_loader.Data_loadertest_mask import NOTLFWestMaskDataset,LFWestMaskDataset
from config_mask import config

# 训练数据的变换
train_data_transforms = transforms.Compose([
    # transforms.Resize([config['image_size'], config['image_size']]), # resize
    #transforms.RandomHorizontalFlip(), # 随机翻转
    transforms.ToTensor(), # 变成tensor
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# 测试数据的变换
test_data_transforms = transforms.Compose([
    transforms.Resize([config['image_size'], config['image_size']]), # resize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])
# 训练数据生成器
# train_dataloader = torch.utils.data.DataLoader(
#     dataset=TrainDataset(
#         face_dir=config['train_data_path'],
#         mask_dir = config['mask_data_path'],
#         csv_name=config['train_data_index'],
#         num_triplets=config['num_train_triplets'],
#         training_triplets_path=config['train_triplets_path'],
#         transform=train_data_transforms,
#         predicter_path=config['predicter_path'],
#         img_size=config['image_size']
#     ),
#     batch_size=config['train_batch_size'],
#     num_workers=config['num_workers'],
#     shuffle=False
# )
# 训练数据生成器
train_dataloader = torch.utils.data.DataLoader(
    dataset = TrainDataset(
        face_dir = config['train_data_path'],
        mask_dir = config['mask_data_path'],
        csv_name = config['train_data_index'],
        num_triplets = config['num_train_triplets'],
        training_triplets_path = config['train_triplets_path'],
        transform = train_data_transforms,
        txt_mask='txt',
        predicter_path = config['predicter_path'],
        img_size = config['image_size']
    ),
    batch_size = config['train_batch_size'],
    num_workers = config['num_workers'],
    shuffle = False
)

# 测试数据生成器
test_dataloader = torch.utils.data.DataLoader(
    dataset = TestDataset(
        dir = config['LFW_data_path'],
        pairs_path = config['LFW_pairs'],
        predicter_path = config['predicter_path'],
        img_size = config['image_size'],
        transform = test_data_transforms,
        test_pairs_paths = config['test_pairs_paths']
    ),
    batch_size = config['test_batch_size'],
    num_workers = config['num_workers'],
    shuffle = False
)

# LFW戴口罩测试数据生成器
LFWestMask_dataloader = torch.utils.data.DataLoader(
    dataset=LFWestMaskDataset(
        dir=config['LFW_data_path'],
        pairs_path=config['LFW_pairs'],
        predicter_path=config['predicter_path'],
        img_size=config['image_size'],
        transform=test_data_transforms,
        test_pairs_paths=config['test_pairs_paths']
    ),
    batch_size=config['test_batch_size'],
    num_workers=config['num_workers'],
    shuffle=False
)

# 训练数据生成器
from config_notmask import config as notcf
V9_train_dataloader = torch.utils.data.DataLoader(
    dataset = TrainDataset(
        face_dir = config['train_data_path'],
        mask_dir = notcf['mask_data_path'],
        csv_name = config['train_data_index'],
        num_triplets = config['num_train_triplets'],
        training_triplets_path = config['train_triplets_path'],
        transform = train_data_transforms,
        txt_mask='mask',
        predicter_path = config['predicter_path'],
        img_size = config['image_size']
    ),
    batch_size = config['train_batch_size'],
    num_workers = config['num_workers'],
    shuffle = False
)

# # 非LFW戴口罩测试数据生成器
NOTLFWestMask_dataloader = torch.utils.data.DataLoader(
    dataset = NOTLFWestMaskDataset(
        dir=config['LFW_data_path'],
        pairs_path=config['LFW_pairs'],
        predicter_path=config['predicter_path'],
        img_size=config['image_size'],
        transform=test_data_transforms,
        test_pairs_paths=config['test_pairs_paths']
    ),
    batch_size=config['test_batch_size'],
    num_workers=config['num_workers'],
    shuffle=False
)



