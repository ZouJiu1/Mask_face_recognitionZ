import torch
import os
import sys
from torch.nn.modules.distance import PairwiseDistance
l2_distance = PairwiseDistance(2)#.cuda()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pwd = os.path.abspath('./')

version = 'V3'
mask = False #是否给人脸戴口罩
if version=='V1':
    from Models.CBAM_Face_attention_Resnet_maskV1 import resnet18_cbam, resnet50_cbam, resnet101_cbam, resnet34_cbam, \
        resnet152_cbam
elif version=='V2':
    from Models.CBAM_Face_attention_Resnet_maskV2 import resnet18_cbam, resnet50_cbam, resnet101_cbam, resnet34_cbam, \
        resnet152_cbam
elif version=='V3':
    from Models.CBAM_Face_attention_Resnet_notmaskV3 import resnet18_cbam, resnet50_cbam, resnet101_cbam, resnet34_cbam, \
        resnet152_cbam
import numpy as np
import tqdm
from config_mask import config
from validate_on_LFW import evaluate_lfw
from torch.nn.modules.distance import PairwiseDistance
from Data_loader.Data_loader_facenet_mask import NOTLFWestMask_dataloader
from Data_loader.Data_loader_facenet_notmask import NOTLFWestNOTMask_dataloader

if config['model'] == 18:
    model = resnet18_cbam(pretrained=True, showlayer= False,num_classes=128)
elif config['model'] == 34:
    model = resnet34_cbam(pretrained=True, showlayer= False, num_classes=128)
elif config['model'] == 50:
    model = resnet50_cbam(pretrained=True, showlayer= False, num_classes=128)
elif config['model'] == 101:
    model = resnet101_cbam(pretrained=True, showlayer= False, num_classes=128)
elif config['model'] == 152:
    model = resnet152_cbam(pretrained=True, showlayer= False, num_classes=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(pwd, 'Model_training_checkpoints')
x = [int(i.split('_')[4]) for i in os.listdir(model_path) if version in i]
x.sort()
for i in os.listdir(model_path):
    if (len(x)!=0) and ('epoch_'+str(x[-1]) in i) and (version in i):
        model_pathi = os.path.join(model_path, i)
        break
if version=='V1':
    model_pathi = os.path.join(model_path, 'model_34_triplet_epoch_30_rocNotMasked0.819_rocMasked0.764maskV1.pt')
elif version=='V2':
    model_pathi = os.path.join(model_path, 'model_34_triplet_epoch_27_rocNotMasked0.919_rocMasked0.798notmaskV2.pt')
elif version=='V3':
    model_pathi = os.path.join(model_path, 'model_34_triplet_epoch_97_rocNotMasked0.951_rocMasked0.766notmaskV3.pt')
elif version=='V6':
    model_pathi = os.path.join(model_path, 'model_34_triplet_epoch_63_rocNotMasked0.922_rocMasked0.834maskV6.pt')
elif version=='V8':
    model_pathi = os.path.join(model_path, 'model_34_triplet_epoch_39_rocNotMasked0.926_rocMasked0.856maskV8.pt')
elif version=='V9':
    model_pathi = os.path.join(model_path, 'model_34_triplet_epoch_19_rocNotMasked0.918_rocMasked0.831notmaskV9.pt')
print(model_path)
if os.path.exists(model_pathi) and (version in model_pathi):
    if torch.cuda.is_available():
        model_state = torch.load(model_pathi)
    else:
        model_state = torch.load(model_pathi, map_location='cpu')
    model.load_state_dict(model_state['model_state_dict'])
    start_epoch = model_state['epoch']
    print('loaded %s' % model_pathi)
else:
    print('不存在预训练模型！')
    sys.exit(0)

if torch.cuda.is_available():
    model.cuda()

model.eval()
print('开始测试验证集')
with torch.no_grad(): # 不传梯度了
    distances, labels = [], []
    # progress_bar = enumerate(tqdm(NOTLFWestMask_dataloader))
    # for batch_index, (data_a, data_b, label) in progress_bar:
    for batch_index, (data_a, data_b, label) in enumerate(NOTLFWestNOTMask_dataloader):
        # data_a, data_b, label这仨是一批的矩阵
        data_a = data_a.cuda()
        data_b = data_b.cuda()
        label = label.cuda()
        output_a, output_b = model(data_a), model(data_b)
        output_a = torch.div(output_a, torch.norm(output_a))
        output_b = torch.div(output_b, torch.norm(output_b))
        distance = l2_distance.forward(output_a, output_b)
        # 列表里套矩阵
        labels.append(label.cpu().detach().numpy())
        distances.append(distance.cpu().detach().numpy())
    # 展平
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for distance in distances for subdist in distance])
    true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
    tar, far = evaluate_lfw(
        distances=distances,
        labels=labels,
        pltshow=True
    )

# 打印日志内容
print('test_log:\tAUC: {:.4f}\tACC: {:.4f}+-{:.4f}\trecall: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\t'.format(
    roc_auc,
    np.mean(accuracy),
    np.std(accuracy),
    np.mean(recall),
    np.std(recall),
    np.mean(precision),
    np.std(precision))+'\tbest_distance:{:.4f}\t'.format(np.mean(best_distances))
)

with torch.no_grad(): # 不传梯度了
    distances, labels = [], []
    # progress_bar = enumerate(tqdm(NOTLFWestMask_dataloader))
    # for batch_index, (data_a, data_b, label) in progress_bar:
    for batch_index, (data_a, data_b, label) in enumerate(NOTLFWestMask_dataloader):
        # data_a, data_b, label这仨是一批的矩阵
        data_a = data_a.cuda()
        data_b = data_b.cuda()
        label = label.cuda()
        output_a, output_b = model(data_a), model(data_b)
        output_a = torch.div(output_a, torch.norm(output_a))
        output_b = torch.div(output_b, torch.norm(output_b))
        distance = l2_distance.forward(output_a, output_b)
        # 列表里套矩阵
        labels.append(label.cpu().detach().numpy())
        distances.append(distance.cpu().detach().numpy())
    # 展平
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for distance in distances for subdist in distance])
    true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
    tar, far = evaluate_lfw(
        distances=distances,
        labels=labels,
        pltshow=True
    )

# 打印日志内容
print('MASKED_test_log:\tAUC: {:.4f}\tACC: {:.4f}+-{:.4f}\trecall: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\t'.format(
    roc_auc,
    np.mean(accuracy),
    np.std(accuracy),
    np.mean(recall),
    np.std(recall),
    np.mean(precision),
    np.std(precision))+'\tbest_distance:{:.4f}\t'.format(np.mean(best_distances))
)