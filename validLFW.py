import torch
version = 'V1'
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
from config_mask import config
import tqdm
import os
from validate_on_LFW import evaluate_lfw
from torch.nn.modules.distance import PairwiseDistance
from Data_loader.Data_loader_facenet_mask import LFWestMask_dataloader, test_dataloader


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

model_path = r'/media/Mask_face_recognitionZ/Model_training_checkpoints/model_34_triplet_epoch_7_rocNMD0.753_rocMasked0.626maskV1.pt'
if os.path.exists(model_path) and (version in model_path):
    model_state = torch.load(model_path)
    model.load_state_dict(model_state['model_state_dict'])
    start_epoch = model_state['epoch']
    print('loaded %s' % model_path)
else:
    print('不存在预训练模型！')

l2_distance = PairwiseDistance(2).cuda()
# 出测试集准确度
print("Validating on TestDataset! ...")
model.eval() # 验证模式
with torch.no_grad():  # 不传梯度了
    distances, labels = [], []
    progress_bar = enumerate(tqdm(test_dataloader))
    for batch_index, (data_a, data_b, label) in progress_bar:
        # data_a, data_b, label这仨是一批的矩阵
        data_a = data_a.cuda()
        data_b = data_b.cuda()
        label = label.cuda()
        output_a, output_b = model(data_a), model(data_b)
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
        epoch='',
        tag='NOTMaskedLFW_aucnotmask_valid',
        version=version,
        pltshow=True
    )

print("Validating on LFWMASKTestDataset! ...")
with torch.no_grad():  # 不传梯度了
    distances, labels = [], []
    progress_bar = enumerate(tqdm(LFWestMask_dataloader))
    for batch_index, (data_a, data_b, label) in progress_bar:
        # data_a, data_b, label这仨是一批的矩阵
        data_a = data_a.cuda()
        data_b = data_b.cuda()
        label = label.cuda()
        output_a, output_b = model(data_a), model(data_b)
        distance = l2_distance.forward(output_a, output_b)
        # 列表里套矩阵
        labels.append(label.cpu().detach().numpy())
        distances.append(distance.cpu().detach().numpy())
    # 展平
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for distance in distances for subdist in distance])
    true_positive_rate_mask, false_positive_rate_mask, precision_mask, recall_mask, \
    accuracy_mask, roc_auc_mask, best_distances_mask, \
    tar_mask, far_mask = evaluate_lfw(
        distances=distances,
        labels=labels,
        epoch='',
        tag='MaskedLFW_aucmask_valid',
        version=version,
        pltshow=True
    )
# 打印日志内容
print('LFW没带口罩的结果test_log:\tAUC: {:.4f}\tACC: {:.4f}+-{:.4f}\trecall: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\t'.format(
    roc_auc,
    np.mean(accuracy),
    np.std(accuracy),
    np.mean(recall),
    np.std(recall),
    np.mean(precision),
    np.std(precision),)
)
print('\nLFW带口罩的结果test_log:\tAUC: {:.4f}\tACC: {:.4f}+-{:.4f}\trecall: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\t'.format(
    roc_auc_mask,
    np.mean(accuracy_mask),
    np.std(accuracy_mask),
    np.mean(recall_mask),
    np.std(recall_mask),
    np.mean(precision_mask),
    np.std(precision_mask),)
)