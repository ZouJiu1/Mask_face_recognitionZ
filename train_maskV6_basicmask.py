# 路径置顶
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append(os.getcwd())
# 导入包
from torch.nn.modules.distance import PairwiseDistance
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import time
# 导入文件
# from Models.Model_for_facenet import model, optimizer_model, start_epoch, flag_train_multi_gpu
from Data_loader.Data_loader_facenet_mask import train_dataloader, test_dataloader, LFWestMask_dataloader
from Losses.Triplet_loss import TripletLoss
from validate_on_LFW import evaluate_lfw
from config_mask import config
# from Models.Attention_resnet_lossinforward import Resnet34_Triplet,ResNet,resnet18
from Models.Resnet34 import resnet34

print("Using {} model architecture.".format(config['model']))
start_epoch = 0

model = resnet34(pretrained=True)

model_path = r'/media/Mask_face_recognitionZ/Model_training_checkpoints'
x = [int(i.split('_')[4]) for i in os.listdir(model_path) if 'V6' in i]
x.sort()
for i in os.listdir(model_path):
    if (len(x)!=0) and ('epoch_'+str(x[-1]) in i) and ('V6' in i):
        model_path = os.path.join(model_path, i)
        break

if os.path.exists(model_path) and ('V6' in model_path):
    model_state = torch.load(model_path)
    # model.load_state_dict(model_state['model_state_dict'])
    start_epoch = model_state['epoch']

    now_state_dict = model.state_dict()
    state_dict = {k: v for k, v in model_state.items() if (k in now_state_dict.keys()) and \
                  ('fc.weight' not in now_state_dict.keys())}
    now_state_dict.update(state_dict)
    # now_state_dict.update(pretrained_state_dict)
    model.load_state_dict(now_state_dict)
    print('loaded %s' % model_path)
else:
    print('不存在预训练模型！')

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

def adjust_learning_rate(optimizer, epoch):
    if epoch<30:
        lr =  0.125
    elif (epoch>=30) and (epoch<60):
        lr = 0.0625
    elif (epoch >= 60) and (epoch < 90):
        lr = 0.0155
    elif (epoch >= 90) and (epoch < 120):
        lr = 0.003
    elif (epoch>=120) and (epoch < 160):
        lr = 0.0001
    else:
        lr = 0.00006
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_optimizer(model, new_lr):
    # setup optimizer
    if config['optimizer'] == "sgd":
        optimizer_model = torch.optim.SGD(model.parameters(), lr = new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=0)
    elif config['optimizer'] == "adagrad":
        optimizer_model = torch.optim.Adagrad(model.parameters(), lr = new_lr,
                                  lr_decay=1e-4,
                                  weight_decay=0)

    elif config['optimizer'] == "rmsprop":
        optimizer_model = torch.optim.RMSprop(model.parameters(), lr = new_lr)

    elif config['optimizer'] == "adam":
        optimizer_model = torch.optim.Adam(model.parameters(), lr = new_lr,
                               weight_decay=0)
    return optimizer_model

# 随机种子
seed = 0
optimizer_model = create_optimizer(model, 0.125)
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# 打卡时间、epoch
total_time_start = time.time()
start_epoch = start_epoch
end_epoch = start_epoch + config['epochs']
# 导入l2计算的
l2_distance = PairwiseDistance(2).cuda()
# 为了打日志先预制个最佳auc和最佳acc在前头
best_roc_auc = -1
best_accuracy = -1
print('Countdown 3 seconds')
time.sleep(1)
print('Countdown 2 seconds')
time.sleep(1)
print('Countdown 1 seconds')
time.sleep(1)

# epoch大循环
for epoch in range(start_epoch, end_epoch):
    print("\ntraining on TrainDataset! ...")
    epoch_time_start = time.time()
    triplet_loss_sum = 0
    attention_loss_sum = 0
    num_hard = 0

    model.train()  # 训练模式
    # step小循环
    progress_bar = enumerate(tqdm(train_dataloader))
    for batch_idx, (batch_sample) in progress_bar:
        # for batch_idx, (batch_sample) in enumerate(train_dataloader):
        # length = len(train_dataloader)
        # fl=open('/home/Mask-face-recognition/output.txt', 'w')
        # for batch_idx, (batch_sample) in enumerate(train_dataloader):
        # print(batch_idx, end=' ')
        # fl.write(str(batch_idx)+' '+str(round((time.time()-epoch_time_start)*length/((batch_idx+1)*60), 2))+'；  ')
        # 获取本批次的数据
        # 取出三张人脸图(batch*图)
        anc_img = batch_sample['anc_img'].cuda()
        pos_img = batch_sample['pos_img'].cuda()
        neg_img = batch_sample['neg_img'].cuda()
        # 取出三张mask图(batch*图)
        position_anc = batch_sample['mask_anc'].cuda()
        position_pos = batch_sample['mask_pos'].cuda()
        position_neg = batch_sample['mask_neg'].cuda()

        # 模型运算
        # 前向传播过程-拿模型分别跑三张图，生成embedding和loss（在训练阶段的输入是两张图，输出带loss，而验证阶段输入一张图，输出只有embedding）
        anc_embedding = model(anc_img)
        pos_embedding  = model(pos_img)
        neg_embedding = model(neg_img)
        anc_embedding = torch.div(anc_embedding, torch.norm(anc_embedding)) * 50
        pos_embedding = torch.div(pos_embedding, torch.norm(pos_embedding)) * 50
        neg_embedding = torch.div(neg_embedding, torch.norm(neg_embedding)) * 50
        # print(99999999, anc_embedding.size())
        # 寻找困难样本
        # 计算embedding的L2
        pos_dist = l2_distance.forward(anc_embedding, pos_embedding)
        neg_dist = l2_distance.forward(anc_embedding, neg_embedding)
        # 找到满足困难样本标准的样本
        all = (neg_dist - pos_dist < config['margin']).cpu().numpy().flatten()
        hard_triplets = np.where(all == 1)
        if len(hard_triplets[0]) == 0:
            continue

        # 选定困难样本——困难embedding
        anc_hard_embedding = anc_embedding[hard_triplets].cuda()
        pos_hard_embedding = pos_embedding[hard_triplets].cuda()
        neg_hard_embedding = neg_embedding[hard_triplets].cuda()
        # 选定困难样本——困难样本对应的attention loss

        # 损失计算
        # 计算这个批次困难样本的三元损失
        triplet_loss = TripletLoss(margin=config['margin']).forward(
            anchor=anc_hard_embedding,
            positive=pos_hard_embedding,
            negative=neg_hard_embedding
        ).cuda()
        # triplet_loss = TripletLoss(margin=config['margin']).forward(
        #     anchor=anc_embedding,
        #     positive=pos_embedding,
        #     negative=neg_embedding
        # ).cuda()
        # 计算这个批次困难样本的attention loss（这个loss实际上在forward过程里已经计算了，这里就是整合一下求个mean）
        # 计算总顺势
        LOSS = triplet_loss
        # LOSS = triplet_loss

        # 反向传播过程
        optimizer_model.zero_grad()
        LOSS.backward()
        optimizer_model.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer_model, epoch)

        # 记录log相关信息
        # 计算本个批次内的困难样本数量
        num_hard += len(anc_hard_embedding)
        # 计算这个epoch内的总三元损失和计算损失所用的困难样本个数
        triplet_loss_sum += triplet_loss.item()
        # if batch_idx>10:
        #     break
    # if batch_idx==9:
    # tim = time.time() - epoch_time_start
    # print("需要的时间是：",round((tim*length)/600,2),"分钟")
    # fl.close()

    # 计算这个epoch里的平均损失
    avg_triplet_loss = 0 if (num_hard == 0) else triplet_loss_sum / num_hard
    avg_attention_loss = 0 if (num_hard == 0) else attention_loss_sum / num_hard
    avg_loss = avg_triplet_loss + avg_attention_loss
    epoch_time_end = time.time()

    # 出测试集准确度
    print("Validating on TestDataset! ...")
    model.eval()  # 验证模式
    with torch.no_grad():  # 不传梯度了
        distances, labels = [], []
        progress_bar = enumerate(tqdm(test_dataloader))
        for batch_index, (data_a, data_b, label) in progress_bar:
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
            epoch = 'epoch_'+str(epoch),
            tag = 'NOTMaskedLFW_auc',
            version = 'V6',
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
            output_a = torch.div(output_a, torch.norm(output_a))
            output_b = torch.div(output_b, torch.norm(output_b))
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
            epoch = 'epoch_'+str(epoch),
            tag = 'MaskedLFW_auc',
            version = 'V6',
            pltshow=True
        )

    # 打印并保存日志
    # 从之前的文件里读出来最好的roc和acc，并进行更新
    if os.path.exists('logs/lfw_{}_log_tripletmaskV6.txt'.format(config['model'])):
        with open('logs/lfw_{}_log_tripletmaskV6.txt'.format(config['model']), 'r') as f:
            lines = f.readlines()
            my_line = lines[-3]
            my_line = my_line.split('\t')
            best_roc_auc = float(my_line[3].split(':')[1])
            best_accuracy = float(my_line[5].split(':')[1])

    # 确定什么时候保存权重：最后一个epoch就保存，AUC出现新高就保存
    save = True
    if config['save_last_model'] and epoch == end_epoch - 1:
        save = True
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        save = True
    if np.mean(accuracy) > best_accuracy:
        best_accuracy = np.mean(accuracy)
    if epoch % 3 == 0:
        save = True
    print('save: ', save)

    # 打印不戴口罩日志内容
    print('Epoch {}:\n \
               train_log:\tLOSS: {:.3f}\ttri_loss: {:.3f}\tatt_loss: {:.3f}\thard_sample: {}\ttrain_time: {}\n \
               NOTMASK_LFW_test_log:\tAUC: {:.3f}\tACC: {:.3f}+-{:.3f}\trecall: {:.3f}+-{:.3f}\tPrecision {:.3f}+-{:.3f}\t'.format(
        epoch + 1,
        avg_loss,
        avg_triplet_loss,
        avg_attention_loss,
        num_hard,
        (epoch_time_end - epoch_time_start) / 3600,
        roc_auc,
        np.mean(accuracy),
        np.std(accuracy),
        np.mean(recall),
        np.std(recall),
        np.mean(precision),
        np.std(precision),
    )
    )
    # 打印戴口罩日志内容
    print('Epoch {}:\n \
               train_log:\tLOSS: {:.3f}\ttri_loss: {:.3f}\tatt_loss: {:.3f}\thard_sample: {}\ttrain_time: {}\n \
               MASKED_LFW_test_log:\tAUC: {:.3f}\tACC: {:.3f}+-{:.3f}\trecall: {:.3f}+-{:.3f}\tPrecision {:.3f}+-{:.3f}\t'.format(
        epoch + 1,
        avg_loss,
        avg_triplet_loss,
        avg_attention_loss,
        num_hard,
        (epoch_time_end - epoch_time_start) / 3600,
        roc_auc_mask,
        np.mean(accuracy_mask),
        np.std(accuracy_mask),
        np.mean(recall_mask),
        np.std(recall_mask),
        np.mean(precision_mask),
        np.std(precision_mask),
    )
    )

    # 保存日志文件
    with open('logs/lfw_{}_log_tripletmaskV6.txt'.format(config['model']), 'a') as f:
        val_list = [
            'epoch: ' + str(epoch + 1) + '\t',
            'train:\t',
            'LOSS: ' + str('%.3f' % avg_loss) + '\t',
            'tri_loss: ' + str('%.3f' % avg_triplet_loss) + '\t',
            'att_loss: ' + str('%.3f' % avg_attention_loss) + '\t',
            'hard_sample: ' + str(num_hard) + '\t',
            'train_time: ' + str('%.3f' % ((epoch_time_end - epoch_time_start) / 3600))
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n')
        val_list = [
            'epoch: ' + str(epoch + 1) + '\t',
            'test:\t',
            'auc_masked: ' + str('%.3f' % roc_auc_mask) + '\t',
            'best_auc_MD: ' + str('%.3f' % best_roc_auc) + '\t',
            'acc_MD: ' + str('%.3f' % np.mean(accuracy_mask)) + '+-' + str('%.3f' % np.std(accuracy_mask)) + '\t',
            'best_acc_MD: ' + str('%.3f' % best_accuracy) + '\t',
            'recall_MD: ' + str('%.3f' % np.mean(recall_mask)) + '+-' + str('%.3f' % np.std(recall_mask)) + '\t',
            'precision_MD: ' + str('%.3f' % np.mean(precision_mask)) + '+-' + str('%.3f' % np.std(precision_mask)) + '\t',
            'best_distances_MD: ' + str('%.3f' % np.mean(best_distances_mask)) + '+-' + str(
                '%.3f' % np.std(best_distances_mask)) + '\t',
            'tar_m: ' + str('%.3f' % np.mean(tar_mask)) + '\t',
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n')
        val_list = [
            'epoch: ' + str(epoch + 1) + '\t',
            'test:\t',
            'auc: ' + str('%.3f' % roc_auc) + '\t',
            'best_auc: ' + str('%.3f' % best_roc_auc) + '\t',
            'acc: ' + str('%.3f' % np.mean(accuracy)) + '+-' + str('%.3f' % np.std(accuracy)) + '\t',
            'best_acc: ' + str('%.3f' % best_accuracy) + '\t',
            'recall: ' + str('%.3f' % np.mean(recall)) + '+-' + str('%.3f' % np.std(recall)) + '\t',
            'precision: ' + str('%.3f' % np.mean(precision)) + '+-' + str('%.3f' % np.std(precision)) + '\t',
            'best_distances: ' + str('%.3f' % np.mean(best_distances)) + '+-' + str(
                '%.3f' % np.std(best_distances)) + '\t',
            'tar_m: ' + str('%.3f' % np.mean(tar)) + '\t',
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n')
        val_list = [
            'epoch: ' + str(epoch + 1) + '\t',
            'config:\t',
            'LR: ' + str(config['Learning_rate']) + '\t',
            'optimizer: ' + str(config['optimizer']) + '\t',
            'embedding_dim: ' + str(config['embedding_dim']) + '\t',
            'pretrained: ' + str(config['pretrained']) + '\t',
            'image_size: ' + str(config['image_size'])
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n' + '\n')

    # 保存模型权重
    if save:
        state = {
            'epoch': epoch + 1,
            'embedding_dimension': config['embedding_dim'],
            'batch_size_training': config['train_batch_size'],
            'model_state_dict': model.state_dict(),
            'model_architecture': config['model'],
            'optimizer_model_state_dict': optimizer_model.state_dict()
        }
        #
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()
        # For storing best euclidean distance threshold during LFW validation
        # if flag_validate_lfw:
        # state['best_distance_threshold'] = np.mean(best_distances)
        #
        torch.save(state, 'Model_training_checkpoints/model_{}_triplet_epoch_{}_rocNotMasked{:.3f}_rocMasked{:.3f}maskV6.pt'.format(config['model'],
                                                                                                     epoch + 1,
                                                                                                     roc_auc, roc_auc_mask))
# Training loop end
total_time_end = time.time()
total_time_elapsed = total_time_end - total_time_start
print("\nTraining finished: total time elapsed: {:.2f} hours.".format(total_time_elapsed / 3600))
