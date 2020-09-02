# import minpy.numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Function
import cv2

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
# class Mask_BCE_Loss(Function):
#     def __init__(self):
#         super(TripletLoss, self).__init__()
#         # self.mask = mask
#         # self.predict = predict
#     def forward(self, mask, predict):

#         loss = list()
#         for one_pre_forward, one_mask in zip(predict, mask):
#             for one_pre_mask in one_pre_forward:
#                 shape = one_pre_mask.shape
#                 one_mask = cv2.resize(one_mask, shape)
#                 one_mask = torch.from_numpy(one_mask)
#                 one_loss = F.binary_cross_entropy(predict, one_mask)
#                 loss.append(one_loss)

#         Loss = torch.mean(loss)
#         return Loss

class Attention_loss(nn.Module):
    def forward(self, hot_map, mask):
        Loss = list()

        for one_hot_map, one_mask in zip(hot_map, mask):

            shape = one_hot_map.shape[2]

            one_mask = one_mask.cpu()
            one_mask = np.array(one_mask)/255
            # print(one_mask.shape)
            # one_mask = np.transpose(one_mask, (1, 2, 0))
            # one_mask = cv2.cvtColor(one_mask, cv2.COLOR_RGB2GRAY)
            one_mask = cv2.resize(one_mask, (shape, shape))
            one_mask = one_mask[np.newaxis, :]
            # print(one_mask.shape)
            #from torchvision import transforms
            #trn = transforms.ToTensor()
            #rez = transforms.Resize([shape,shape])
            #one_mask = trn(one_mask)
            #one_mask =rez(one_mask)
            #one_mask.unsqueeze_(1)

            one_mask = torch.from_numpy(one_mask)
            one_mask = one_mask.cuda().float()

            one_loss = F.binary_cross_entropy(one_hot_map, one_mask)
            Loss.append(one_loss)

        Loss = np.array(Loss, dtype=np.float64)
        Loss = torch.from_numpy(Loss)
        return Loss



# class Attention_loss(nn.Module):
#     def forward(self, hot_map_list, mask):
#         Loss = []

#         mask = mask.cpu()
#         mask = np.array(mask)
#         # hot_map_list = np.array(hot_map_list)
#         hot_map_list = hot_map_list

#         for hot_map in hot_map_list:
#             for one_mask, one_hotmap in zip(mask, hot_map):
#                 one_hotmap = one_hotmap.cpu()
#                 # one_hotmap = np.array(one_hotmap)
#                 shape = one_hotmap.shape[2]
#                 one_mask = cv2.cvtColor(one_mask,cv2.COLOR_RGB2GRAY)
#                 one_mask = cv2.resize(one_mask, (shape, shape))

#                 one_mask = torch.from_numpy(one_mask)
#                 one_mask = one_mask.cuda()
#                 one_hotmap = torch.from_numpy(one_hotmap)
#                 one_hotmap = one_hotmap.cuda()

#                 loss = F.binary_cross_entropy(one_hotmap, one_mask)
#                 Loss.append(loss)
#         Loss = np.array(Loss, dtype=np.float64)
#         Loss = torch.from_numpy(Loss)
#         return torch.mean(Loss)

class LevelAttention_loss(nn.Module):

    def forward(self, img_batch_shape, attention_mask, bboxs):

        h, w = img_batch_shape[2], img_batch_shape[3]

        mask_losses = []

        batch_size = bboxs.shape[0]
        for j in range(batch_size):
            bbox_annotation = bboxs[j, :, :]
            # bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            cond1 = torch.le(bbox_annotation[:, 0], w)
            cond2 = torch.le(bbox_annotation[:, 1], h)
            cond3 = torch.le(bbox_annotation[:, 2], w)
            cond4 = torch.le(bbox_annotation[:, 3], h)
            cond = cond1 * cond2 * cond3 * cond4

            bbox_annotation = bbox_annotation[cond, :]

            if bbox_annotation.shape[0] == 0:
                mask_losses.append(torch.tensor(0).float().cuda())
                continue

            bbox_area = (bbox_annotation[:, 2] - bbox_annotation[:, 0]) * (bbox_annotation[:, 3] - bbox_annotation[:, 1])

            mask_loss = []
            for id in range(len(attention_mask)):
                attention_map = attention_mask[id][j, 0, :, :]
                min_area = (2 ** (id + 5)) ** 2 * 0.5
                max_area = (2 ** (id + 5) * 1.58) ** 2 * 2

                level_bbox_indice1 = torch.ge(bbox_area, min_area)
                level_bbox_indice2 = torch.le(bbox_area, max_area)

                level_bbox_indice = level_bbox_indice1 * level_bbox_indice2

                level_bbox_annotation = bbox_annotation[level_bbox_indice, :].clone()

                #level_bbox_annotation = bbox_annotation.clone()

                attention_h, attention_w = attention_map.shape

                if level_bbox_annotation.shape[0]:
                    level_bbox_annotation[:, 0] = torch.tensor(level_bbox_annotation[:,0].cpu().numpy()[0]*(attention_w / w)).long().cuda()
                    level_bbox_annotation[:, 1] = torch.tensor(level_bbox_annotation[:,1].cpu().numpy()[0]*(attention_h / h)).long().cuda()
                    level_bbox_annotation[:, 2] = torch.tensor(level_bbox_annotation[:,2].cpu().numpy()[0]*(attention_w / w)).long().cuda()
                    level_bbox_annotation[:, 3] = torch.tensor(level_bbox_annotation[:,3].cpu().numpy()[0]*(attention_h / h)).long().cuda()
                mask_gt = torch.zeros(attention_map.shape)
                mask_gt = mask_gt.cuda()

                for i in range(level_bbox_annotation.shape[0]):

                    x1 = max(int(level_bbox_annotation[i, 0]), 0)
                    y1 = max(int(level_bbox_annotation[i, 1]), 0)
                    x2 = min(math.ceil(level_bbox_annotation[i, 2]) + 1, attention_w)
                    y2 = min(math.ceil(level_bbox_annotation[i, 3]) + 1, attention_h)

                    mask_gt[y1:y2, x1:x2] = 1

                mask_gt = mask_gt[mask_gt >= 0]
                mask_predict = attention_map[attention_map >= 0]

                mask_loss.append(F.binary_cross_entropy(mask_predict, mask_gt))
            mask_losses.append(torch.stack(mask_loss).mean())
        return torch.stack(mask_losses)#.mean(dim=0, keepdim=True)








