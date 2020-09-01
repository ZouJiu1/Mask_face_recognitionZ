import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import time
import torch
import torch.nn as nn
from torch.nn.modules.distance import PairwiseDistance
from Losses.Triplet_loss import TripletLoss
from validate_on_LFW import evaluate_lfw
from tqdm import tqdm
from config import config
from Data_loader.Data_loader_facenet import train_dataloader, test_dataloader
from Models.Model_for_facenet import model, optimizer_model, start_epoch, flag_train_multi_gpu

# total_time_start = time.time()
# start_epoch = start_epoch
end_epoch = start_epoch + config['epochs']
l2_distance = PairwiseDistance(2).cuda()
# best_roc_auc = 0
# best_accuracy = 0

        
model.eval()
with torch.no_grad():
    distances, labels = [], []

    print("Validating on LFW! ...")
    progress_bar = enumerate(tqdm(test_dataloader))

    for batch_index, (data_a, data_b, label) in progress_bar:

        data_a, data_b, label = data_a.cuda(), data_b.cuda(), label.cuda()

        output_a, output_b = model(data_a), model(data_b)
        distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

        distances.append(distance.cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
        # print(label.cpu().detach().numpy())

    # print(label)
    # print(labels)

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for distance in distances for subdist in distance])

    true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
            distances=distances,
            labels=labels
        )

    print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\tROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\tTAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
            np.mean(accuracy),
            np.std(accuracy),
            np.mean(precision),
            np.std(precision),
            np.mean(recall),
            np.std(recall),
            roc_auc,
            np.mean(best_distances),
            np.std(best_distances),
            np.mean(tar),
            np.std(tar),
            np.mean(far)
        )
    )