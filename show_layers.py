import torch
# from torchvision.models import resnet18
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
import csv
import numpy as np

from Models.Model_for_facenet import model

from torchsummary import summary

# summary(model, input_size=(3, 200, 200))


def viz(module, input, output):
    ix = input[0].cpu()
    ix = np.array(ix)
    print('ix.shape', ix.shape)
    ix = np.squeeze(ix, axis=0)
    ix = np.transpose(ix, (1, 2, 0))
    print('ix.shape', ix.shape)
    ix = softmax(ix)
    ix = ix * 255
    cv2.imwrite('input_img.jpg',ix)

    ox = output.cpu()
    ox = np.array(ox)
    print('ox.shape', ox.shape)
    ox = np.squeeze(ox, axis=0)
    ox = np.transpose(ox, (1, 2, 0))
    print('ox.shape', ox.shape)
    ox = ox * 255
    cv2.imwrite('output_img.jpg',ox)

def softmax(x):
    w, h, c = x.shape
    x = x.flatten()
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    softmax_x = np.reshape(softmax_x, (w, h, c))
    return softmax_x

def main():
    t = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                            ])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    for name, m in model.named_modules():
        print(name)
        if name == 'model.sa2.sigmoid':
            m.register_forward_hook(viz)

    img = cv2.imread('1_face_img.jpg')
    img = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        model(img)

        # xsa = model.get_xsa(img)
        # print(len(xsa))
        # x = np.array(xsa[0])
        # x = x * 255
        # print(x.shape)
        # x = np.squeeze(x, axis=0)
        # print(x.shape)
        # x = np.transpose(x, (1, 2, 0))
        # print(x.shape)
        # cv2.imwrite('images/img0.jpg',x)

        # x = np.array(xsa[1])
        # x = x * 255
        # x = np.squeeze(x, axis=0)
        # x = np.transpose(x, (1, 2, 0))
        # with open("image_test.csv","w") as csvfile: 
        #     writer = csv.writer(csvfile)
        #     writer.writerows(x)

if __name__ == '__main__':
    main()