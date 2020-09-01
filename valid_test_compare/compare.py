#encoding = utf-8
import torch
from Models.CBAM_Face_attention_Resnet_maskV1 import resnet18_cbam, resnet50_cbam, resnet101_cbam, resnet34_cbam, \
    resnet152_cbam
# from Models.CBAM_Face_attention_Resnet_maskV2 import resnet18_cbam, resnet50_cbam, resnet101_cbam, resnet34_cbam, \
#     resnet152_cbam
# from Models.CBAM_Face_attention_Resnet_notmaskV3 import resnet18_cbam, resnet50_cbam, resnet101_cbam, resnet34_cbam, \
#     resnet152_cbam
import numpy as np
import dlib
import cv2
from config_mask import config
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance


if config['model'] == 18:
    model = resnet18_cbam(pretrained=True, num_classes=128)
elif config['model'] == 34:
    model = resnet34_cbam(pretrained=True, num_classes=128)
elif config['model'] == 50:
    model = resnet50_cbam(pretrained=True, num_classes=128)
elif config['model'] == 101:
    model = resnet101_cbam(pretrained=True, num_classes=128)
elif config['model'] == 152:
    model = resnet152_cbam(pretrained=True, num_classes=128)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r'/home/Mask-face-recognitionV1/Model_training_checkpoints/model_resnet34_attention_triplet_epoch_2_roc0.6185.pt'
model.load_state_dict(torch.load(model_path)['model_state_dict'])

model.eval()
test_data_transforms = transforms.Compose([
    transforms.Resize([config['image_size'], config['image_size']]), # resize
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

img1_path = r''
img2_path = r''
isame = 1
threshold = 0.9
detector = dlib.get_frontal_face_detector()
predicter_path=config['predicter_path']
predictor = dlib.shape_predictor(predicter_path)
img_size = config['image_size']

def preprocess(image_path, detector, predictor, img_size):
    image = dlib.load_rgb_image(image_path)
    face_img, TF = None, 0
    # 人脸对齐、切图
    dets = detector(image, 1)
    from PIL import Image
    if len(dets) == 1:
        faces = dlib.full_object_detections()
        faces.append(predictor(image, dets[0]))
        images = dlib.get_face_chips(image, faces, size=img_size)

        image = np.array(images[0]).astype(np.uint8)
        face_img = Image.fromarray(image).convert('RGB')  # cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # 生成人脸mask
        dets = detector(image, 1)
        if len(dets) == 1:
            point68 = predictor(image, dets[0])
            landmarks = list()
            INDEX = [0, 2, 14, 16, 17, 18, 19, 24, 25, 26]
            eyebrow_list = [19, 24]
            eyes_list = [36, 45]
            eyebrow = 0
            eyes = 0

            for eb, ey in zip(eyebrow_list, eyes_list):
                eyebrow += point68.part(eb).y
                eyes += point68.part(ey).y
            add_pixel = int(eyes / 2 - eyebrow / 2)

            for idx in INDEX:
                x = point68.part(idx).x
                if idx in eyebrow_list:
                    y = (point68.part(idx).y - 2 * add_pixel) if (point68.part(idx).y - 2 * add_pixel) > 0 else 0
                else:
                    y = point68.part(idx).y
                landmarks.append((x, y))
            belows = []
            for i in range(2, 15, 1):
                belows.append([point68.part(i).x, point68.part(i).y])
            belows = np.array(belows)
            colors = [(200, 183, 144), (163, 150, 134), (172, 170, 169), \
                      (167, 168, 166), (173, 171, 170), (161, 161, 160), \
                      (170, 162, 162)]
            cl = np.random.choice(len(colors), 1)[0]
            cv2.fillConvexPoly(face_img, belows, colors[cl])
    return face_img

def ishowm(ima, imb):
    imgone = ima.cpu().numpy()
    imgtwo = imb.cpu().numpy()
    imgall = np.concatenate([imgone, imgtwo], axis=1)
    cv2.namedWindow('images')
    cv2.resizeWindow('images', 600, 600)
    cv2.imshow('images', imgall)
    cv2.waitKey(0)
    cv2.destroyWindow('images')

ima = preprocess(img1_path, detector, predictor, img_size)
imb = preprocess(img2_path, detector, predictor, img_size)
ima, imb = test_data_transforms(ima), test_data_transforms(imb)
data_a = ima.cuda()
data_b = imb.cuda()

ishowm(ima, imb)

output_a, output_b = model(data_a), model(data_b)
l2_distance = PairwiseDistance(2).cuda()
distance = l2_distance.forward(output_a, output_b)
print('从两张图片提取出来的特征向量的欧氏距离是：%1.4f' % distance)
