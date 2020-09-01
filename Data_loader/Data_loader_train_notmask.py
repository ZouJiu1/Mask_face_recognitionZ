# 路径置顶
import sys
import os

sys.path.append(os.getcwd())
# 导入包
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import numpy as np
import torch
import dlib
import cv2
# 导入文件
# from Data_preprocessing.Preprocess import preprocess
from config_notmask import config


# 训练数据
class TrainDataset(Dataset):
    def __init__(self, face_dir, mask_dir, csv_name, num_triplets, predicter_path, img_size,
                 training_triplets_path=None, transform=None):
        # 初始化
        self.df = pd.read_csv(csv_name, dtype={'id': object, 'name': object, 'class': int})
        self.face_dir = face_dir
        self.mask_dir = mask_dir
        self.num_triplets = num_triplets
        self.transform = transform

        # self.detector = dlib.get_frontal_face_detector()
        # self.predictor = dlib.shape_predictor(predicter_path)
        # self.img_size = img_size

        # 如果有配好的三元组数据就直接用，要不就生成一下
        if os.path.exists(training_triplets_path):
            print("\nload {} triplets...".format(num_triplets))
            self.training_triplets = np.load(training_triplets_path)
            print('{} triplets loaded!'.format(num_triplets))
        else:
            self.training_triplets = self.generate_triplets(self, self.df, self.num_triplets)

    # 静态方法
    @staticmethod
    def generate_triplets(self, df, num_triplets):
        '''
        生成三元组数据
        每个三元组包括锚样本、正样本、负样本，其中锚样本和正样本属于同一个类（人），负样本属于另一个类（人）

        输入：原始数据的列表信息和所要的对数
        '''
        print("\nGenerating {} triplets:".format(num_triplets))

        def make_dictionary_for_face_class(df):
            # df包括：id,name,class三列，对应的图片名称、人名、类别
            face_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in face_classes:
                    face_classes[label] = []
                # 往字典中的这个类里加图片名称
                face_classes[label].append(df.iloc[idx, 0])
            # face_classes = {'class0': [class0_id0, class0_id1, ...], 'class1': [class1_id0, ...], ...}
            return face_classes

        triplets = []
        classes = df['class'].unique()
        print("Generating face_classes...")
        face_classes = make_dictionary_for_face_class(df)
        print("Generating npy file...")
        # 做进度条用的
        # progress_bar = tqdm(range(num_triplets))
        # for _ in progress_bar:
        progress_bar = tqdm(range(num_triplets))
        for _ in progress_bar:

            # FIXME 类和人名其实是一一对应的，只保留一个说不定就行
            # 随机选两个类当做正类负类
            pos_class = np.random.choice(classes)
            neg_class = np.random.choice(classes)
            # 如果选出来的正类里的图片数少于2就重新选一个正类
            while len(face_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            # 如果选出来的正负类相等就重新选个负类
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            # FIXME 这个name其实到后来没啥用啊。。
            # 选出这个类对应的人名
            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]

            # 如果正类里的图片数等于2就分别取出来当做锚样本、正样本（取index）
            if len(face_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            # 绕不然就随机取俩
            else:
                ianc = np.random.randint(0, len(face_classes[pos_class]))
                ipos = np.random.randint(0, len(face_classes[pos_class]))
                # 取出来的俩一样的话，就重新拿个正样本
                while ianc == ipos:
                    ipos = np.random.randint(0, len(face_classes[pos_class]))

            ineg = np.random.randint(0, len(face_classes[neg_class]))

            triplets.append(
                [
                    face_classes[pos_class][ianc],  # 锚样本图片名称
                    face_classes[pos_class][ipos],  # 正样本图片名称
                    face_classes[neg_class][ineg],  # 负样本图片名称
                    pos_class,  # 正类
                    neg_class,  # 负类
                    pos_name,  # 正类人名
                    neg_name  # 负类人名
                ]
            )

        print("Saving training triplets list in datasets/ directory ...")
        np.save(config['train_triplets_path'], triplets)
        print("Training triplets' list Saved!\n")

        # 这里返回是因为第一次生成的时候就直接用返回的而不直接读文件了
        return triplets

    # def preprocess(self, image_path, detector, predictor, img_size):
    #     image = dlib.load_rgb_image(image_path)
    #     face_img, mask_img, TF = None, None, 1
    #     # 人脸对齐、切图
    #     dets = detector(image, 1)
    #     if len(dets) == 1:
    #         faces = dlib.full_object_detections()
    #         faces.append(predictor(image, dets[0]))
    #         images = dlib.get_face_chips(image, faces, size=img_size)

    #         image = np.array(images[0]).astype(np.uint8)
    #         face_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #         # 生成人脸mask
    #         dets = detector(image, 1)
    #         if len(dets) == 1:
    #             point68 = predictor(image, dets[0])
    #             landmarks = list()
    #             INDEX = [0,1,2,14,15,16,17,18,19,20,23,24,25,26]
    #             eyebrow_list = [18,20,23,25]
    #             eyes_list = [36,39,42,45]
    #             eyebrow = 0
    #             eyes = 0

    #             for eb, ey in zip(eyebrow_list, eyes_list):
    #                 eyebrow += point68.part(eb).y
    #                 eyes += point68.part(ey).y
    #             add_pixel = int(eyes/4 - eyebrow/4)

    #             for idx in INDEX:
    #                 x = point68.part(idx).x
    #                 if idx in eyebrow_list:
    #                     y = (point68.part(idx).y - 2*add_pixel) if (point68.part(idx).y - 2*add_pixel) > 0 else 0
    #                 else:
    #                     y = point68.part(idx).y
    #                 landmarks.append((x,y))
    #             landmarks = np.array(landmarks)
    #             hull = cv2.convexHull(landmarks)
    #             # print(hull.shape)
    #             mask = np.zeros(face_img.shape, dtype=np.uint8)
    #             mask_img = cv2.fillPoly(mask, [hull], (255, 255, 255))

    #     if np.max(face_img) is None or np.max(mask_img) is None:
    #         TF = 0
    #     return face_img, mask_img, TF

    def __len__(self):
        return len(self.training_triplets)

    def add_extension(self, path):
        # 文件格式比较迷，可能有这两种情况
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = self.add_extension(os.path.join(self.face_dir, str(pos_name), str(anc_id)))
        pos_img = self.add_extension(os.path.join(self.face_dir, str(pos_name), str(pos_id)))
        neg_img = self.add_extension(os.path.join(self.face_dir, str(neg_name), str(neg_id)))

        anc_mask = self.add_extension(os.path.join(self.mask_dir, str(pos_name), str(anc_id)))
        pos_mask = self.add_extension(os.path.join(self.mask_dir, str(pos_name), str(pos_id)))
        neg_mask = self.add_extension(os.path.join(self.mask_dir, str(neg_name), str(neg_id)))
        # face_anc, mask_anc, TF_anc = self.preprocess(anc_img, self.detector, self.predictor, self.img_size)
        # face_pos, mask_pos, TF_pos = self.preprocess(pos_img, self.detector, self.predictor, self.img_size)
        # face_neg, mask_neg, TF_neg = self.preprocess(neg_img, self.detector, self.predictor, self.img_size)

        # if TF_anc + TF_pos + TF_neg == 3:
        #     judge = False
        #     # print('成功！')
        # else:
        #     index = np.random.randint(0, len(self.training_triplets))
        #     print('失败！')
        anc_img = Image.open(anc_img).convert('RGB')
        pos_img = Image.open(pos_img).convert('RGB')
        neg_img = Image.open(neg_img).convert('RGB')

        anc_mask = cv2.imread(anc_mask, cv2.IMREAD_GRAYSCALE)
        pos_mask = cv2.imread(pos_mask, cv2.IMREAD_GRAYSCALE)
        neg_mask = cv2.imread(neg_mask, cv2.IMREAD_GRAYSCALE)

        # face_anc = Image.fromarray(cv2.cvtColor(face_anc,cv2.COLOR_BGR2RGB))
        # face_pos = Image.fromarray(cv2.cvtColor(face_pos,cv2.COLOR_BGR2RGB))
        # face_neg = Image.fromarray(cv2.cvtColor(face_neg,cv2.COLOR_BGR2RGB))

        # mask_anc = Image.fromarray(mask_anc)
        # mask_pos = Image.fromarray(mask_pos)
        # mask_neg = Image.fromarray(mask_neg)

        # 把类转成tensor
        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {
            'anc_img': anc_img,
            'pos_img': pos_img,
            'neg_img': neg_img,

            'mask_anc': anc_mask,
            'mask_pos': pos_mask,
            'mask_neg': neg_mask,

            'pos_class': pos_class,
            'neg_class': neg_class
        }

        # 如果做转换就转换一下下
        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        return sample












