# 路径置顶
import sys 
import os 
sys.path.append(os.getcwd()) 
# 导入包
import torchvision.datasets as datasets
from tqdm import tqdm
import numpy as np
import dlib 
import cv2 
# 导入文件
from config_mask import config

#非LFW数据戴口罩
class NOTLFWestMaskDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, predicter_path, img_size, transform=None, test_pairs_paths=None):
        super(NOTLFWestMaskDataset, self).__init__(dir, transform)
        
        self.pairs_path = '/media/Mask_face_recognitionZ/Datasets/testpairs.txt'

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predicter_path)
        self.img_size = img_size
        self.validation_images = self.read_lfw_pairs(self.pairs_path)
        # if os.path.exists(test_pairs_paths):
        #     print("\nload lfw_paths...")
        #     self.validation_images = np.load(test_pairs_paths)
        #     print('lfw_paths loaded!')
        # else:
        #     print("\nGenerating lfw_paths...")
        #     self.validation_images = self.get_lfw_paths(dir)
        #     print('lfw_paths generated')

    def read_lfw_pairs(self, pairs_filename):
        # 把那个预制的配对文件读出来
        pairs = []
        with open(pairs_filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                pair = line.strip().split(' ')
                pairs.append(pair)
        return np.array(pairs)

    def add_extension(self, path):
        # 搞一下图片格式
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def get_lfw_paths(self, lfw_dir):
        # 读出图片对
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0 # 没配到对儿的计数
        path_list = [] # 一对儿一对儿的图片路径
        # issame_list = [] # 每一对儿立马是不是同一个人
        '''
        图片对儿张这个样亚子：
        Zico	1	2
        Zico	2	3
        Abdel_Madi_Shabneh	1	Dean_Barker	1
        Abdel_Madi_Shabneh	1	Giancarlo_Fisichella	1
        '''
        for pair in tqdm(pairs):
            # 

            if (pair[2] == '1') or (int(pair[2]) == 1):
                # 是一个人
                issame = 1
                img_0, TF_0 = self.preprocess(pair[0], self.detector, self.predictor, self.img_size)
                img_1, TF_1 = self.preprocess(pair[1], self.detector, self.predictor, self.img_size)

            elif (pair[2] == '0') or (int(pair[2]) == 0):
                # 他不是一个人
                issame = 0
                img_0, TF_0 = self.preprocess(pair[0], self.detector, self.predictor, self.img_size)
                img_1, TF_1 = self.preprocess(pair[1], self.detector, self.predictor, self.img_size)

            #这俩文件都存在就加
            if os.path.exists(pair[0]) and os.path.exists(pair[1]) and TF_0 == 1 and TF_1 == 1:
                path_list.append([pair[0], pair[1], issame])

                # issame_list.append(issame)
            #不存在就是配不上
            else:
                nrof_skipped_pairs += 1
        # 万一真有配不上的就打印一哈
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        # print("Saving testing pairs list in datasets/ directory ...")
        # np.save(config['test_pairs_paths'], path_list)
        # print("Testing pairs list Saved!\n")

        return path_list

    def preprocess(self, image_path, detector, predictor, img_size):
        image = dlib.load_rgb_image(image_path)
        # print(image_path)
        face_img, TF = None, 0
        # 人脸对齐、切图
        dets = detector(image, 1)
        from PIL import Image
        if len(dets) == 1:
            faces = dlib.full_object_detections()
            faces.append(predictor(image, dets[0]))
            images = dlib.get_face_chips(image, faces, size=img_size)

            image = np.array(images[0]).astype(np.uint8)
            face_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
        return Image.fromarray(face_img).convert('RGB')

    def __len__(self):
        return len(self.validation_images)

    def __getitem__(self, idx):
        
        def transform(img_path):
            #img = self.loader(img_path)
            img= self.preprocess(img_path, self.detector, self.predictor, self.img_size)
            return self.transform(img)

        path_1, path_2, issame = self.validation_images[idx]
        img1, img2 = transform(path_1), transform(path_2)

        # print(issame, type(issame), bool(issame), print(int(issame)), print(bool(int(issame))))
        # print(issame, bool(issame))
        if issame == '1':
            issame = True
        elif issame == '0':
            issame = False
        # print(issame)
        return img1, img2, issame


#非LFW数据集不戴口罩
class NOTLFWestNOTMaskDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, predicter_path, img_size, transform=None, test_pairs_paths=None):
        super(NOTLFWestNOTMaskDataset, self).__init__(dir, transform)

        self.pairs_path = '/media/Mask_face_recognitionZ/Datasets/testpairs.txt'

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predicter_path)
        self.img_size = img_size
        self.validation_images = self.read_lfw_pairs(self.pairs_path)
        # if os.path.exists(test_pairs_paths):
        #     print("\nload lfw_paths...")
        #     self.validation_images = np.load(test_pairs_paths)
        #     print('lfw_paths loaded!')
        # else:
        #     print("\nGenerating lfw_paths...")
        #     self.validation_images = self.get_lfw_paths(dir)
        #     print('lfw_paths generated')

    def read_lfw_pairs(self, pairs_filename):
        # 把那个预制的配对文件读出来
        pairs = []
        with open(pairs_filename, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                pair = line.strip().split(' ')
                pairs.append(pair)
        return np.array(pairs)

    def add_extension(self, path):
        # 搞一下图片格式
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def get_lfw_paths(self, lfw_dir):
        # 读出图片对
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0  # 没配到对儿的计数
        path_list = []  # 一对儿一对儿的图片路径
        # issame_list = [] # 每一对儿立马是不是同一个人
        '''
        图片对儿张这个样亚子：
        Zico	1	2
        Zico	2	3
        Abdel_Madi_Shabneh	1	Dean_Barker	1
        Abdel_Madi_Shabneh	1	Giancarlo_Fisichella	1
        '''
        for pair in tqdm(pairs):
            #

            if (pair[2] == '1') or (int(pair[2]) == 1):
                # 是一个人
                issame = 1
                img_0, TF_0 = self.preprocess(pair[0], self.detector, self.predictor, self.img_size)
                img_1, TF_1 = self.preprocess(pair[1], self.detector, self.predictor, self.img_size)

            elif (pair[2] == '0') or (int(pair[2]) == 0):
                # 他不是一个人
                issame = 0
                img_0, TF_0 = self.preprocess(pair[0], self.detector, self.predictor, self.img_size)
                img_1, TF_1 = self.preprocess(pair[1], self.detector, self.predictor, self.img_size)

            # 这俩文件都存在就加
            if os.path.exists(pair[0]) and os.path.exists(pair[1]) and TF_0 == 1 and TF_1 == 1:
                path_list.append([pair[0], pair[1], issame])

                # issame_list.append(issame)
            # 不存在就是配不上
            else:
                nrof_skipped_pairs += 1
        # 万一真有配不上的就打印一哈
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        # print("Saving testing pairs list in datasets/ directory ...")
        # np.save(config['test_pairs_paths'], path_list)
        # print("Testing pairs list Saved!\n")

        return path_list

    def preprocess(self, image_path, detector, predictor, img_size):
        image = dlib.load_rgb_image(image_path)
        # print(image_path)
        face_img, TF = None, 0
        # 人脸对齐、切图
        dets = detector(image, 1)
        from PIL import Image
        if len(dets) == 1:
            faces = dlib.full_object_detections()
            faces.append(predictor(image, dets[0]))
            images = dlib.get_face_chips(image, faces, size=img_size)

            image = np.array(images[0]).astype(np.uint8)
        return Image.fromarray(image).convert('RGB')

    def __len__(self):
        return len(self.validation_images)

    def __getitem__(self, idx):

        def transform(img_path):
            # img = self.loader(img_path)
            img = self.preprocess(img_path, self.detector, self.predictor, self.img_size)
            return self.transform(img)

        path_1, path_2, issame = self.validation_images[idx]
        img1, img2 = transform(path_1), transform(path_2)

        # print(issame, type(issame), bool(issame), print(int(issame)), print(bool(int(issame))))
        # print(issame, bool(issame))
        if issame == '1':
            issame = True
        elif issame == '0':
            issame = False
        # print(issame)
        return img1, img2, issame


class LFWestMaskDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, predicter_path, img_size, transform=None, test_pairs_paths=None):
        super(LFWestMaskDataset, self).__init__(dir, transform)

        self.pairs_path = pairs_path

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predicter_path)
        self.img_size = img_size
        if os.path.exists(test_pairs_paths):
            print("\nload lfw_paths...")
            self.validation_images = np.load(test_pairs_paths)
            print('lfw_paths loaded!')
        else:
            print("\nGenerating lfw_paths...")
            self.validation_images = self.get_lfw_paths(dir)
            print('lfw_paths generated')

    def read_lfw_pairs(self, pairs_filename):
        # 把那个预制的配对文件读出来
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def add_extension(self, path):
        # 搞一下图片格式
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def get_lfw_paths(self, lfw_dir):
        # 读出图片对
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0  # 没配到对儿的计数
        path_list = []  # 一对儿一对儿的图片路径
        # issame_list = [] # 每一对儿立马是不是同一个人
        '''
        图片对儿张这个样亚子：
        Zico	1	2
        Zico	2	3
        Abdel_Madi_Shabneh	1	Dean_Barker	1
        Abdel_Madi_Shabneh	1	Giancarlo_Fisichella	1
        '''
        for pair in tqdm(pairs):
            #

            if len(pair) == 3:
                # 是一个人
                path0 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
                issame = 1
                img_0, TF_0 = self.preprocess(path0, self.detector, self.predictor, self.img_size)
                img_1, TF_1 = self.preprocess(path1, self.detector, self.predictor, self.img_size)

            elif len(pair) == 4:
                # 他不是一个人
                path0 = self.add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
                path1 = self.add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
                issame = 0
                img_0, TF_0 = self.preprocess(path0, self.detector, self.predictor, self.img_size)
                img_1, TF_1 = self.preprocess(path1, self.detector, self.predictor, self.img_size)

            # 这俩文件都存在就加
            if os.path.exists(path0) and os.path.exists(path1) and TF_0 == 1 and TF_1 == 1:
                path_list.append([path0, path1, issame])

                # issame_list.append(issame)
            # 不存在就是配不上
            else:
                nrof_skipped_pairs += 1
        # 万一真有配不上的就打印一哈
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        print("Saving testing pairs list in datasets/ directory ...")
        np.save(config['test_pairs_paths'], path_list)
        print("Testing pairs list Saved!\n")

        return path_list

    def preprocess(self, image_path, detector, predictor, img_size):
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
            # face_img = Image.fromarray(image).convert('RGB')  # cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            face_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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
            TF = 1
        return Image.fromarray(face_img).convert('RGB'), TF

    def __len__(self):
        return len(self.validation_images)

    def __getitem__(self, idx):

        def transform(img_path):
            # img = self.loader(img_path)
            img, TF = self.preprocess(img_path, self.detector, self.predictor, self.img_size)
            return self.transform(img)

        path_1, path_2, issame = self.validation_images[idx]
        img1, img2 = transform(path_1), transform(path_2)

        # print(issame, type(issame), bool(issame), print(int(issame)), print(bool(int(issame))))
        # print(issame, bool(issame))
        if issame == '1':
            issame = True
        elif issame == '0':
            issame = False
        # print(issame)
        return img1, img2, issame


