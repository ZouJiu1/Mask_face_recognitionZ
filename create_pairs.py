import numpy as np
import os
import itertools
from PIL import Image
import os
import dlib
import cv2
pwd = os.path.join(os.path.abspath('./'), 'Datasets')
from config_mask import config

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config['predicter_path'])

def preprocess(image_path):
    image = dlib.load_rgb_image(image_path)
    # print(image_path)
    face_img, TF = None, 0
    # 人脸对齐、切图
    dets = detector(image, 1)
    if len(dets) == 1:
        faces = dlib.full_object_detections()
        faces.append(predictor(image, dets[0]))
        images = dlib.get_face_chips(image, faces, size=config['image_size'])

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
            return 111
        else:
            return None
    else:
        return None

def samechoice(lists, path, allsame, f, num):
    length = len(lists)
    count = 0
    for i in range(100000):
        dir = lists[i%length]
        dir_path = os.path.join(path, dir)
        files = [os.path.join(dir_path, dp) for dp in os.listdir(dir_path)]
        if len(files)==1:
            continue
        choice = tuple(np.random.choice(files, 2, replace=False))
        cho = (choice[1], choice[0])
        r1 = preprocess(cho[0])
        r2 = preprocess(cho[1])
        if (r1 == None) or (r2 == None):
            continue
        if (choice in allsame) or (cho in allsame):
            continue
        else:
            allsame.add(choice)
            f.write(choice[0]+' '+choice[1]+' 1\n')
            count += 1
        if count==num:
            return

def notsamechoice(lists, path, allnotsame, f, num):
    count = 0
    for i in range(100000):
        for i in itertools.combinations(lists, 2):
            dir_pathone = os.path.join(path, i[0])
            dir_pathtwo = os.path.join(path, i[1])
            filesone = [os.path.join(dir_pathone, dp) for dp in os.listdir(dir_pathone)]
            filestwo = [os.path.join(dir_pathtwo, dp) for dp in os.listdir(dir_pathtwo)]
            choiceone = np.random.choice(filesone, 1)[0]
            choicetwo = np.random.choice(filestwo, 1)[0]
            choice = (choiceone, choicetwo)
            cho = (choicetwo, choiceone)
            r1 = preprocess(cho[0])
            r2 = preprocess(cho[1])
            if (r1==None) or (r2==None):
                continue
            if (choice in allnotsame) or (cho in allnotsame):
                continue
            else:
                allnotsame.add(choice)
                f.write(choice[0] + ' ' + choice[1] + ' 0\n')
                count += 1
            if count == num:
                return

path = os.path.join(pwd, 'tmp')
lists = [os.path.join(path, i) for i in os.listdir(path)]
allsame = set()
allnotsame = set()
f = open(os.path.join(pwd, 'testpairs.txt'), 'w')
Kfold = 10
num = 300
for i in range(Kfold):
    samechoice(lists, path, allsame, f, num)
    notsamechoice(lists, path, allnotsame, f, num)
f.close()



