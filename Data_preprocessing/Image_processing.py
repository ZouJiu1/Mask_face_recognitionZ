# 路径置顶
import sys
import copy
import os 
sys.path.append(os.getcwd()) 
# 导入包
from tqdm import tqdm 
import numpy as np 
import dlib 
import cv2
import json
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-s1', '--segment1',type=int, default = 0)
parser.add_argument('-s2', '--segment2', type=int, default=10)
parser.add_argument('-sa', '--segmentall', type=int, default=10)
args = parser.parse_args()
if (args.segmentall<args.segment2) or (args.segment1>=args.segment2) or (args.segmentall<args.segment1):
    print('分片设置有误！')
    sys.exit(-1)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

pwd = os.path.abspath('../')

print('dlib.DLIB_USE_CUDA:', dlib.DLIB_USE_CUDA)
print('dlib.cuda.get_num_devices():', dlib.cuda.get_num_devices())

def preprocess(image_path, saveimg_path, savemask_path, saveimg_path_notmask, savemask_path_notmask, \
               detector, predictor, img_size, masked, notmasked):
    global jsonfile
    image = dlib.load_rgb_image(image_path)
    face_img, mask_img = None, None
    # 人脸对齐、切图
    dets = detector(image, 1)
    name, id_ = image_path.split(os.sep)[-2:]
    id_ = id_.split('.')[0]
    xmin = None
    xmax = None
    ymin = None
    ymax = None
    if len(dets) == 1:
        faces = dlib.full_object_detections()
        faces.append(predictor(image, dets[0]))
        images = dlib.get_face_chips(image, faces, size=img_size)

        image = np.array(images[0]).astype(np.uint8)
        face_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        origin_face_img = copy.deepcopy(face_img)
        # 生成人脸mask
        dets = detector(image, 1)
        if len(dets) == 1:
            point68 = predictor(image, dets[0])
            landmarks = list()
            INDEX = [0,2,14,16,17,18,19,24,25,26]
            eyebrow_list = [19, 24]
            eyes_list = [36, 45]
            eyebrow = 0
            eyes = 0

            for eb, ey in zip(eyebrow_list, eyes_list):
                eyebrow += point68.part(eb).y
                eyes += point68.part(ey).y
            add_pixel = int(eyes/2 - eyebrow/2)

            for idx in INDEX:
                x = point68.part(idx).x
                if idx in eyebrow_list:
                    y = (point68.part(idx).y - 2*add_pixel) if (point68.part(idx).y - 2*add_pixel) > 0 else 0
                else:
                    y = point68.part(idx).y
                landmarks.append((x, y))
            # for i in range(2, 14, 1):
            #     cv2.line(image, (point68.part(i).x, point68.part(i).y), \
            #              (point68.part(i+1).x, point68.part(i+1).y), [255, 0, 255], 1)
            # cv2.line(image, (point68.part(14).x-2, point68.part(14).y+6), \
            #              (point68.part(2).x+2, point68.part(2).y+6), [255, 0, 255], 1)
            # cv2.line(image, (point68.part(14).x, point68.part(14).y), \
            #          (point68.part(2).x, point68.part(2).y), [0, 0, 255], 1)
            belows = []
            for i in range(2, 15, 1):
                belows.append([point68.part(i).x, point68.part(i).y])
            belows = np.array(belows)
            colors = [(200, 183, 144), (163, 150, 134), (172, 170, 169), \
                      (167, 168, 166), (173, 171, 170), (161, 161, 160), \
                      (170, 162, 162)]
            cl = np.random.choice(len(colors), 1)[0]
            cv2.fillConvexPoly(face_img, belows, colors[cl])
            # cv2.imshow('img', image)
            # cv2.waitKey(0)
            landmarks = np.array(landmarks)
            hull = cv2.convexHull(landmarks)
            mask = np.zeros(origin_face_img.shape, dtype=np.uint8)
            mask_img = cv2.fillPoly(mask, [hull], [255, 255, 255])

            lm = np.array(landmarks)
            h, w, c = face_img.shape
            xmin = int(max(0, np.min(lm[:, 0])))
            xmax = int(min(np.max(lm[:, 0]), w))
            # ymin = np.min(lm[:, 1])
            ymax = int(min(np.max(lm[:, 1]), h))
    # if np.max(face_img) is not None and np.max(mask_img) is not None:
    if np.max(face_img) is not None and xmin is not None:
        if notmasked:
            cv2.imwrite(savemask_path_notmask, mask_img)
        if masked:
            with open(savemask_path, 'w') as obj:
                obj.write(str(xmin) + ',0,'+str(xmax)+','+str(ymax))
                obj.write('\n')
        if masked:
            cv2.imwrite(saveimg_path, face_img)
        if notmasked:
            cv2.imwrite(saveimg_path_notmask, origin_face_img)
        # if name not in jsonfile.keys():
        #     jsonfile[name] = {id_: [xmin, 0, xmax, ymax]}
        # else:
        #     jsonfile[name].update({id_: [xmin, 0, xmax, ymax]})

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
img_size = 256

data_path = os.path.join(pwd, f'Datasets{os.sep}vggface2_train')
# data_path = '/data/face-datasets/vggface2_train'
face_path_mask = os.path.join(pwd, f'Datasets{os.sep}vggface2_train_face_mask')
face_path_notmask = os.path.join(pwd, f'Datasets{os.sep}vggface2_train_face_notmask')
mask_path_mask = os.path.join(pwd, f'Datasets{os.sep}vggface2_train_mask_mask')
mask_path_notmask = os.path.join(pwd, f'Datasets{os.sep}vggface2_train_mask_notmask')
if not os.path.exists(face_path_mask):
    os.mkdir(face_path_mask)
if not os.path.exists(mask_path_mask):
    os.mkdir(mask_path_mask)
if not os.path.exists(face_path_notmask):
    os.mkdir(face_path_notmask)
if not os.path.exists(mask_path_notmask):
    os.mkdir(mask_path_notmask)

files_list = os.listdir(data_path)
length = len(files_list)
files_list = files_list[(length*args.segment1//args.segmentall):(length*args.segment2//args.segmentall)]
# preprocess(r'C:\Users\EDZ\Desktop\1.jpeg',\
#            r'C:\Users\EDZ\Desktop\projects\wwwww.jpg',\
#            r'C:\Users\EDZ\Desktop\projects\mmM.jpg',\
#            detector, predictor, img_size)

def read_json(jsonfile):
    with open(jsonfile, 'r', encoding = 'utf-8') as f:
        file = json.load(f)
    return file

#是否生成戴口罩图片和相应mask
masked = True

#是否生成不戴口罩的图片和相应mask
notmasked = True

for man in tqdm(files_list):
    pic_list = os.listdir(os.path.join(data_path, man))
    if masked:
        if not os.path.exists(os.path.join(mask_path_mask, man)):
            os.mkdir(os.path.join(mask_path_mask, man))
        if not os.path.exists(os.path.join(face_path_mask, man)):
            os.mkdir(os.path.join(face_path_mask, man))
    if notmasked:
        if not os.path.exists(os.path.join(mask_path_notmask, man)):
            os.mkdir(os.path.join(mask_path_notmask, man))
        if not os.path.exists(os.path.join(face_path_notmask, man)):
            os.mkdir(os.path.join(face_path_notmask, man))

    for pic in pic_list:
        img_path = os.path.join(data_path, man, pic)
        save_txt_path_mask = os.path.join(mask_path_mask, man, pic.replace('.jpg', '.txt').replace('.png', '.txt'))
        save_mask_path_notmask = os.path.join(mask_path_notmask, man, pic)
        save_face_path_mask = os.path.join(face_path_mask, man, pic)
        save_face_path_notmask = os.path.join(face_path_notmask, man, pic)
        preprocess(img_path, save_face_path_mask, save_txt_path_mask, save_face_path_notmask, \
                   save_mask_path_notmask, detector, predictor, img_size,masked, notmasked)

# with open(os.path.join(pwd, f"Datasets{os.sep}classid.json"), "w") as f:
#     f.write(json.dumps(jsonfile, ensure_ascii=False, indent=4, separators=(',', ':')))